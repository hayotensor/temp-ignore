from __future__ import annotations

import asyncio
import itertools
import random
import time
import uuid
from typing import AsyncIterator, List, Optional, Tuple

import torch

from mesh import P2P, PeerID, deserialize_torch_tensor, get_logger, serialize_torch_tensor
from mesh.proto import inference_protocol_pb2, runtime_pb2
from mesh.subnet.client.config import ClientConfig
from mesh.subnet.client.routing.inference_manager_v2 import RemoteManager
from mesh.subnet.data_structures import ServerState
from mesh.subnet.protocols.inference_protocol_v4 import InferenceProtocol
from mesh.utils.auth import AuthorizerBase

logger = get_logger(__name__)

# class _ServerInferenceSession:
#     def __init__(
#         self,
#         config: ClientConfig,
#         # inputs_queue: asyncio.Queue,
#         # outputs_aiter: AsyncIterator,
#         *,
#         max_length: int,
#         **metadata,
#     ):
#         self.config = config
#         # self._inputs_queue: asyncio.Queue[runtime_pb2.ExpertRequest] = inputs_queue
#         # self._outputs_stream: AsyncIterator[runtime_pb2.ExpertResponse] = outputs_aiter
#         self.session_id = str(uuid.uuid4())
#         self.session_metadata = dict(max_length=max_length, **metadata)
#         self.closed = False

#         self.history = None  # Used in case of server failures to regenerate attention caches on new servers

#     @classmethod
#     async def create(
#         cls,
#         config: ClientConfig,
#         p2p: P2P,
#         peer_id: PeerID,
#         **metadata,
#     ) -> _ServerInferenceSession:
#         """Create a new session for a given remote module. This code is meant to be run inside RemoteExpertWorker"""
#         stub = InferenceProtocol.get_server_stub(p2p, peer_id)
#         inputs_queue = asyncio.Queue()
#         outputs_stream = await asyncio.wait_for(
#             stub.rpc_inference(cls._read_inputs_from_queue(inputs_queue)),
#             config.connect_timeout,
#         )
#         return cls(config, inputs_queue, outputs_stream, **metadata)

#     async def inference(
#         prompt: str,
#         tensor: torch.Tensor,
#         stub
#     ):
#         input_stream = runtime_pb2.InferenceRequestTest(
#             input=prompt, max_new_tokens=5, tensor=serialize_torch_tensor(tensor)
#         )

#         async with asyncio.Semaphore(float("inf")):
#             response_stream = await stub.rpc_inference_stream(input_stream)
#             async for response in response_stream:
#                 for tensor_bytes in response.tensors:
#                     tensor = deserialize_torch_tensor(tensor_bytes)
#                     print(tensor)

class InferenceSession:
    """
    An interface to call inference on a peer hoster
    """

    def __init__(
        self,
        remote_manager: RemoteManager,
        max_length: int,
        authorizer: Optional[AuthorizerBase] = None,
    ):
        self._remote_manager = remote_manager
        self._server_session = None
        self.server = None
        self._closed = False
        self._max_length = max_length
        self.history: List[Tuple[str, str]] = []  # (role, content) e.g., ("user", "Hello")
        self.past_key_values = None
        self.authorizer = authorizer

    # def create_session(self, user_id: str = None) -> str:
    #     """Create a new conversation session"""
    #     session_id = str(uuid.uuid4())
    #     self.sessions[session_id] = ConversationSession(user_id=user_id)
    #     return session_id

    async def inference(
        self,
        prompt: str,
        tensor: torch.Tensor,
        peer_id: PeerID
    ):

        if self.server is None:
            self._update_server(0)

        active_servers = [
            info.peer_id
            for info in self._remote_manager.state.remote_servers_infos.server_infos
            if info.server.state == ServerState.ONLINE
        ]

        random_peer_id = random.choice(active_servers)

        stub = InferenceProtocol.get_server_stub(
            self._remote_manager.state.p2p,
            random_peer_id,
            self.authorizer
        )

        # input_stream = runtime_pb2.InferenceRequestTest(
        #     input=prompt, max_new_tokens=5, tensor=serialize_torch_tensor(tensor)
        # )
        input_stream = inference_protocol_pb2.InferenceRequestAuth(
            input=prompt,
            max_new_tokens=5,
            tensor=serialize_torch_tensor(tensor),
        )

        try:
            async with asyncio.Semaphore(float("inf")):
                response_stream = await stub.rpc_inference_stream(input_stream)
                async for response in response_stream:
                    for tensor_bytes in response.tensors:
                        tensor = deserialize_torch_tensor(tensor_bytes)
                        print(tensor)
        except Exception as e:
            logger.error(f"test_inference3 failed to stream from {random_peer_id}: {e}", exc_info=True)
            return

    async def inference_v2(
        self,
        prompt: str,
        tensor: torch.Tensor,
        max_retries: Optional[int] = 4,
    ) -> AsyncIterator[torch.Tensor]:
        for attempt_no in itertools.count():
            try:
                server_session = None
                # Fetch new server if they don't exist
                if not self._server_session or attempt_no >= 1:
                    # Get new server if None of fails
                    self._update_server(attempt_no)

                # Update session to the current server
                server_session = self._server_session

                # Fetch server stub to call inference on
                stub = InferenceProtocol.get_server_stub(
                    self._remote_manager.state.p2p,
                    server_session.peer_id,
                    self.authorizer
                )

                # input_stream = runtime_pb2.InferenceRequestTest(
                #     input=prompt, max_new_tokens=5, tensor=serialize_torch_tensor(tensor)
                # )
                input_stream = inference_protocol_pb2.InferenceRequestAuth(
                    input=prompt,
                    max_new_tokens=5,
                    tensor=serialize_torch_tensor(tensor),
                )

                async with asyncio.Semaphore(float("inf")):
                    response_stream = await stub.rpc_inference_stream(input_stream)
                    async for response in response_stream:
                        for tensor_bytes in response.tensors:
                            tensor = deserialize_torch_tensor(tensor_bytes)
                            yield tensor
                return
            except Exception as e:
                self._remote_manager.on_request_failure(
                    server_session.peer_id if server_session is not None else None
                )
                if attempt_no + 1 == self._remote_manager.config.max_retries or attempt_no + 1 >= max_retries:
                    raise
                delay = self._remote_manager.get_retry_delay(attempt_no)
                logger.warning(
                    f"Caught exception when running inference via {server_session.peer_id if server_session is not None else None} "
                    f"(retry in {delay:.0f} sec): {repr(e)}"
                )
                time.sleep(delay)

    async def inference_v3(
        self,
        prompt: str,
        tensor: torch.Tensor,
        max_retries: Optional[int] = 4,
    ) -> AsyncIterator[torch.Tensor]:
        for attempt_no in itertools.count():
            try:
                server_session = None
                # Fetch new server if they don't exist
                if not self._server_session or attempt_no >= 1:
                    # Get new server if None of fails
                    self._update_server(attempt_no)

                # Update session to the current server
                server_session = self._server_session

                # Fetch server stub to call inference on
                stub = InferenceProtocol.get_server_stub(
                    self._remote_manager.state.p2p,
                    server_session.peer_id,
                    self.authorizer
                )

                # input_stream = runtime_pb2.InferenceRequestTest(
                #     input=prompt, max_new_tokens=5, tensor=serialize_torch_tensor(tensor)
                # )
                input_stream = inference_protocol_pb2.InferenceRequestAuth(
                    input=prompt,
                    max_new_tokens=5,
                    tensor=serialize_torch_tensor(tensor),
                )

                async with asyncio.Semaphore(float("inf")):
                    response_stream = await stub.rpc_inference_stream(input_stream)
                    async for response in response_stream:
                        for tensor_bytes in response.tensors:
                            tensor = deserialize_torch_tensor(tensor_bytes)
                            yield tensor
                return
            except Exception as e:
                self._remote_manager.on_request_failure(
                    server_session.peer_id if server_session is not None else None
                )
                if attempt_no + 1 == self._remote_manager.config.max_retries or attempt_no + 1 >= max_retries:
                    raise
                delay = self._remote_manager.get_retry_delay(attempt_no)
                logger.warning(
                    f"Caught exception when running inference via {server_session.peer_id if server_session is not None else None} "
                    f"(retry in {delay:.0f} sec): {repr(e)}"
                )
                time.sleep(delay)

    def _update_server(self, attempt_no: int):
        if attempt_no >= 1:
            logger.debug("Server failure")

        new_server_session = self._remote_manager.make_sequence()
        print(f"_update_server new_server_session: {new_server_session}")
        self._server_session = new_server_session

    async def close(self):
        if self._closed:
            return
        self._closed = True
