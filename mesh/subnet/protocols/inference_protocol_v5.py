from __future__ import annotations

import asyncio
import io
import multiprocessing as mp
from typing import AsyncIterator, Optional

import torch

import mesh
from mesh import DHT, get_dht_time
from mesh.compression.serialization import deserialize_torch_tensor, serialize_torch_tensor
from mesh.p2p import P2P, P2PContext, PeerID, ServicerBase
from mesh.proto import dht_pb2, inference_protocol_pb2, runtime_pb2
from mesh.subnet.protocols.inference_model import AsyncInferenceServer, InferenceModel
from mesh.subnet.utils.consensus import get_consensus_key
from mesh.subnet.utils.key import extract_rsa_peer_id, extract_rsa_peer_id_from_ssh
from mesh.substrate.chain_functions_v2 import Hypertensor
from mesh.utils import get_logger
from mesh.utils.asyncio import switch_to_uvloop
from mesh.utils.auth import AuthorizerBase, AuthRole, AuthRPCWrapperStreamer
from mesh.utils.mpfuture import MPFuture
from mesh.utils.serializer import MSGPackSerializer

logger = get_logger(__name__)


class InferenceProtocol(mp.context.ForkProcess, ServicerBase):

    _async_model: AsyncInferenceServer

    def __init__(
        self,
        dht: DHT,
        subnet_id: int,
        model_name: Optional[str] = None,
        balanced: bool = True,
        shutdown_timeout: float = 3,
        hypertensor: Optional[Hypertensor] = None,
        authorizer: Optional[AuthorizerBase] = None,
        start: bool = False,
    ):
        super().__init__()
        self.dht = dht
        self.subnet_id = subnet_id
        self.peer_id = dht.peer_id
        self.node_id = dht.node_id
        self.node_info = dht_pb2.NodeInfo(node_id=self.node_id.to_bytes())
        self.balanced, self.shutdown_timeout = balanced, shutdown_timeout
        self._p2p = None
        self.authorizer = authorizer
        self.ready = MPFuture()
        self.rpc_semaphore = asyncio.Semaphore(float("inf"))
        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=True)
        self.model_name = model_name
        self.daemon = True
        self.hypertensor = hypertensor

        self._current_consensus_tensor = None # the current epochs random tensor for hosters to validate
        self._current_consensus_tensor_expiration = None # when we re-query for new tensors

        if start:
            self.run_in_background(await_ready=True)

    def run(self):
        torch.set_num_threads(1)
        loop = switch_to_uvloop()
        stop = asyncio.Event()
        loop.add_reader(self._inner_pipe.fileno(), stop.set)

        async def _run():
            try:
                self._p2p = await self.dht.replicate_p2p()
                """Add rpc_* methods from this class to the P2P servicer"""
                logger.info("Adding P2P handlers")
                if self.authorizer is not None:
                    logger.info("Adding P2P handlers with authorizer")
                    await self.add_p2p_handlers(
                        self._p2p,
                        AuthRPCWrapperStreamer(self, AuthRole.SERVICER, self.authorizer),
                    )
                else:
                    await self.add_p2p_handlers(self._p2p, balanced=self.balanced)

                """
                Run pytorch functions and classes in the child process
                Read more:
                    - https://stackoverflow.com/questions/22950047/cuda-initialization-error-after-fork/22950549#22950549
                    - https://github.com/pytorch/pytorch/issues/17199
                """

                if self.model_name is not None:
                    logger.info("Loading Inference Model")
                    model = InferenceModel(self.model_name)
                    logger.info("Setting Up Async Inference Server")
                    self._async_model = AsyncInferenceServer(model)
                    logger.info("Starting Async Inference Server Worker")
                    asyncio.create_task(self._async_model._worker())
                    logger.info("Async Inference Server Complete")

                self.ready.set_result(None)
            except Exception as e:
                logger.debug(e, exc_info=True)
                self.ready.set_exception(e)

            try:
                await stop.wait()
            finally:
                await self.remove_p2p_handlers(self._p2p)

        try:
            loop.run_until_complete(_run())
        except KeyboardInterrupt:
            logger.debug("Caught KeyboardInterrupt, shutting down")

    def run_in_background(self, await_ready: bool = True, timeout: Optional[float] = None) -> None:
        """
        Starts InferenceProtocol in a background process. If :await_ready:, this method will wait until
        it is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()

    def shutdown(self):
        if self.is_alive():
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning(
                    "InferenceProtocol did not shut down within the grace period; terminating it the hard way"
                )
                self.terminate()
        else:
            logger.warning("InferenceProtocol shutdown had no effect, the process is already dead")

    def get_stub(self, p2p: P2P, peer: PeerID) -> AuthRPCWrapperStreamer:
        """
        Get a stub that sends requests to a given peer.

        It's important here to wrap the stub with an authentication wrapper, see AuthRPCWrapper
        """
        stub = super().get_stub(p2p, peer)
        return AuthRPCWrapperStreamer(stub, AuthRole.CLIENT, self.authorizer, service_public_key=None)

    @classmethod
    def get_server_stub(
        cls,
        p2p: P2P,
        peer: PeerID,
        authorizer: Optional[AuthorizerBase] = None
    ) -> "InferenceProtocolStub":  # type: ignore # noqa: F821
        """
        Get a stub that sends requests to a given peer.

        This function can be used to get the RPC methods from this protocol outside of this class.

        This is useful for client-side requests.
        """

        stub = super().get_stub(p2p, peer)
        return AuthRPCWrapperStreamer(stub, AuthRole.CLIENT, authorizer, service_public_key=None)

    async def rpc_info(self, request: runtime_pb2.Empty, context: P2PContext) -> runtime_pb2.NodeData:
        """Return metadata about stored block uids and current load"""

        result = {
            "version": mesh.__version__,
            "dht_client_mode": self.dht.client_mode,
            "role": "hoster" if self.model_name is not None else "validator"
        }

        return runtime_pb2.NodeData(serialized_info=MSGPackSerializer.dumps(result))

    async def call_inference_stream(
        self, peer: PeerID, prompt: str, tensor: torch.Tensor
    ) -> AsyncIterator[torch.Tensor]:
        """
        Call another peer to perform an inference stream on the `tensor`

        The inference will be returned as a streamed
        """
        input_stream = inference_protocol_pb2.InferenceRequestAuth(
            input=prompt,
            max_new_tokens=5,
            tensor=serialize_torch_tensor(tensor),
        )

        try:
            async with self.rpc_semaphore:
                p2p = await self.dht.replicate_p2p()
                response_stream = await self.get_stub(p2p, peer).rpc_inference_stream(input_stream)
                async for response in response_stream:
                    for tensor_bytes in response.tensors:
                        tensor = deserialize_torch_tensor(tensor_bytes)
                        yield tensor
        except Exception as e:
            logger.error(f"InferenceProtocol failed to stream from {peer}: {e}", exc_info=True)
            return

    def should_process_inference(self, tensor: torch.Tensor) -> bool:
        """
        Ensures inference request doesn't match the current epochs random prompt

        This ensure peers/hosters cannot call inference on other hosters to use
        to copy.
        """
        try:
            dht_time = get_dht_time()
            if self._current_consensus_tensor_expiration is not None and dht_time < self._current_consensus_tensor_expiration:
                return True
            else:
                epoch_data = self.hypertensor.get_epoch_progress()
                curr_epoch = epoch_data.epoch
                seconds_remaining = epoch_data.seconds_remaining

                validator = self.hypertensor.get_elected_validator_node(self.subnet_id, curr_epoch)
                validator_peer_id = validator["peer_id"]

                key = get_consensus_key(curr_epoch)
                results = self.dht.get(key)

                if results is None:
                    return True

                self._current_consensus_tensor_expiration = dht_time + seconds_remaining

                first_prompt = None
                validator_prompt = None
                for public_key, data in results.value.items():
                    value = data.value
                    if first_prompt is None:
                        first_prompt = value

                    peer_id = extract_rsa_peer_id(public_key)
                    if peer_id is not None and validator_peer_id == peer_id:
                        validator_prompt = value
                        break

                data = validator_prompt or first_prompt
                epoch_tensor = torch.load(io.BytesIO(data), weights_only=False)
                self._current_consensus_tensor = epoch_tensor
                return epoch_tensor != tensor
        except Exception as e:
            logger.warning(f"should_process_inference Err={e}")
            return True

    async def rpc_inference_stream(
        self, requests: inference_protocol_pb2.InferenceRequestAuth, context: P2PContext
    ) -> AsyncIterator[inference_protocol_pb2.InferenceResponseAuth]:
        """
        A peer wants us to perform an inference stream
        """
        tensor = deserialize_torch_tensor(requests.tensor)

        caller_peer_id = extract_rsa_peer_id_from_ssh(requests.auth.client_access_token.public_key)
        """
        Don't allow other hosters to call inference on me if it matches
        the current epochs random consensus tensors
        """
        if self.authorizer is not None and not caller_peer_id.__eq__(self.peer_id):
            # Don't bother pinging the decentralized storage unless we have to
            run_inference = self.should_process_inference(tensor)
            if run_inference is False:
                raise ValueError("Tensor must not match the current validation tensor.")

        async for token_tensor in await self._async_model.submit(tensor):
            yield inference_protocol_pb2.InferenceResponseAuth(
                peer=self.node_info,
                dht_time=get_dht_time(),
                output=str(token_tensor.item()),
                tensors=[serialize_torch_tensor(token_tensor)]
            )
