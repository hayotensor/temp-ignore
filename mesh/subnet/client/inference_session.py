from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from typing import AsyncIterator, List, Optional, Tuple

import torch

from mesh import MSGPackSerializer, anext, deserialize_torch_tensor, get_logger, serialize_torch_tensor
from mesh.p2p import P2P
from mesh.proto import runtime_pb2
from mesh.p2p.servicer import ServicerBase
from mesh.subnet.client.config import ClientConfig
from mesh.subnet.client.routing import RemoteSequenceManager, maybe_log_traceback
from mesh.subnet.client.routing.inference_manager import RemoteManager
from mesh.subnet.data_structures import CHAIN_DELIMITER, ModuleUID, RemoteInfo, RPCInfo
from mesh.subnet.utils.misc import DUMMY, DUMMY_INT64, is_dummy
from mesh.subnet.utils.packaging import pack_args_kwargs
from mesh.utils.remote_worker import RemoteWorker
from mesh.utils.tensor_descr import BatchTensorDescriptor

logger = get_logger(__name__)


class _ServerInferenceSession:
    """
    An interface to a single multi-step *inference* session for a a set of blocks on a specific server.

    :note: This class is *not* fault-tolerant out of the box.
    """

    def __init__(
        self,
        config: ClientConfig,
        span: RemoteInfo,
        uid: ModuleUID,
        rpc_info: RPCInfo,
        inputs_queue: asyncio.Queue,
        outputs_aiter: AsyncIterator,
        *,
        max_length: int,
        **metadata,
    ):
        self.config = config
        self.span, self.uid, self.rpc_info = span, uid, rpc_info
        self._inputs_queue: asyncio.Queue[runtime_pb2.ExpertRequest] = inputs_queue
        self._outputs_stream: AsyncIterator[runtime_pb2.ExpertResponse] = outputs_aiter
        self.session_id = str(uuid.uuid4())
        self.session_metadata = dict(max_length=max_length, **metadata)
        self.stepped = False
        self.closed = False

        self.history = None  # Used in case of server failures to regenerate attention caches on new servers
        self.next_session = None

    @classmethod
    async def create(
        cls,
        config: ClientConfig,
        p2p: P2P,
        info: RemoteInfo,
        uid: ModuleUID,
        rpc_info: RPCInfo,
        **metadata,
    ) -> _ServerInferenceSession:
        """Create a new session for a given remote module. This code is meant to be run inside RemoteWorker"""
        stub = ServicerBase.get_stub(p2p, info.peer_id)
        inputs_queue = asyncio.Queue()
        outputs_stream = await asyncio.wait_for(
            stub.rpc_inference(cls._read_inputs_from_queue(inputs_queue)),
            config.connect_timeout,
        )
        return cls(config, info, uid, rpc_info, inputs_queue, outputs_stream, **metadata)

    @staticmethod
    async def _read_inputs_from_queue(queue: asyncio.Queue, input_timeout: Optional[float] = None) -> AsyncIterator:
        while True:
            next_input_message = await asyncio.wait_for(queue.get(), input_timeout)
            yield next_input_message
            if not next_input_message.uid and not next_input_message.tensors:
                break  # this message means "done sending"

    def step(
        self,
        inputs: torch.Tensor,
        prompts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference step: send a chunk of input tensors and receive a chunk of outputs
        :prompts: optional DEEP prompts, added to a prefix of each layer's outputs,
          if specified, deep prompts should have shape [num_layers, batch_size, prefix_len, hid_size]
        """
        if self.closed:
            raise Exception("Session is closed, cannot perform step")

        outputs_serialized = RemoteWorker.run_coroutine(
            self._step(
                runtime_pb2.ExpertRequest(
                    uid=self.uid,
                    tensors=[
                        serialize_torch_tensor(tensor.to(proto.dtype), proto.compression)
                        for tensor, proto in zip(input_tensors, inference_schema)
                    ],
                    metadata=MSGPackSerializer.dumps(request_metadata),
                )
            )
        )
        outputs = list(map(deserialize_torch_tensor, outputs_serialized.tensors))
        assert (
            outputs[0].shape == inputs.shape
        ), f"output activation shape is different from input shape: {outputs[0].shape} != {inputs.shape}"

        return outputs[0]

    async def _step(self, inputs_serialized: runtime_pb2.ExpertRequest) -> runtime_pb2.ExpertResponse:
        """Inference step on serialized data. This code is meant to be run inside RemoteWorker"""
        await self._inputs_queue.put(inputs_serialized)
        self.stepped = True
        return await asyncio.wait_for(anext(self._outputs_stream), self.config.request_timeout)

    def close(self):
        """Finish a given inference session, close the underlying connection"""
        if self._outputs_stream is None:
            return  # already closed
        RemoteWorker.run_coroutine(self._aclose_stream())
        self._outputs_stream = self._inputs_queue = None
        self.closed = True

    async def _aclose_stream(self):
        """Close the inference session. This code is meant to be run inside RemoteWorker"""
        if self._outputs_stream is None:
            return  # already closed
        if self.stepped:
            await self._inputs_queue.put(runtime_pb2.ExpertRequest())  # empty request will trigger end of session
            try:
                await anext(self._outputs_stream)
            except StopAsyncIteration:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        assert not self.closed
        return self

    def __exit__(self, *exc_details):
        self.close()


class InferenceSession:
    """
    An interface to a multi-step *inference* session for a sequence of remote transformer blocks
    """

    def __init__(self, sequence_manager: RemoteManager, max_length: int):
        self._sequence_manager = sequence_manager
        self._closed = False
        self._server_sessions = []
        self._position = 0
        self._max_length = max_length
        self.output_ids = None
        self.past_key_values = None

    @property
    def position(self) -> int:
        return self._position

    @position.setter
    def position(self, start_from_position: int) -> None:
        self._position = start_from_position
        for session in self._server_sessions:
            assert isinstance(session, _ServerInferenceSession)
            session.position = start_from_position

    def _enter_server_sessions(self, chosen_server: RemoteInfo) -> List[_ServerInferenceSession]:
        try:
            metadata = self._sequence_manager.get_request_metadata("rpc_inference", 0, peer_id=chosen_server.peer_id)
            session = RemoteWorker.run_coroutine(
                _ServerInferenceSession.create(
                    self._sequence_manager.config,
                    self._sequence_manager.state.p2p,
                    chosen_server,
                    0,
                    rpc_info=self._sequence_manager.rpc_info,
                    max_length=self._max_length,
                    **metadata,
                )
            )
            server_sessions.append(session)
            session.__enter__()
            return server_sessions
        except:
            self._exit_server_sessions(server_sessions)
            raise

    def _exit_server_sessions(self, server_sessions: List[_ServerInferenceSession]) -> None:
        for session in reversed(server_sessions):
            try:
                session.__exit__(None, None, None)
            except Exception:
                logger.debug("Caught exception while closing connection to server:", exc_info=True)

    def __enter__(self) -> "InferenceSession":
        assert not self._closed and not self._server_sessions
        return self

    def step(
        self,
        inputs: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        hypo_ids: Optional[torch.Tensor] = None,
        max_retries: Optional[int] = None,
    ) -> torch.Tensor:
        assert not self._closed
        if torch.is_grad_enabled():
            logger.warning("Running inference session with grad enabled. Gradients will *not* be propagated correctly.")

        if hypo_ids is None or is_dummy(hypo_ids):
            hypo_ids = DUMMY_INT64
        else:
            assert len(hypo_ids) == len(inputs)
            assert hypo_ids.dtype == torch.int64

        inputs_device = inputs.device
        inputs_dtype = inputs.dtype
        inputs = inputs.cpu()
        prompts = prompts.cpu()
        hypo_ids = hypo_ids.cpu()
        step_id = str(uuid.uuid4())

        n_input_tokens = inputs.shape[1]
        if self._position + n_input_tokens > self._max_length:
            raise ValueError(
                f"Maximum length exceeded: prefix {self._position} + current {n_input_tokens} exceeds pre-allocated maximum {self._max_length}"
            )

        server_idx = 0
        for attempt_no in itertools.count():
            logger.debug(f"Inference attempt {attempt_no}")
            server_session = None
            try:
                if not self._server_sessions or attempt_no >= 1:
                    self._update_sequence(server_idx, attempt_no)

                server_session = self._server_sessions[server_idx]

                assert server_session.position == self.position, f"{server_session.position} and {self.position}"
                inputs = server_session.step(
                    inputs,
                    prompts[server_session.span.start : server_session.span.end],
                    hypo_ids,
                    step_id=step_id,
                )

                server_idx += 1
                self._sequence_manager.on_request_success(server_session.span.peer_id)
                break
            except Exception as e:
                self._sequence_manager.on_request_failure(
                    server_session.span.peer_id if server_session is not None else None
                )
                if attempt_no + 1 == self._sequence_manager.config.max_retries or attempt_no + 1 >= max_retries:
                    raise
                delay = self._sequence_manager.get_retry_delay(attempt_no)
                logger.warning(
                    f"Caught exception when running inference via {server_session.span if server_session is not None else None} "
                    f"(retry in {delay:.0f} sec): {repr(e)}"
                )
                maybe_log_traceback(e)
                time.sleep(delay)

        self._position += n_input_tokens
        outputs = inputs[:, -n_input_tokens:]
        outputs = outputs.to(device=inputs_device, dtype=inputs_dtype)
        return outputs

    def _update_sequence(self, server_idx: int, attempt_no: int) -> int:
        # If there is a failed server session, this code closes it
        self._exit_server_sessions(self._server_sessions[server_idx : server_idx + 1])

        n_prev_spans = len(self._server_sessions)
        update_end = self._server_sessions[server_idx].span.end if server_idx < n_prev_spans else self.num_blocks
        if attempt_no >= 1:
            logger.debug(
                "Due to a server failure, remote attention caches "
            )

        updated_server = self._sequence_manager.make_sequence(
            cache_tokens_needed=self._max_length
        )
        # make_sequence() could return a longer sequence
        updated_sessions = self._enter_server_sessions(updated_server)
        logger.debug("Found server")

    def close(self, *exc_details):
        """Finish a given inference session, close the underlying connection"""
        if not self._closed:
            self._exit_server_sessions(self._server_sessions)
            self._server_sessions.clear()
            self._closed = True

    def __exit__(self, *exc_details):
        self.close(*exc_details)

    def __del__(self):
        self.close()

    @property
    def last_token_id(self) -> Optional[torch.Tensor]:  # Backward compatibility with Subnet < 2.1.0
        return self.output_ids[:, -1:] if self.output_ids is not None else None

    @last_token_id.setter
    def last_token_id(self, value: torch.Tensor):  # Backward compatibility with Subnet < 2.1.0
        if self.output_ids is None:
            raise RuntimeError("Can't override `last_token_id` since the session has not stepped yet")
        self.output_ids[:, -1:] = value
