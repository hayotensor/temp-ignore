from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional, Union

import torch
from torch.multiprocessing import Pipe, Process

from mesh.compression.serialization import deserialize_torch_tensor, serialize_torch_tensor
from mesh.dht.dht import DHT
from mesh.p2p import P2P, P2PContext, PeerID, ServicerBase
from mesh.proto import runtime_pb2
from mesh.subnet.protocols.inference_model import AsyncInferenceServer, InferenceModel
from mesh.utils import get_logger
from mesh.utils.asyncio import switch_to_uvloop
from mesh.utils.auth import AuthorizerBase, AuthRole, AuthRPCWrapper
from mesh.utils.mpfuture import MPFuture

logger = get_logger(__name__)


class InferenceProtocol(Process, ServicerBase):

    _async_model: AsyncInferenceServer

    def __init__(
        self,
        dht: DHT,
        model_name: Optional[str] = None,
        balanced: bool = True,
        shutdown_timeout: float = 3,
        authorizer: Optional[AuthorizerBase] = None,
        start: bool = False,
    ):
        super().__init__()
        self.dht = dht
        self.balanced, self.shutdown_timeout = balanced, shutdown_timeout
        self._p2p = None
        self.authorizer = authorizer
        self.ready = MPFuture()
        self.rpc_semaphore = asyncio.Semaphore(float("inf"))
        self._inner_pipe, self._outer_pipe = Pipe(duplex=False)
        self.model_name = model_name
        self.daemon = True

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

    def get_stub(self, p2p: P2P, peer: PeerID) -> AuthRPCWrapper:
        """
        Get a stub that sends requests to a given peer.

        It's important here to wrap the stub with an authentication wrapper, see AuthRPCWrapper
        """

        stub = super().get_stub(p2p, peer)
        return AuthRPCWrapper(stub, AuthRole.CLIENT, self.authorizer, service_public_key=None)

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
        return AuthRPCWrapper(stub, AuthRole.CLIENT, authorizer, service_public_key=None)

    async def call_inference_stream(
        self, peer: PeerID, prompt: str, tensor: torch.Tensor
    ) -> AsyncIterator[torch.Tensor]:
        """
        Call another peer to perform an inference stream on the `tensor`

        The inference will be returned as a streamed
        """
        input_stream = runtime_pb2.InferenceRequestTest(
            input=prompt, max_new_tokens=5, tensor=serialize_torch_tensor(tensor)
        )

        try:
            async with self.rpc_semaphore:
                p2p = await self.dht.replicate_p2p()
                response_stream = await self.get_stub(p2p, peer).rpc_inference_stream(input_stream)
                logger.debug(f"InferenceProtocol call_inference_stream response_stream: {response_stream}")
                async for response in response_stream:
                    for tensor_bytes in response.tensors:
                        tensor = deserialize_torch_tensor(tensor_bytes)
                        yield tensor
        except Exception as e:
            logger.error(f"InferenceProtocol failed to stream from {peer}: {e}", exc_info=True)
            return

    async def rpc_inference_stream(
        self, requests: runtime_pb2.InferenceRequestTest, context: P2PContext
    ) -> AsyncIterator[runtime_pb2.InferenceResponse]:
        """
        A peer wants us to perform an inference stream
        """
        print("rpc_inference_stream")
        logger.info("rpc_inference_stream")
        tensor = deserialize_torch_tensor(requests.tensor)
        print(f"rpc_inference_stream tensor {tensor}")
        logger.info(f"rpc_inference_stream tensor {tensor}")

        async for token_tensor in await self._async_model.submit(tensor):
            yield runtime_pb2.InferenceResponse(
                output=str(token_tensor.item()), tensors=[serialize_torch_tensor(token_tensor)]
            )
