"""
Utility functions that call RPC forward or backward on a single remote server
"""
import asyncio
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from subnet.client.config import ClientConfig
from subnet.data_structures import ModuleUID, RPCInfo

from mesh import nested_compare, nested_flatten, nested_pack, serialize_torch_tensor
from mesh.compression.serialization import deserialize_tensor_stream, deserialize_torch_tensor
from mesh.p2p import StubBase
from mesh.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, MAX_UNARY_PAYLOAD_SIZE
from mesh.proto import runtime_pb2
from mesh.utils.asyncio import aiter_with_timeout, iter_as_aiter
from mesh.utils.streaming import split_for_streaming
from mesh.utils.tensor_descr import BatchTensorDescriptor


async def _inference_unary(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, config: ClientConfig, **kwargs
) -> List[torch.Tensor]:
    outputs: runtime_pb2.ExpertResponse = await stub.rpc_forward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors), **kwargs),
        timeout=config.request_timeout,
    )
    return [deserialize_torch_tensor(t) for t in outputs.tensors]


async def _inference_stream(
    serialized_tensors: Iterable[runtime_pb2.InferenceRequestTest], stub, config: ClientConfig, **kwargs
) -> List[torch.Tensor]:
    input_stream = runtime_pb2.InferenceRequestTest(
        input="", max_new_tokens=5, tensor=serialized_tensors
    )

    try:
        async with asyncio.Semaphore(float("inf")):
            response_stream = await stub.rpc_inference_stream(input_stream)
            async for response in response_stream:
                for tensor_bytes in response.tensors:
                    tensor = deserialize_torch_tensor(tensor_bytes)
                    yield tensor
    except Exception as e:
        return

    # parts = (
    #     runtime_pb2.ExpertRequest(uid=uid, tensors=[part], **kwargs)
    #     for tensor in serialized_tensors
    #     for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
    # )
    # outputs = await asyncio.wait_for(stub.rpc_forward_stream(iter_as_aiter(parts)), config.connect_timeout)
    # outputs = aiter_with_timeout(outputs, config.request_timeout)
    # return await deserialize_tensor_stream(msg.tensors async for msg in outputs)


async def run_remote_inference(
    uid: ModuleUID,
    stub: StubBase,
    rpc_info: RPCInfo,
    *inputs: torch.Tensor,
    config: ClientConfig,
    metadata: Optional[bytes] = None,
    **kwargs,
) -> Tuple[torch.Tensor, ...]:
    # Asynchronous serialization
    # loop = asyncio.get_running_loop()
    # serialized_tensors = await asyncio.gather(
    #     *(
    #         loop.run_in_executor(None, serialize_torch_tensor, tensor.to(proto.dtype), proto.compression)
    #         for tensor, proto in zip(inputs, forward_schema)
    #     )
    # )

    input_stream = runtime_pb2.InferenceRequestTest(
        input="", max_new_tokens=5, tensor=serialize_torch_tensor(inputs)
    )

    await _inference_stream(input_stream, stub, config, **kwargs)


    # """
    # Serializes input tensors and calls "rpc_forward" on a remote server.
    # Mostly adapted from https://github.com/learning-at-home/hivemind/blob/7a7c93aefffc9494c39e7b170c07cb06d8c09c4c/hivemind/moe/client/expert.py#L198
    # but without RemoteExpertWorker.run_coroutine() call that leads to deadlock here.
    # """

    # # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
    # # detach to avoid pickling the computation graph
    # assert len(kwargs) == len(rpc_info["keyword_names"]), f"Keyword args should be {rpc_info['keyword_names']}"
    # kwargs = {key: kwargs[key] for key in rpc_info["keyword_names"]}

    # # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors
    # forward_inputs = tuple(nested_flatten((inputs, kwargs)))
    # args_schema, kwargs_schema = rpc_info["forward_schema"]
    # compression = args_schema[0].compression
    # forward_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in forward_inputs)
    # inputs = tuple(tensor.cpu().detach() for tensor in forward_inputs)
    # # TODO: create more explicit way to check servers schema and client's structure
    # assert len(inputs) >= len(args_schema) + 1, "Inputs and prompt tensors are necessary for a forward step"

    # # Asynchronous serialization
    # loop = asyncio.get_running_loop()
    # serialized_tensors = await asyncio.gather(
    #     *(
    #         loop.run_in_executor(None, serialize_torch_tensor, tensor.to(proto.dtype), proto.compression)
    #         for tensor, proto in zip(inputs, forward_schema)
    #     )
    # )

    # # call RPC on remote server
    # size = sum(t.element_size() * t.nelement() for t in inputs)
    # forward_fn = _forward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else _forward_unary
    # # Hotfix: we use "// 2" since hivemind==1.1.5 serializes bfloat16 tensors in float32, so they take 2x more space
    # deserialized_outputs = await forward_fn(uid, serialized_tensors, stub, config, metadata=metadata, **kwargs)
    # return nested_pack(deserialized_outputs, structure=rpc_info["outputs_schema"])
