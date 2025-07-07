import asyncio

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mesh.subnet.protocols.inference_model import AsyncInferenceServer, InferenceModel

# FLAKY
# pytest tests/test_inference_model.py -rP

# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME = "bigscience/bloom-560m"

@pytest.fixture(scope="module")
def model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    inference_model = InferenceModel(MODEL_NAME)
    return inference_model

# pytest tests/test_inference_model.py::test_async_inference_stream_basic -rP
# pytest tests/test_inference_model.py::test_async_inference_stream_basic --log-cli-level=DEBUG

@pytest.mark.asyncio
async def test_async_inference_stream_basic(model):
    server = AsyncInferenceServer(model)
    await server.start()

    prompt = "<|user|>Hello, how are you?</s>"
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids

    stream = await server.submit(input_ids, priority=5)
    tokens = []
    async for token in stream:
        tokens.append(token.item())

    assert len(tokens) > 0, "No tokens were streamed back"

# pytest tests/test_inference_model.py::test_priority_queue_order -rP

@pytest.mark.asyncio
async def test_priority_queue_order(model):
    server = AsyncInferenceServer(model)
    await server.start()

    prompt1 = "<|user|>Hello, how are you?</s>"
    prompt2 = "<|user|>A cat sat</s>"

    input1 = model.tokenizer(prompt1, return_tensors="pt").input_ids.to(model.model.device)
    input2 = model.tokenizer(prompt2, return_tensors="pt").input_ids.to(model.model.device)

    stream1_future = asyncio.create_task(server.submit(input1, priority=10))  # low priority
    stream2_future = asyncio.create_task(server.submit(input2, priority=1))   # high priority

    stream2 = await stream2_future
    stream1 = await stream1_future

    out2, out1 = [], []
    async for t in stream2: out2.append(t.item())
    async for t in stream1: out1.append(t.item())

    assert len(out1) > 0 and len(out2) > 0
    assert out2 != out1  # just to ensure both streams are distinct and processed

