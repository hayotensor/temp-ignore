import asyncio
import os
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from mesh.dht.crypto import RSASignatureValidator
from mesh.subnet.protocols.inference_protocol_v4 import InferenceProtocol
from mesh.subnet.utils.key import (
    generate_rsa_private_key_file,
    get_rsa_peer_id,
    get_rsa_private_key,
)
from mesh.utils.auth import TokenRSAAuthorizerBase
from mesh.utils.logging import get_logger

from test_utils.dht_swarms import (
    launch_dht_instances_with_record_validators,
    launch_dht_instances_with_record_validators_and_authorizers,
)

logger = get_logger(__name__)

# pytest tests/test_inference_protocol_v4.py -rP



# converted_model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
converted_model_name_or_path = "bigscience/bloom-560m"

# pytest tests/test_inference_protocol_v4.py::test_call_inference_stream -rP

# pytest tests/test_inference_protocol_v4.py::test_call_inference_stream --log-cli-level=DEBUG

@pytest.mark.forked
@pytest.mark.asyncio
async def test_call_inference_stream():
    peers_len = 2

    test_paths = []
    record_validators: List[RSASignatureValidator] = []
    for i in range(peers_len):
        test_path = f"rsa_test_path_{i}.key"
        test_paths.append(test_path)
        private_key, public_key, public_bytes, encoded_public_key, encoded_digest, peer_id = generate_rsa_private_key_file(test_path)
        loaded_key = get_rsa_private_key(test_path)
        record_validator = RSASignatureValidator(loaded_key)
        record_validators.append(record_validator)
        peer_id = get_rsa_peer_id(public_bytes)

    dhts = launch_dht_instances_with_record_validators(
        record_validators=record_validators,
        identity_paths=test_paths,
    )

    hoster_dht = dhts[0]
    hoster_peer_id = hoster_dht.peer_id
    validator_dht = dhts[1]
    validator_peer_id = validator_dht.peer_id

    hoster_inference_protocol = InferenceProtocol(
        dht=hoster_dht,
        model_name=converted_model_name_or_path,
        start=True
    )
    validator_inference_protocol = InferenceProtocol(
        dht=validator_dht,
        start=True
    )

    outputs = []
    tokenizer = AutoTokenizer.from_pretrained(converted_model_name_or_path)
    prompt = "<|user|>How are you?</s>"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    async def run_stream_test():
        async for tensor in validator_inference_protocol.call_inference_stream(hoster_peer_id, prompt, inputs):
            outputs.append(tensor)

        assert len(outputs) > 0, "Tensor stream is 0 length"

    await asyncio.sleep(1.0) # redundant

    await run_stream_test()

    print("outputs", outputs)

    if outputs:
        # Concatenate all the output token tensors
        output_tensor = torch.cat(outputs, dim=-1)  # shape: [1, N]
        decoded_text = tokenizer.decode(output_tensor.squeeze(), skip_special_tokens=True)
        print("Decoded output:", decoded_text)
    else:
        print("No tokens received.")

    for dht in dhts:
        dht.shutdown()

    hoster_inference_protocol.shutdown()
    # validator_inference_protocol.shutdown()

    for path in test_paths:
        os.remove(path)

# pytest tests/test_inference_protocol_v4.py::test_call_inference_stream_with_authorizer -rP

# pytest tests/test_inference_protocol_v4.py::test_call_inference_stream_with_authorizer --log-cli-level=DEBUG

@pytest.mark.forked
@pytest.mark.asyncio
async def test_call_inference_stream_with_authorizer():
    peers_len = 2

    test_paths = []
    record_validators: List[RSASignatureValidator] = []
    token_rsa_validators: List[TokenRSAAuthorizerBase] = []
    for i in range(peers_len):
        test_path = f"rsa_test_path_{i}.key"
        test_paths.append(test_path)
        private_key, public_key, public_bytes, encoded_public_key, encoded_digest, peer_id = generate_rsa_private_key_file(test_path)
        loaded_key = get_rsa_private_key(test_path)
        record_validator = RSASignatureValidator(loaded_key)
        record_validators.append(record_validator)
        token_rsa_validator = TokenRSAAuthorizerBase(loaded_key)
        token_rsa_validators.append(token_rsa_validator)
        peer_id = get_rsa_peer_id(public_bytes)

    # dhts = launch_dht_instances_with_record_validators_and_authorizers(
    #     record_validators=record_validators,
    #     authorizers=token_rsa_validators,
    #     identity_paths=test_paths,
    # )
    dhts = launch_dht_instances_with_record_validators(
        record_validators=record_validators,
        identity_paths=test_paths,
    )

    hoster_dht = dhts[0]
    hoster_peer_id = hoster_dht.peer_id
    hoster_rsa_validator = token_rsa_validators[0]
    validator_dht = dhts[1]
    validator_peer_id = validator_dht.peer_id
    validator_rsa_validator = token_rsa_validators[1]

    print("hoster_peer_id   ", hoster_peer_id)
    print("validator_peer_id", validator_peer_id)

    hoster_inference_protocol = InferenceProtocol(
        dht=hoster_dht,
        model_name=converted_model_name_or_path,
        authorizer=hoster_rsa_validator,
        start=True
    )
    validator_inference_protocol = InferenceProtocol(
        dht=validator_dht,
        authorizer=validator_rsa_validator,
        start=True
    )

    outputs = []
    tokenizer = AutoTokenizer.from_pretrained(converted_model_name_or_path)
    prompt = "<|user|>How are you?</s>"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    async def run_stream_test():
        async for tensor in validator_inference_protocol.call_inference_stream(hoster_peer_id, prompt, inputs):
            outputs.append(tensor)

        assert len(outputs) > 0, "Tensor stream is 0 length"

    await asyncio.sleep(1.0) # redundant

    await run_stream_test()

    print("outputs", outputs)

    if outputs:
        # Concatenate all the output token tensors
        output_tensor = torch.cat(outputs, dim=-1)  # shape: [1, N]
        decoded_text = tokenizer.decode(output_tensor.squeeze(), skip_special_tokens=True)
        print("Decoded output:", decoded_text)
    else:
        print("No tokens received.")

    for dht in dhts:
        dht.shutdown()

    hoster_inference_protocol.shutdown()
    validator_inference_protocol.shutdown()

    for path in test_paths:
        os.remove(path)

# pytest tests/test_inference_protocol_v4.py::test_call_inference_stream -rP

# pytest tests/test_inference_protocol_v4.py::test_call_inference_stream --log-cli-level=DEBUG

@pytest.mark.forked
@pytest.mark.asyncio
async def test_call_inference_stream_on_self():
    peers_len = 2

    test_paths = []
    record_validators: List[RSASignatureValidator] = []
    for i in range(peers_len):
        test_path = f"rsa_test_path_{i}.key"
        test_paths.append(test_path)
        private_key, public_key, public_bytes, encoded_public_key, encoded_digest, peer_id = generate_rsa_private_key_file(test_path)
        loaded_key = get_rsa_private_key(test_path)
        record_validator = RSASignatureValidator(loaded_key)
        record_validators.append(record_validator)
        peer_id = get_rsa_peer_id(public_bytes)

    dhts = launch_dht_instances_with_record_validators(
        record_validators=record_validators,
        identity_paths=test_paths,
    )

    hoster_dht = dhts[0]
    hoster_peer_id = hoster_dht.peer_id

    hoster_inference_protocol = InferenceProtocol(
        dht=hoster_dht,
        model_name=converted_model_name_or_path,
        start=True
    )

    outputs = []
    tokenizer = AutoTokenizer.from_pretrained(converted_model_name_or_path)
    prompt = "<|user|>How are you?</s>"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    async def run_stream_test():
        async for tensor in hoster_inference_protocol.call_inference_stream(hoster_peer_id, prompt, inputs):
            outputs.append(tensor)

        assert len(outputs) > 0, "Tensor stream is 0 length"

    await asyncio.sleep(1.0) # redundant

    await run_stream_test()

    print("outputs", outputs)

    if outputs:
        # Concatenate all the output token tensors
        output_tensor = torch.cat(outputs, dim=-1)  # shape: [1, N]
        decoded_text = tokenizer.decode(output_tensor.squeeze(), skip_special_tokens=True)
        print("Decoded output:", decoded_text)
    else:
        print("No tokens received.")

    for dht in dhts:
        dht.shutdown()

    hoster_inference_protocol.shutdown()
    # validator_inference_protocol.shutdown()

    for path in test_paths:
        os.remove(path)
