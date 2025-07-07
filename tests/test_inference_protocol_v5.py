import asyncio
import io
import os
from typing import List
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoTokenizer

from mesh import get_dht_time
from mesh.dht.crypto import RSASignatureValidator
from mesh.subnet.protocols.inference_protocol_v5 import InferenceProtocol
from mesh.subnet.utils.consensus import get_consensus_key, get_consensus_subkey_rsa
from mesh.subnet.utils.key import (
    generate_rsa_private_key_file,
    get_rsa_peer_id,
    get_rsa_private_key,
)
from mesh.substrate.chain_functions_v2 import EpochData
from mesh.substrate.config import BLOCK_SECS
from mesh.utils.auth import TokenRSAAuthorizerBase
from mesh.utils.logging import get_logger

from test_utils.dht_swarms import (
    launch_dht_instances_with_record_validators,
    launch_dht_instances_with_record_validators_and_authorizers,
)

logger = get_logger(__name__)

# pytest tests/test_inference_protocol_v4.py -rP

class MockHypertensor:
    interface = None  # only used for mock epoch length

    def get_epoch_length(self):
        return 10

    def get_block_number(self):
        return 1

    def get_epoch(self):
        return 1

    def get_elected_validator_node(self, subnet_id: int, epoch: int):
        ...

    def get_epoch_progress(self) -> EpochData:
        current_block = self.get_block_number()
        epoch_length = self.get_epoch_length()
        epoch = current_block // epoch_length
        blocks_elapsed = current_block % epoch_length
        percent_complete = blocks_elapsed / epoch_length
        blocks_remaining = epoch_length - blocks_elapsed
        seconds_elapsed = blocks_elapsed * BLOCK_SECS
        seconds_remaining = blocks_remaining * BLOCK_SECS

        return EpochData(
            block=current_block,
            epoch=epoch,
            block_per_epoch=epoch_length,
            seconds_per_epoch=epoch_length * BLOCK_SECS,
            percent_complete=percent_complete,
            blocks_elapsed=blocks_elapsed,
            blocks_remaining=blocks_remaining,
            seconds_elapsed=seconds_elapsed,
            seconds_remaining=seconds_remaining
        )

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
        subnet_id=1,
        model_name=converted_model_name_or_path,
        start=True
    )
    validator_inference_protocol = InferenceProtocol(
        dht=validator_dht,
        subnet_id=1,
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

# pytest tests/test_inference_protocol_v5.py::test_call_inference_stream_raise_tensor_matches_consensus_tensor -rP

# pytest tests/test_inference_protocol_v5.py::test_call_inference_stream_raise_tensor_matches_consensus_tensor --log-cli-level=DEBUG

@pytest.mark.forked
@pytest.mark.asyncio
async def test_call_inference_stream_raise_tensor_matches_consensus_tensor():
    peers_len = 5

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

    dhts = launch_dht_instances_with_record_validators(
        record_validators=record_validators,
        identity_paths=test_paths,
    )

    hoster_dht = dhts[0]
    hoster_peer_id = dhts[0].peer_id

    validator_dht = dhts[1]
    node_1_peer_id = dhts[1].peer_id

    node_2_dht = dhts[2]
    node_2_peer_id = dhts[2].peer_id

    node_3_dht = dhts[3]
    node_3_peer_id = dhts[3].peer_id

    node_4_dht = dhts[4]
    node_4_peer_id = dhts[4].peer_id

    hoster_rsa_validator = token_rsa_validators[0]
    validator_rsa_validator = token_rsa_validators[1]

    hypertensor = MockHypertensor()

    # Submit a random tensor acting as this epochs validator
    tensor = torch.randn(3, 3)
    epoch = 1
    hypertensor.get_epoch = Mock(return_value=epoch)


    # --- Store the consensus tensor into the DHT ---
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    consensus_prompt_tensor = buffer.read()

    key = get_consensus_key(epoch)
    for i, dht in enumerate(dhts):
        if i == 1:
            # Use the 1 node as the current elected validator node
            mock_subnet_node = {
                "id": i,
                "hotkey": "0x1234567890abcdef1234567890abcdef12345678",
                "peer_id": dht.peer_id.to_base58(),
                "bootstrap_peer_id": b"",
                "client_peer_id": b"",
                "classification": "Validator",  # or the enum name as a string
                "delegate_reward_rate": 1000000000000,
                "last_delegate_reward_rate_update": 123456789,
                "a": None,
                "b": None,
                "c": None
            }
            hypertensor.get_elected_validator_node = Mock(return_value=mock_subnet_node)

        # Every node sends prompts to the DHT
        subkey = get_consensus_subkey_rsa(record_validators[i])
        dht.store(
            key,
            consensus_prompt_tensor,
            get_dht_time() + 999,
            subkey
        )

    hoster_inference_protocol = InferenceProtocol(
        dht=hoster_dht,
        subnet_id=1,
        model_name=converted_model_name_or_path,
        hypertensor=hypertensor,
        authorizer=hoster_rsa_validator,
        start=True
    )
    validator_inference_protocol = InferenceProtocol(
        dht=validator_dht,
        subnet_id=1,
        authorizer=validator_rsa_validator,
        hypertensor=hypertensor,
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

    await run_stream_test()

    async def run_stream_test_raise():
        with pytest.raises(ValueError):
            async for tensor in validator_inference_protocol.call_inference_stream(hoster_peer_id, consensus_prompt_tensor, inputs):
                outputs.append(tensor)

    await run_stream_test_raise()

    for dht in dhts:
        dht.shutdown()

    hoster_inference_protocol.shutdown()
    validator_inference_protocol.shutdown()

    for path in test_paths:
        os.remove(path)
