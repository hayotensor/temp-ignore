import asyncio
import hashlib
import io
import os
import random
from dataclasses import dataclass
from typing import Dict, List
from unittest.mock import MagicMock

import pytest
import torch

from mesh import PeerID, get_dht_time
from mesh.dht.crypto import RSASignatureValidator
from mesh.dht.validation import HypertensorPredicateValidator, RecordValidatorBase
from mesh.subnet.consensus import Consensus
from mesh.subnet.data_structures import ServerClass
from mesh.subnet.roles.hoster import Hoster
from mesh.subnet.roles.validator import Validator
from mesh.subnet.utils.consensus import MAX_CONSENSUS_TIME, get_consensus_key
from mesh.subnet.utils.hoster import get_hoster_commit_key, get_hoster_reveal_key
from mesh.subnet.utils.key import (
    extract_rsa_peer_id,
    generate_rsa_private_key_file,
    get_rsa_peer_id,
    get_rsa_private_key,
)
from mesh.subnet.utils.validator import (
    get_validator_commit_key,
    get_validator_reveal_key,
)
from mesh.substrate.chain_data import SubnetNode
from mesh.substrate.chain_functions_v2 import EpochData
from mesh.substrate.config import BLOCK_SECS

from test_utils.dht_swarms import (
    launch_dht_instances_with_record_validators,
    launch_dht_instances_with_record_validators2,
)
from test_utils.hypertensor_predicate_v3 import hypertensor_consensus_predicate
from test_utils.mock_hypertensor_json import MockHypertensor, increase_progress_and_write, write_epoch_json

# pytest tests/test_consensus.py -rP

class DummyInferenceProtocol:
    async def rpc_inference_stream(self, tensor):
        return tensor * 2  # fake inference

# pytest tests/test_consensus.py::test_consensus -rP
# pytest tests/test_consensus.py::test_consensus --log-cli-level=DEBUG

"""
Simulate 2 epochs in a row
"""
@pytest.mark.forked
@pytest.mark.asyncio
async def test_consensus():
    hypertensor = MockHypertensor()

    # start at commit phase 0%
    block_per_epoch = 100
    seconds_per_epoch = BLOCK_SECS * block_per_epoch
    current_block = 100
    epoch_length = 100
    epoch = current_block // epoch_length
    blocks_elapsed = current_block % epoch_length
    percent_complete = blocks_elapsed / epoch_length
    blocks_remaining = epoch_length - blocks_elapsed
    seconds_elapsed = blocks_elapsed * BLOCK_SECS
    seconds_remaining = blocks_remaining * BLOCK_SECS

    write_epoch_json({
        "block": current_block,
        "epoch": epoch,
        "block_per_epoch": block_per_epoch,
        "seconds_per_epoch": seconds_per_epoch,
        "percent_complete": percent_complete,
        "blocks_elapsed": blocks_elapsed,
        "blocks_remaining": blocks_remaining,
        "seconds_elapsed": seconds_elapsed,
        "seconds_remaining": seconds_remaining
    })

    hosters_length = 3
    validators_length = 3

    peers_len = 10
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
        identity_paths=test_paths
    )

    all_peer_ids = []
    for i, dht in enumerate(dhts):
        if i + 1 == len(dhts):
            announce_maddrs = dht.get_visible_maddrs()
            peer_id = str(announce_maddrs[0]).split("/p2p/")[-1]
            all_peer_ids.append(dht.peer_id)
            all_peer_ids.append(PeerID(peer_id))
        else:
            all_peer_ids.append(dht.peer_id)

    for record_validator in record_validators:
        peer_id = extract_rsa_peer_id(record_validator.local_public_key)

    # Update elected validator to hoster
    hypertensor.get_formatted_elected_validator_node = lambda subnet_id, epoch: SubnetNode(
        id=1,
        hotkey="0x1234567890abcdef1234567890abcdef12345678",
        peer_id=dhts[0].peer_id,
        bootstrap_peer_id="QmNV5G3hq2UmAck2htEgsqrmPFBff5goFZAdmKDcZLBZLX",
        client_peer_id="QmNV5G3hq2UmAck2htEgsqrmPFBff5goFZAdmKDcZLBZLX",
        classification="Validator",
        delegate_reward_rate=0,
        last_delegate_reward_rate_update=0,
        a=None,
        b=None,
        c=None,
    )

    validator = hypertensor.get_formatted_elected_validator_node(1, epoch)
    assert validator is not None

    hosters: List[Hoster] = []
    hoster_peer_ids = []
    for i in range(0, hosters_length):
        hoster_peer_ids.append(dhts[i].peer_id)
        hoster = Hoster(
            dht=dhts[i],
            subnet_id=1,
            inference_protocol=DummyInferenceProtocol(),
            record_validator=record_validators[i],
            hypertensor=hypertensor,
        )
        hosters.append(hoster)

    consensuses: List[Consensus] = []
    validators: List[Validator] = []
    validator_peer_ids = []
    for i in range(hosters_length, hosters_length + validators_length):
        validator_peer_ids.append(dhts[i].peer_id)
        validator = Validator(
            role=ServerClass.VALIDATOR,
            dht=dhts[i],
            record_validator=record_validators[i],
            hypertensor=hypertensor,
        )
        validators.append(validator)

        consensus = Consensus(
          dht=dhts[i],
          subnet_id=1,
          subnet_node_id=i,
          role=ServerClass.VALIDATOR,
          record_validator=record_validators[i],
          hypertensor=hypertensor,
          converted_model_name_or_path="bigscience/bloom-560m",
          validator=validator,
          hoster=None,
          start=False,
        )
        consensuses.append(consensus)

    someone = random.choice(dhts)
    """
    Step 1:

    SIMULATE `await self.run_consensus(current_epoch)` IN `async def run_forever(self)`

    Store random input tensor as the chosen validator
    """
    consensus_key = get_consensus_key(epoch)
    tensor = torch.randn(2, 2)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    value = buffer.read()

    dhts[0].store(consensus_key, value, get_dht_time() + 999, subkey=record_validators[0].local_public_key),
    results2 = someone.get(consensus_key)
    assert results2 is not None
    payload = next(iter(results2.value.values())).value
    assert payload == value, "Incorrect value in payload. "

    """
    Step 2:

    SIMULATE `await self.hoster.run()` IN `async def run_forever(self)`

    Run inference on prompt and commit hash of outputs
    """
    for i in range(0, hosters_length):
        hoster = hosters[i]
        # --- Verify try_load_tensor retrieves the same tensor ---
        loaded = await hoster.try_load_tensor(consensus_key, epoch)
        assert loaded is not None
        assert torch.allclose(loaded, tensor)

        # --- Run inference via DummyInferenceProtocol ---
        result = await hoster.inference_protocol.rpc_inference_stream(loaded)
        expected_result = tensor * 2
        assert torch.allclose(result, expected_result)

        # --- COMMIT PHASE ---
        hoster.commit(epoch, result)

        # --- REVEAL PHASE ---
        hoster.reveal(epoch, result)

    commit_key = get_hoster_commit_key(epoch)

    commit_records = someone.get(commit_key)
    assert len(commit_records.value) == len(hosters)

    for subkey, value in commit_records.value.items():
        peer_id = extract_rsa_peer_id(subkey)
        assert any(peer_id == _peer_id for _peer_id in all_peer_ids)

    reveal_key = get_hoster_reveal_key(epoch)

    commit_records = validators[0].dht.get(commit_key)
    assert commit_records is not None

    reveal_records = validators[0].dht.get(reveal_key)
    assert reveal_records is not None

    """
    Step 3:

    (Skipping `await self.hoster.reveal(current_epoch)`, this is the first epoch)

    SIMULATE `await self.validator.score_nodes(current_epoch)` IN `async def run_forever(self)`
    """
    validator_scores = {}
    for i in range(0, validators_length):
        scores = await validators[i].score_nodes(epoch)
        print("scores", scores)
        assert scores is not None or len(scores) == len(hosters)
        validator_scores[i] = scores

    """
    Step 4:

    SIMULATE `await self.validator.commit(current_epoch)` IN `async def run_forever(self)`
    """
    for i in range(0, validators_length):
        scores = validator_scores[i]
        validators[i].commit(scores, epoch)
        assert scores is not None

    validator_commit_key = get_validator_commit_key(epoch)
    validator_commit_records = someone.get(validator_commit_key)
    assert validator_commit_records is not None and len(validator_commit_records.value) == len(validators)



    """
    **********
    ==========
    NEXT EPOCH
    ==========
    **********
    """
    block_per_epoch = 100
    seconds_per_epoch = BLOCK_SECS * block_per_epoch
    current_block = 200
    epoch_length = 100
    epoch = current_block // epoch_length
    blocks_elapsed = current_block % epoch_length
    percent_complete = blocks_elapsed / epoch_length
    blocks_remaining = epoch_length - blocks_elapsed
    seconds_elapsed = blocks_elapsed * BLOCK_SECS
    seconds_remaining = blocks_remaining * BLOCK_SECS

    write_epoch_json({
        "block": current_block,
        "epoch": epoch,
        "block_per_epoch": block_per_epoch,
        "seconds_per_epoch": seconds_per_epoch,
        "percent_complete": percent_complete,
        "blocks_elapsed": blocks_elapsed,
        "blocks_remaining": blocks_remaining,
        "seconds_elapsed": seconds_elapsed,
        "seconds_remaining": seconds_remaining
    })

    epoch = hypertensor.get_epoch()
    assert epoch == 2

    """
    Step 1:
    """
    consensus_key = get_consensus_key(epoch)
    tensor = torch.randn(2, 2)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    value = buffer.read()

    dhts[0].store(consensus_key, value, get_dht_time() + 999, subkey=record_validators[0].local_public_key),
    results = someone.get(consensus_key)
    assert results is not None
    payload = next(iter(results.value.values())).value
    assert payload == value, "Incorrect value in payload. "

    """
    Step 2:
    """
    for i in range(0, hosters_length):
        hoster = hosters[i]
        # --- Verify try_load_tensor retrieves the same tensor ---
        loaded = await hoster.try_load_tensor(consensus_key, epoch)
        assert loaded is not None
        assert torch.allclose(loaded, tensor)

        # --- Run inference via DummyInferenceProtocol ---
        result = await hoster.inference_protocol.rpc_inference_stream(loaded)
        expected_result = tensor * 2
        assert torch.allclose(result, expected_result)

        # --- COMMIT PHASE ---
        hoster.commit(epoch, result)

        # --- REVEAL PHASE ---
        hoster.reveal(epoch, result)

    commit_key = get_hoster_commit_key(epoch)

    commit_records = someone.get(commit_key)
    assert len(commit_records.value) == len(hosters)

    for subkey, value in commit_records.value.items():
        peer_id = extract_rsa_peer_id(subkey)
        assert any(peer_id == _peer_id for _peer_id in all_peer_ids)

    reveal_key = get_hoster_reveal_key(epoch)

    commit_records = validators[0].dht.get(commit_key)
    assert commit_records is not None

    reveal_records = validators[0].dht.get(reveal_key)
    assert reveal_records is not None

    """
    Step 3.0:

    Reveal the previous epochs scores commits
    """
    for i in range(0, validators_length):
        assert validators[i].latest_commit is not None
        await validators[i].reveal(epoch)
        assert validators[i].latest_commit is None

    validator_reveal_key = get_validator_reveal_key(epoch)
    validator_reveal_records = someone.get(validator_reveal_key)
    assert validator_reveal_records is not None

    """
    Step 3.1:
    """
    validator_scores = {}
    for i in range(0, validators_length):
        scores = await validators[i].score_nodes(epoch)
        print("epoch 2 scores", scores)
        assert scores is not None and len(scores) == len(hosters) + len(validators)
        validator_scores[i] = scores

    """
    Step 4:
    """
    for i in range(0, validators_length):
        scores = validator_scores[i]
        validators[i].commit(scores, epoch)
        assert scores is not None

    validator_commit_key = get_validator_commit_key(epoch)
    validator_commit_records = someone.get(validator_commit_key)
    assert len(validator_commit_records.value) == len(validators)

    for path in test_paths:
        os.remove(path)

    for dht in dhts:
        dht.shutdown()


# pytest tests/test_consensus.py::test_consensus_with_key_validator -rP
# pytest tests/test_consensus.py::test_consensus_with_key_validator --log-cli-level=DEBUG

@pytest.mark.forked
@pytest.mark.asyncio
async def test_consensus_with_key_validator():
    predicate = hypertensor_consensus_predicate()
    hypertensor = MockHypertensor()
    # start at commit phase 0%

    block_per_epoch = 100
    seconds_per_epoch = BLOCK_SECS * block_per_epoch
    current_block = 100
    epoch_length = 100
    epoch = current_block // epoch_length
    blocks_elapsed = current_block % epoch_length
    percent_complete = blocks_elapsed / epoch_length
    blocks_remaining = epoch_length - blocks_elapsed
    seconds_elapsed = blocks_elapsed * BLOCK_SECS
    seconds_remaining = blocks_remaining * BLOCK_SECS

    write_epoch_json({
        "block": current_block,
        "epoch": epoch,
        "block_per_epoch": block_per_epoch,
        "seconds_per_epoch": seconds_per_epoch,
        "percent_complete": percent_complete,
        "blocks_elapsed": blocks_elapsed,
        "blocks_remaining": blocks_remaining,
        "seconds_elapsed": seconds_elapsed,
        "seconds_remaining": seconds_remaining
    })

    peers_len = 10
    test_paths = []
    record_validators: List[List[RecordValidatorBase]] = []
    for i in range(peers_len):
        test_path = f"rsa_test_path_{i}.key"
        test_paths.append(test_path)
        private_key, public_key, public_bytes, encoded_public_key, encoded_digest, peer_id = generate_rsa_private_key_file(test_path)
        loaded_key = get_rsa_private_key(test_path)
        record_validator = RSASignatureValidator(loaded_key)
        consensus_predicate = HypertensorPredicateValidator(
            hypertensor=hypertensor,
            record_predicate=predicate,
        )
        record_validators.append([record_validator, consensus_predicate])
        peer_id = get_rsa_peer_id(public_bytes)

    dhts = launch_dht_instances_with_record_validators2(
        record_validators=record_validators,
        identity_paths=test_paths
    )

    all_peer_ids = []
    for i, dht in enumerate(dhts):
        if i + 1 == len(dhts):
            announce_maddrs = dht.get_visible_maddrs()
            peer_id = str(announce_maddrs[0]).split("/p2p/")[-1]
            all_peer_ids.append(dht.peer_id)
            all_peer_ids.append(PeerID(peer_id))
        else:
            all_peer_ids.append(dht.peer_id)

    hosters_length = 3
    validators_length = 3

    epoch = hypertensor.get_epoch()
    assert epoch == 1

    # Update elected validator to hoster
    hypertensor.get_formatted_elected_validator_node = lambda subnet_id, epoch: SubnetNode(
        id=1,
        hotkey="0x1234567890abcdef1234567890abcdef12345678",
        peer_id=dhts[0].peer_id,
        bootstrap_peer_id="QmNV5G3hq2UmAck2htEgsqrmPFBff5goFZAdmKDcZLBZLX",
        client_peer_id="QmNV5G3hq2UmAck2htEgsqrmPFBff5goFZAdmKDcZLBZLX",
        classification="Validator",
        delegate_reward_rate=0,
        last_delegate_reward_rate_update=0,
        a=None,
        b=None,
        c=None,
    )

    validator = hypertensor.get_formatted_elected_validator_node(1, epoch)
    assert validator is not None

    hosters: List[Hoster] = []
    hoster_peer_ids = []
    for i in range(0, hosters_length):
        hoster_peer_ids.append(dhts[i].peer_id)
        hoster = Hoster(
            dht=dhts[i],
            subnet_id=1,
            inference_protocol=DummyInferenceProtocol(),
            record_validator=record_validators[i][0],
            hypertensor=hypertensor,
        )
        hosters.append(hoster)

    consensuses: List[Consensus] = []
    validators: List[Validator] = []
    validator_peer_ids = []
    for i in range(hosters_length, hosters_length + validators_length):
        validator_peer_ids.append(dhts[i].peer_id)
        validator = Validator(
            role=ServerClass.VALIDATOR,
            dht=dhts[i],
            record_validator=record_validators[i][0],
            hypertensor=hypertensor,
        )
        validators.append(validator)

        consensus = Consensus(
          dht=dhts[i],
          subnet_id=1,
          subnet_node_id=i,
          role=ServerClass.VALIDATOR,
          record_validator=record_validators[i][0],
          hypertensor=hypertensor,
          converted_model_name_or_path="bigscience/bloom-560m",
          validator=validator,
          hoster=None,
          start=False,
        )
        consensuses.append(consensus)

    someone = random.choice(dhts)

    """
    Step 1: ⸺ 00-15%

    SIMULATE `await self.run_consensus(current_epoch)` IN `async def run_forever(self)`

    Store random input tensor as the chosen validator
    """
    _max_consensus_time = MAX_CONSENSUS_TIME - 60

    consensus_key = get_consensus_key(epoch)
    tensor = torch.randn(2, 2)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    value = buffer.read()

    store_ok = dhts[0].store(consensus_key, value, get_dht_time() + _max_consensus_time, subkey=record_validators[0][0].local_public_key)
    assert store_ok is True

    results2 = someone.get(consensus_key)
    assert results2 is not None
    payload = next(iter(results2.value.values())).value
    assert payload == value, "Incorrect value in payload. "

    """
    Step 2: ⸺ 15-50%

    SIMULATE `await self.hoster.run()` IN `async def run_forever(self)`

    Run inference on prompt and commit hash of outputs
    """
    increase_progress_and_write(0.16)

    hoster_tensors = {}
    hoster_commits = {}
    for i in range(0, hosters_length):
        hoster = hosters[i]
        # --- Verify try_load_tensor retrieves the same tensor ---
        loaded = await hoster.try_load_tensor(consensus_key, epoch)
        assert loaded is not None
        assert torch.allclose(loaded, tensor)

        # --- Run inference via DummyInferenceProtocol ---
        result = await hoster.inference_protocol.rpc_inference_stream(loaded)
        expected_result = tensor * 2
        assert torch.allclose(result, expected_result)
        hoster_tensors[i] = result

        # --- COMMIT PHASE ---
        commit_result = hoster.commit(epoch, result)
        assert commit_result is not None
        hoster_commits[i] = commit_result

        assert isinstance(commit_result, bytes) and len(commit_result) == hashlib.sha256().digest_size


    commit_key = get_hoster_commit_key(epoch)

    commit_records = someone.get(commit_key)
    assert len(commit_records.value) == len(hosters)

    """
    Step 3: ⸺ 50-60%

    SIMULATE `await self.hoster.run()` IN `async def run_forever(self)`

    Run inference on prompt and commit hash of outputs
    """
    increase_progress_and_write(0.51)

    for i in range(0, hosters_length):
        hoster = hosters[i]
        # --- REVEAL PHASE ---
        tensor = hoster_tensors[i]
        reveal_result = hoster.reveal(epoch, tensor)
        assert reveal_result is True

    commit_key = get_hoster_commit_key(epoch)

    commit_records = someone.get(commit_key)
    assert len(commit_records.value) == len(hosters)

    for subkey, value in commit_records.value.items():
        peer_id = extract_rsa_peer_id(subkey)
        assert any(peer_id == _peer_id for _peer_id in all_peer_ids)

    reveal_key = get_hoster_reveal_key(epoch)

    commit_records = validators[0].dht.get(commit_key)
    assert commit_records is not None

    reveal_records = validators[0].dht.get(reveal_key)
    assert reveal_records is not None

    """
    Step 4.0: ⸺ 60-100%

    SIMULATE `await self.validator.score_nodes(current_epoch)` IN `async def run_forever(self)`

    Score nodes
    """
    validator_scores = {}
    for i in range(0, validators_length):
        scores = await validators[i].score_nodes(epoch)
        assert scores is not None or len(scores) == len(hosters)
        validator_scores[i] = scores

    increase_progress_and_write(0.60)
    """
    Step 4.1:

    SIMULATE `await self.validator.commit(scores, current_epoch)` IN `async def run_forever(self)`
    """
    for i in range(0, validators_length):
        scores = validator_scores[i]
        assert scores is not None
        result = validators[i].commit(scores, epoch)
        assert result is not None


    """
    **********
    ==========
    NEXT EPOCH
    ==========
    **********
    """
    block_per_epoch = 100
    seconds_per_epoch = BLOCK_SECS * block_per_epoch
    current_block = 200
    epoch_length = 100
    epoch = current_block // epoch_length
    blocks_elapsed = current_block % epoch_length
    percent_complete = blocks_elapsed / epoch_length
    blocks_remaining = epoch_length - blocks_elapsed
    seconds_elapsed = blocks_elapsed * BLOCK_SECS
    seconds_remaining = blocks_remaining * BLOCK_SECS

    write_epoch_json({
        "block": current_block,
        "epoch": epoch,
        "block_per_epoch": block_per_epoch,
        "seconds_per_epoch": seconds_per_epoch,
        "percent_complete": percent_complete,
        "blocks_elapsed": blocks_elapsed,
        "blocks_remaining": blocks_remaining,
        "seconds_elapsed": seconds_elapsed,
        "seconds_remaining": seconds_remaining
    })

    epoch = hypertensor.get_epoch()
    assert epoch == 2

    """
    Step 1:
    """
    consensus_key = get_consensus_key(epoch)
    tensor = torch.randn(2, 2)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    value = buffer.read()

    dht_time = get_dht_time()
    print("dht_time test local", dht_time)
    print("expiration_time test local", dht_time + 999)
    store_ok = dhts[0].store(consensus_key, value, get_dht_time() + _max_consensus_time, subkey=record_validators[0][0].local_public_key)
    assert store_ok is True
    results = someone.get(consensus_key)
    assert results is not None
    payload = next(iter(results.value.values())).value
    assert payload == value, "Incorrect value in payload. "

    """
    Step 2: ⸺ 15-50%

    SIMULATE `await self.hoster.run()` IN `async def run_forever(self)`

    Run inference on prompt and commit hash of outputs
    """
    increase_progress_and_write(0.16)

    hoster_tensors = {}
    hoster_commits = {}
    for i in range(0, hosters_length):
        hoster = hosters[i]
        # --- Verify try_load_tensor retrieves the same tensor ---
        loaded = await hoster.try_load_tensor(consensus_key, epoch)
        assert loaded is not None
        assert torch.allclose(loaded, tensor)

        # --- Run inference via DummyInferenceProtocol ---
        result = await hoster.inference_protocol.rpc_inference_stream(loaded)
        expected_result = tensor * 2
        assert torch.allclose(result, expected_result)
        hoster_tensors[i] = result

        # --- COMMIT PHASE ---
        commit_result = hoster.commit(epoch, result)
        assert commit_result is not None
        hoster_commits[i] = commit_result

        assert isinstance(commit_result, bytes) and len(commit_result) == hashlib.sha256().digest_size


    commit_key = get_hoster_commit_key(epoch)

    commit_records = someone.get(commit_key)
    assert len(commit_records.value) == len(hosters)

    """
    Step 3.0: ⸺ 50-60%

    SIMULATE `self.validator.run_reveal(current_epoch)` IN `async def run_forever(self)`

    Validator reveals previous epochs commits
    """
    increase_progress_and_write(0.51)

    for i in range(0, validators_length):
        scores = validator_scores[i]
        assert scores is not None
        result = await validators[i].reveal(epoch)
        assert result is not False and result is not None

    """
    Step 3.1: ⸺ 50-60%

    SIMULATE `await self.hoster.run()` IN `async def run_forever(self)`

    Run inference on prompt and commit hash of outputs
    """
    for i in range(0, hosters_length):
        hoster = hosters[i]
        # --- REVEAL PHASE ---
        tensor = hoster_tensors[i]
        reveal_result = hoster.reveal(epoch, tensor)
        assert reveal_result is True

    commit_key = get_hoster_commit_key(epoch)

    commit_records = someone.get(commit_key)
    assert len(commit_records.value) == len(hosters)

    for subkey, value in commit_records.value.items():
        peer_id = extract_rsa_peer_id(subkey)
        assert any(peer_id == _peer_id for _peer_id in all_peer_ids)

    reveal_key = get_hoster_reveal_key(epoch)

    commit_records = validators[0].dht.get(commit_key)
    assert commit_records is not None

    reveal_records = validators[0].dht.get(reveal_key)
    assert reveal_records is not None

    """
    Step 4.0: ⸺ 60-100%

    SIMULATE `await self.validator.score_nodes(current_epoch)` IN `async def run_forever(self)`

    Score nodes
    """
    increase_progress_and_write(0.60)

    validator_scores = {}
    for i in range(0, validators_length):
        scores = await validators[i].score_nodes(epoch)
        assert scores is not None and len(scores) == (len(hosters) + len(validators))
        validator_scores[i] = scores

    """
    Step 4.1:
    """
    for i in range(0, validators_length):
        scores = validator_scores[i]
        assert scores is not None
        result = validators[i].commit(scores, epoch)
        assert result is not None


    for path in test_paths:
        os.remove(path)

    for dht in dhts:
        dht.shutdown()

@dataclass
class ValidatorEntry:
    peer_id: str
    score: int


@pytest.fixture
def consensus():
    # Create a minimal Consensus object with mocked dependencies
    # consensus = Consensus(
    #     dht=MagicMock(),
    #     subnet_id=1,
    #     subnet_node_id=123,
    #     record_validator=MagicMock(),
    #     hypertensor=MockHypertensor(),
    #     model_name_or_path="bigscience/bloom-560m"
    # )
    consensus = Consensus(
        dht=MagicMock(),
        subnet_id=1,
        subnet_node_id=1,
        role=ServerClass.VALIDATOR,
        record_validator=MagicMock(),
        hypertensor=MockHypertensor(),
        converted_model_name_or_path="bigscience/bloom-560m",
        validator=MagicMock(),
        hoster=None,
        start=False,
    )

    # Add mock storage for previous validator submissions and attestation percentages
    consensus.mock_validator_submissions = {}
    consensus.mock_attestation_percentages = {}

    # Patch methods that access Substrate chain
    consensus._get_validator_consensus_submission = lambda epoch: consensus.mock_validator_submissions.get(epoch)
    consensus._get_reward_result = lambda epoch: (None, consensus.mock_attestation_percentages.get(epoch, 1e18))

    return consensus


# pytest tests/test_consensus.py::test_compare_exact_match -rP

def test_compare_exact_match(consensus):
    my_data = [{"peer_id": "abc", "score": 1}]
    validator_data = [ValidatorEntry("abc", 1)]
    assert consensus.compare_consensus_data(my_data, validator_data, epoch=1)


def test_empty_data(consensus):
    assert consensus.compare_consensus_data([], [], 1) is True


def test_mismatch_data_score(consensus):
    validator_data = [ValidatorEntry("123", 1)]
    my_data = [{"peer_id": "123", "score": 2}]
    assert consensus.compare_consensus_data(my_data, validator_data, 1) is False


def test_extra_local_node(consensus):
    validator_data = [ValidatorEntry("123", 1)]
    my_data = [{"peer_id": "123", "score": 1}, {"peer_id": "456", "score": 1}]
    assert consensus.compare_consensus_data(my_data, validator_data, 1) is False


def test_missing_node_on_attestor(consensus):
    validator_data = [ValidatorEntry("123", 1), ValidatorEntry("456", 1), ValidatorEntry("789", 1)]
    my_data = [{"peer_id": "123", "score": 1}, {"peer_id": "456", "score": 1}]
    assert consensus.compare_consensus_data(my_data, validator_data, 1) is False


def test_node_joins_after_validator_submission(consensus):
    validator_data = [ValidatorEntry("123", 1), ValidatorEntry("456", 1)]
    my_data = [{"peer_id": "123", "score": 1}, {"peer_id": "456", "score": 1}, {"peer_id": "789", "score": 1}]
    assert consensus.compare_consensus_data(my_data, validator_data, 1) is False


def test_previous_epoch_validates_inclusion(consensus):
    validator_data = [ValidatorEntry("123", 1)]
    my_data = []
    consensus.previous_epoch_data = {frozenset({"peer_id": "123", "score": 1}.items())}
    assert consensus.compare_consensus_data(my_data, validator_data, 2) is True


def test_validator_missing_node_but_previous_epoch_has_it(consensus):
    validator_data = [ValidatorEntry("123", 1)]
    my_data = [{"peer_id": "123", "score": 1}, {"peer_id": "456", "score": 1}]
    consensus.previous_epoch_data = {
        frozenset({"peer_id": "123", "score": 1}.items()),
        frozenset({"peer_id": "456", "score": 1}.items()),
    }
    assert consensus.compare_consensus_data(my_data, validator_data, 2) is True

# pytest tests/test_consensus.py::test_compare_with_previous_validator_submission -rP

def test_compare_with_previous_validator_submission(consensus):
    # This simulates the fallback check to previous validator submission with high attestation
    my_data = []
    validator_data = [ValidatorEntry("abc", 1)]
    consensus.mock_validator_submissions[0] = [ValidatorEntry("abc", 1)]
    consensus.mock_attestation_percentages[1] = 0.9e18  # >66%
    consensus.previous_epoch_data = None
    assert consensus.compare_consensus_data(my_data, validator_data, epoch=1)

# pytest tests/test_consensus.py::test_compare_fails_due_to_low_attestation -rP

def test_compare_fails_due_to_low_attestation(consensus):
    my_data = []
    validator_data = [ValidatorEntry("abc", 1)]
    consensus.mock_validator_submissions[0] = [ValidatorEntry("abc", 1)]
    consensus.mock_attestation_percentages[1] = 0.5e18  # <66%
    consensus.previous_epoch_data = None
    assert not consensus.compare_consensus_data(my_data, validator_data, epoch=1)


def test_invalid_node_not_in_previous_epoch(consensus):
    validator_data = [ValidatorEntry("123", 1), ValidatorEntry("456", 1)]
    my_data = [{"peer_id": "456", "score": 1}]
    consensus.previous_epoch_data = {frozenset({"peer_id": "456", "score": 1}.items())}
    assert consensus.compare_consensus_data(my_data, validator_data, 2) is False


def test_symmetric_difference_with_previous_epoch(consensus):
    validator_data = [ValidatorEntry("123", 1), ValidatorEntry("456", 1)]
    my_data = [{"peer_id": "456", "score": 1}]
    consensus.previous_epoch_data = {frozenset({"peer_id": "123", "score": 1}.items())}
    assert consensus.compare_consensus_data(my_data, validator_data, 2) is True

def test_compare_with_previous_epoch_fix(consensus):
    # mismatch, but should succeed due to previous epoch data
    my_data = [{"peer_id": "abc", "score": 1}]
    validator_data = [ValidatorEntry("def", 1)]
    consensus.previous_epoch_data = {
        frozenset({"peer_id": "abc", "score": 1}.items()),
        frozenset({"peer_id": "def", "score": 1}.items()),
    }
    assert consensus.compare_consensus_data(my_data, validator_data, epoch=1)
