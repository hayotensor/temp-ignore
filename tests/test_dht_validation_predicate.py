import asyncio
import io
import os
import random
import time
from typing import List

import pytest
import torch

import mesh
from mesh import get_dht_time
from mesh.dht.crypto import RSASignatureValidator
from mesh.dht.validation import HypertensorPredicateValidatorV2, RecordValidatorBase
from mesh.subnet.utils.consensus import MAX_CONSENSUS_TIME, get_consensus_key
from mesh.subnet.utils.key import generate_rsa_private_key_file, get_rsa_peer_id, get_rsa_private_key
from mesh.substrate.config import BLOCK_SECS

from test_utils.dht_swarms import launch_dht_instances_with_record_validators2
from test_utils.hypertensor_predicate_v3 import (
    hypertensor_consensus_predicate,
)
from test_utils.mock_hypertensor_json import MockHypertensor, write_epoch_json

# pytest tests/test_dht_validation_predicate.py -rP

# pytest tests/test_dht_validation_predicate.py::test_consensus_with_key_validator -rP

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

    time.sleep(5)

    peers_len = 10
    test_paths = []
    record_validators: List[List[RecordValidatorBase]] = []
    for i in range(peers_len):
        test_path = f"rsa_test_path_{i}.key"
        test_paths.append(test_path)
        private_key, public_key, public_bytes, encoded_public_key, encoded_digest, peer_id = generate_rsa_private_key_file(test_path)
        loaded_key = get_rsa_private_key(test_path)
        record_validator = RSASignatureValidator(loaded_key)
        consensus_predicate = HypertensorPredicateValidatorV2(
            hypertensor=hypertensor,
            record_predicate=predicate,
        )
        record_validators.append([record_validator, consensus_predicate])
        peer_id = get_rsa_peer_id(public_bytes)

    dhts = launch_dht_instances_with_record_validators2(
        record_validators=record_validators,
        identity_paths=test_paths
    )

    used_dhts = []
    used_dhts.append(dhts[0])

    _max_consensus_time = MAX_CONSENSUS_TIME - 60

    consensus_key = get_consensus_key(epoch)
    tensor = torch.randn(2, 2)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    value = buffer.read()

    store_ok = dhts[0].store(consensus_key, value, get_dht_time() + _max_consensus_time, subkey=record_validators[0][0].local_public_key)
    assert store_ok is True

    other_dhts = [dht for dht in dhts if dht not in used_dhts]
    assert other_dhts, "No other DHTs available. "

    someone = random.choice(other_dhts)
    used_dhts.append(someone)

    results = someone.get(consensus_key)
    assert results is not None
    payload = next(iter(results.value.values())).value
    assert payload == value, "Incorrect value in payload. "

    time.sleep(5)

    block_per_epoch = 100
    seconds_per_epoch = BLOCK_SECS * block_per_epoch
    current_block = 190
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

    other_dhts = [dht for dht in dhts if dht not in used_dhts]
    assert other_dhts, "No other DHTs available. "

    someone_new = random.choice(other_dhts)
    used_dhts.append(someone)

    epoch_progress = hypertensor.get_epoch_progress()
    print("epoch_progress", epoch_progress)

    results = someone_new.get(consensus_key)
    print("results", results)
    assert results is not None

    for dht in dhts:
        dht.shutdown()

    for path in test_paths:
        os.remove(path)

# # pytest tests/test_dht_validation_predicate.py::test_dht_hypertensor_predicate_roles -rP

# @pytest.mark.forked
# def test_dht_hypertensor_predicate_roles():
#     predicate = hypertensor_consensus_predicate()
#     hypertensor = MockHypertensor()
#     consensus_predicate = HypertensorPredicateValidator(
#         hypertensor=hypertensor,
#         record_predicate=predicate,
#     )
#     consensus_predicate.epoch_progress = EpochProgress(
#         current_block=1000,
#         epoch=10,
#         percent_complete=0.0,
#         blocks_remaining=0,
#         seconds_remaining=0,
#     )

#     # One app may create a DHT with its validators
#     dht = mesh.DHT(start=True, record_validators=[consensus_predicate])

#     assert dht.store("hoster", b"hoster-data", mesh.get_dht_time() + 10)
#     assert dht.get("hoster", latest=True).value == b"hoster-data"

#     assert dht.store("validator", b"validator-data", mesh.get_dht_time() + 10)
#     assert dht.get("validator", latest=True).value == b"validator-data"

# # pytest tests/test_dht_validation_predicate.py::test_dht_hypertensor_predicate_consensus -rP

# @pytest.mark.forked
# def test_dht_hypertensor_predicate_consensus():
#     predicate = hypertensor_consensus_predicate()
#     hypertensor = MockHypertensor()
#     block_per_epoch = 100
#     seconds_per_epoch = BLOCK_SECS * block_per_epoch
#     current_block = 100 # change block to increase progress
#     epoch_length = 100
#     epoch = current_block // epoch_length
#     blocks_elapsed = current_block % epoch_length
#     percent_complete = blocks_elapsed / epoch_length
#     blocks_remaining = epoch_length - blocks_elapsed
#     seconds_elapsed = blocks_elapsed * BLOCK_SECS
#     seconds_remaining = blocks_remaining * BLOCK_SECS

#     write_epoch_json({
#         "block": current_block,
#         "epoch": epoch,
#         "block_per_epoch": block_per_epoch,
#         "seconds_per_epoch": seconds_per_epoch,
#         "percent_complete": percent_complete,
#         "blocks_elapsed": blocks_elapsed,
#         "blocks_remaining": blocks_remaining,
#         "seconds_elapsed": seconds_elapsed,
#         "seconds_remaining": seconds_remaining
#     })

#     epoch = hypertensor.get_epoch()

#     consensus_predicate = HypertensorPredicateValidator(
#         hypertensor=hypertensor,
#         record_predicate=predicate,
#     )

#     # One app may create a DHT with its validators
#     dht = mesh.DHT(start=True, record_validators=[consensus_predicate])

#     key = f"consensus-epoch_{epoch}"
#     tensor = torch.randn(2, 3)
#     buffer = io.BytesIO()
#     torch.save(tensor, buffer)
#     serialized_tensor = buffer.getvalue()

#     assert dht.store(key, serialized_tensor, mesh.get_dht_time() + 10)

#     # Retrieve and deserialize to check it matches
#     result = dht.get(key, latest=True)
#     assert result is not None

#     loaded_tensor = torch.load(io.BytesIO(result.value))
#     assert torch.allclose(loaded_tensor, tensor)

#     # invalid tensor
#     not_tensor = [0, 1, 1]

#     assert not dht.store(key, bytes(not_tensor), mesh.get_dht_time() + 10)

#     # max time
#     assert not dht.store(key, serialized_tensor, mesh.get_dht_time() + 1000000)

#     # increase time
#     seconds_per_epoch = BLOCK_SECS * block_per_epoch
#     current_block = 190 # change block to increase progress
#     epoch_length = 100
#     epoch = current_block // epoch_length
#     blocks_elapsed = current_block % epoch_length
#     percent_complete = blocks_elapsed / epoch_length
#     blocks_remaining = epoch_length - blocks_elapsed
#     seconds_elapsed = blocks_elapsed * BLOCK_SECS
#     seconds_remaining = blocks_remaining * BLOCK_SECS

#     write_epoch_json({
#         "block": current_block,
#         "epoch": epoch,
#         "block_per_epoch": block_per_epoch,
#         "seconds_per_epoch": seconds_per_epoch,
#         "percent_complete": percent_complete,
#         "blocks_elapsed": blocks_elapsed,
#         "blocks_remaining": blocks_remaining,
#         "seconds_elapsed": seconds_elapsed,
#         "seconds_remaining": seconds_remaining
#     })

#     time.sleep(5.0)
#     print(hypertensor.get_epoch_progress(), "hypertensor.get_epoch_progress()")

#     # assert not dht.store(key, serialized_tensor, mesh.get_dht_time() + 10)

#     # Retrieve and deserialize to check it matches
#     result2 = dht.get(key, latest=True)
#     print("inference result2", result2)
#     assert result2 is not None

#     # loaded_tensor = torch.load(io.BytesIO(result.value))
#     # assert torch.allclose(loaded_tensor, tensor)

#     # # invalid tensor
#     # not_tensor = [0, 1, 1]

#     # assert not dht.store(key, bytes(not_tensor), mesh.get_dht_time() + 10)

# # pytest tests/test_dht_validation_predicate.py::test_dht_hypertensor_predicate_hoster -rP

# @pytest.mark.forked
# def test_dht_hypertensor_predicate_hoster():
#     predicate = hypertensor_consensus_predicate()
#     hypertensor = MockHypertensor()
#     # start at commit phase 15%
#     write_epoch_json({
#         "block": 10,
#         "epoch": 1,
#         "block_per_epoch": 10,
#         "seconds_per_epoch": 60,
#         "percent_complete": 0.16,
#         "blocks_elapsed": 0,
#         "blocks_remaining": 10,
#         "seconds_elapsed": 0,
#         "seconds_remaining": 60
#     })

#     epoch = hypertensor.get_epoch()

#     consensus_predicate = HypertensorPredicateValidator(
#         hypertensor=hypertensor,
#         record_predicate=predicate,
#     )

#     dht = mesh.DHT(start=True, record_validators=[consensus_predicate])

#     # hoster commit
#     hoster_commit_key = f"hoster-commit_epoch_{epoch}"

#     assert dht.store(hoster_commit_key, b"bytes_value", mesh.get_dht_time() + 10)
#     assert dht.get(hoster_commit_key, latest=True).value == b"bytes_value"

#     hoster_bad_commit_key = f"hoster-commit_epoch_{epoch+1}"

#     assert not dht.store(hoster_bad_commit_key, b"bytes_value", mesh.get_dht_time() + 10)
#     assert dht.get(hoster_bad_commit_key, latest=True) is None

#     # hoster reveal in commit phase
#     hoster_reveal_key = f"hoster-reveal_epoch_{epoch}"

#     assert not dht.store(hoster_reveal_key, b"bytes_value", mesh.get_dht_time() + 10)
#     assert dht.get(hoster_reveal_key, latest=True) is None

#     # increase to hoster reveal phase
#     write_epoch_json({
#         "block": 10,
#         "epoch": 1,
#         "block_per_epoch": 10,
#         "seconds_per_epoch": 60,
#         "percent_complete": 0.51,
#         "blocks_elapsed": 0,
#         "blocks_remaining": 10,
#         "seconds_elapsed": 0,
#         "seconds_remaining": 60
#     })

#     dht = mesh.DHT(start=True, record_validators=[consensus_predicate])

#     assert dht.store(hoster_reveal_key, b"bytes_value", mesh.get_dht_time() + 10)
#     assert dht.get(hoster_reveal_key, latest=True).value == b"bytes_value"

#     # hoster commit in reveal phase
#     assert not dht.store(hoster_commit_key, b"bytes_value", mesh.get_dht_time() + 10)
#     assert dht.get(hoster_commit_key, latest=True) is None

#     # max time
#     assert not dht.store(hoster_reveal_key, b"bytes_value", mesh.get_dht_time() + 1000000)


# # pytest tests/test_dht_validation_predicate.py::test_dht_hypertensor_predicate_validator -rP

# @pytest.mark.forked
# def test_dht_hypertensor_predicate_validator():
#     predicate = hypertensor_consensus_predicate()
#     hypertensor = MockHypertensor()
#     # start at commit phase 15%
#     write_epoch_json({
#         "block": 10,
#         "epoch": 1,
#         "block_per_epoch": 10,
#         "seconds_per_epoch": 60,
#         "percent_complete": 0.60,
#         "blocks_elapsed": 0,
#         "blocks_remaining": 10,
#         "seconds_elapsed": 0,
#         "seconds_remaining": 60
#     })

#     consensus_predicate = HypertensorPredicateValidator(
#         hypertensor=hypertensor,
#         record_predicate=predicate,
#     )

#     epoch = hypertensor.get_epoch()

#     dht = mesh.DHT(start=True, record_validators=[consensus_predicate])

#     # validator commit
#     validator_commit_key = f"validator-commit_epoch_{epoch}"

#     assert dht.store(validator_commit_key, b"bytes_value", mesh.get_dht_time() + 10)
#     assert dht.get(validator_commit_key, latest=True).value == b"bytes_value"

#     # cannot commit on anything other than the current epoch
#     validator_bad_commit_key = f"validator-commit_epoch_{11}"

#     assert not dht.store(validator_bad_commit_key, b"bytes_value", mesh.get_dht_time() + 10)
#     assert dht.get(validator_bad_commit_key, latest=True) is None

#     # must reveal on the next epoch - can't reveal on this one
#     validator_bad_reveal_key = f"validator-reveal_epoch_{epoch + 1}"

#     assert not dht.store(validator_bad_reveal_key, b"bytes_value", mesh.get_dht_time() + 10)
#     assert dht.get(validator_bad_reveal_key, latest=True) is None

#     validator_reveal_key = f"validator-reveal_epoch_{epoch}"

#     # max time
#     assert not dht.store(validator_reveal_key, b"bytes_value", mesh.get_dht_time() + 100000000000000000)

#     # reveal on the next epoch
#     write_epoch_json({
#         "block": 10,
#         "epoch": 1,
#         "block_per_epoch": 10,
#         "seconds_per_epoch": 60,
#         "percent_complete": 0.51,
#         "blocks_elapsed": 0,
#         "blocks_remaining": 10,
#         "seconds_elapsed": 0,
#         "seconds_remaining": 60
#     })

#     assert dht.store(validator_reveal_key, b"bytes_value", mesh.get_dht_time() + 10)
#     assert dht.get(validator_reveal_key, latest=True).value == b"bytes_value"



