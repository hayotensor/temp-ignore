import io
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List
from unittest.mock import MagicMock

import pytest
import torch

from mesh import PeerID, get_dht_time
from mesh.dht.crypto import RSASignatureValidator
from mesh.subnet.consensus_v2 import Consensus
from mesh.subnet.data_structures import ServerClass
from mesh.subnet.roles.hoster_v2 import Hoster
from mesh.subnet.roles.validator_v2 import Validator
from mesh.subnet.utils.consensus import get_consensus_key
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
from mesh.substrate.config import BLOCK_SECS

from test_utils.dht_swarms import launch_dht_instances_with_record_validators

# pytest tests/test_consensus.py -rP


class DummyInferenceProtocol:
    async def rpc_inference_stream(self, tensor):
        return tensor * 2  # fake inference

class DummyConsensus:
    def store_hoster_scores(self, hoster_scores: Dict[str, float]):
        self.hoster_scores = hoster_scores

    def store_validator_scores(self, validator_scores: Dict[str, float]):
        self.validator_scores = validator_scores

@dataclass
class EpochData:
  current_block: int
  epoch: int
  epoch_length: int
  percent_complete: float
  blocks_elapsed: int
  blocks_remaining: int
  seconds_remaining: int

class MockHypertensor:
    interface = None  # only used for mock epoch length

    def get_epoch_length(self):
        return 10

    def get_block_number(self):
        return 100

    def get_epoch(self):
        return 1

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


# pytest tests/test_consensus.py::test_merge_scores -rP
# pytest tests/test_consensus.py::test_merge_scores --log-cli-level=DEBUG

@pytest.mark.forked
@pytest.mark.asyncio
async def test_merge_scores():
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


    peer1 = random.randint(0, peers_len-1)
    peer2 = random.randint(0, peers_len-1)

    keys = ["k1", "k2"]
    values = [123, 567]
    dhts[peer1].store(keys[0], values[0], get_dht_time() + 999, subkey=record_validators[peer1].local_public_key),
    dhts[peer2].store(keys[1], values[1], get_dht_time() + 999, subkey=record_validators[peer2].local_public_key),

    you = random.choice(dhts)

    results1 = you.get("k1")
    assert results1 is not None

    # Get another DHT that's not the same
    other_dhts = [dht for dht in dhts if dht != you]
    assert other_dhts, "No other DHTs available. "

    someone = random.choice(other_dhts)

    results1 = someone.get(keys[0])
    first_payload = next(iter(results1.value.values())).value
    assert first_payload == values[0], "Incorrect value in payload. "

    hosters_length = 3
    validators_length = 3

    hypertensor = MockHypertensor()
    epoch = hypertensor.get_epoch()

    hosters: List[Hoster] = []
    hoster_peer_ids = []
    for i in range(0, hosters_length):
        hoster_peer_ids.append(dhts[i].peer_id)
        hoster = Hoster(
            dht=dhts[i],
            inference_protocol=DummyInferenceProtocol(),
            record_validator=record_validators[i],
            hypertensor=hypertensor,
            start=False,
        )
        hosters.append(hoster)

    consensuses: List[Consensus] = []
    validators: List[Validator] = []
    validator_peer_ids = []
    for i in range(hosters_length, hosters_length + validators_length):
        consensus = Consensus(
          dht=dhts[i],
          subnet_id=0,
          subnet_node_id=0,
          record_validator=record_validators[i],
          hypertensor=hypertensor,
          model_name_or_path="bigscience/bloom-560m",
          start=False,
        )
        consensuses.append(consensus)

        validator_peer_ids.append(dhts[i].peer_id)
        validator = Validator(
            role=ServerClass.VALIDATOR,
            dht=dhts[i],
            record_validator=record_validators[i],
            consensus=consensus,
            hypertensor=hypertensor,
            start=False,
        )
        validator.current_epoch = epoch + 1
        validators.append(validator)

    # === Step 1: Store random input tensor as the chosen validator ===
    consensus_key = get_consensus_key(epoch)
    tensor = torch.randn(2, 2)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    value = buffer.read()

    dhts[peer1].store(consensus_key, value, get_dht_time() + 999, subkey=record_validators[peer1].local_public_key),
    results2 = someone.get(consensus_key)
    assert results2 is not None
    payload = next(iter(results2.value.values())).value
    assert payload == value, "Incorrect value in payload. "

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
    print("commit_records", commit_records)
    assert len(commit_records.value) == len(hosters)

    for subkey, value in commit_records.value.items():
        peer_id = extract_rsa_peer_id(subkey)
        assert any(peer_id == _peer_id for _peer_id in all_peer_ids)

    reveal_key = get_hoster_reveal_key(epoch)

    commit_records = validators[0].dht.get(commit_key)
    assert commit_records is not None

    reveal_records = validators[0].dht.get(reveal_key)
    assert reveal_records is not None

    for i in range(0, validators_length):
        validators[i].run_hoster_validation(epoch)

        validator_commit_key = get_validator_commit_key(validators[i].current_epoch)
        validator_commit_records = someone.get(validator_commit_key)
        assert validator_commit_records is not None

        hoster_scores = validators[i].consensus.hoster_scores
        assert hoster_scores is not None


        validators[i].reveal(validators[i].current_epoch - 1)

    for i in range(0, validators_length):
        # ensure score validators works
        validators[i].score_validators()
        validator_scores = validators[i].consensus.validator_scores
        assert validator_scores is not None
        assert len(validator_scores) == validators_length # flaky

    validator_reveal_key = get_validator_reveal_key(validators[i].current_epoch - 1)
    validator_reveal_records = someone.get(validator_reveal_key)
    assert validator_reveal_records is not None

    validator_commit_records = someone.get(validator_commit_key)
    assert validator_commit_records is not None

    merged_scores = consensuses[0].get_merged_scores()
    assert len(merged_scores) == validators_length + hosters_length

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
    consensus = Consensus(
        dht=MagicMock(),
        subnet_id=1,
        subnet_node_id=123,
        record_validator=MagicMock(),
        hypertensor=MockHypertensor(),
        model_name_or_path="bigscience/bloom-560m"
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



