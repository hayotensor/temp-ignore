import io
import os
import random
from typing import Dict, List

import pytest
import torch

from mesh import PeerID, get_dht_time
from mesh.dht.crypto import RSASignatureValidator
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

from test_utils.dht_swarms import launch_dht_instances_with_record_validators

# pytest tests/test_validator.py -rP

# pytest tests/test_validator.py::test_score_speed_with_dynamic_brackets -rP


class DummyInferenceProtocol:
    async def run_inference_stream(self, tensor):
        return tensor * 2  # fake inference

class DummyConsensus:
    def store_hoster_scores(self, hoster_scores: Dict[str, float]):
        self.hoster_scores = hoster_scores

    def store_validator_scores(self, validator_scores: Dict[str, float]):
        self.validator_scores = validator_scores

class MockHypertensor:
    interface = None  # only used for mock epoch length

    def get_epoch_length(self):
        return 10

# pytest tests/test_validator.py::test_dht_reuse_get_with_validators -rP
# pytest tests/test_validator.py::test_dht_reuse_get_with_validators --log-cli-level=DEBUG

@pytest.mark.forked
@pytest.mark.asyncio
# @pytest.mark.xfail(reason="Flaky test", strict=False)
async def test_dht_reuse_get_with_validators():
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

    epoch = 1
    epoch_length = 10
    hosters_length = 3
    validators_length = 3

    hosters: List[Hoster] = []
    hoster_peer_ids = []
    for i in range(0, hosters_length):
        hoster_peer_ids.append(dhts[i].peer_id)
        hoster = Hoster(
            dht=dhts[i],
            inference_protocol=DummyInferenceProtocol(),
            record_validator=record_validators[i],
            hypertensor=MockHypertensor(),
            start=False
        )
        hoster.epoch_length = epoch_length
        hosters.append(hoster)

    validators: List[Validator] = []
    validator_peer_ids = []
    for i in range(hosters_length, hosters_length + validators_length):
        validator_peer_ids.append(dhts[i].peer_id)
        validator = Validator(
            role=ServerClass.VALIDATOR,
            dht=dhts[i],
            record_validator=record_validators[i],
            consensus=DummyConsensus(),
            hypertensor=MockHypertensor(),
            start=False
        )
        validator.epoch_length = epoch_length
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
        print("loaded", loaded)
        assert loaded is not None
        assert torch.allclose(loaded, tensor)

        # --- Run inference via DummyInferenceProtocol ---
        result = await hoster.inference_protocol.run_inference_stream(loaded)
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

    for i in range(0, validators_length):
        validators[i].run_hoster_validation(epoch)

        validator_commit_key = get_validator_commit_key(validators[i].current_epoch)
        validator_commit_records = someone.get(validator_commit_key)
        print("validator_commit_records", validator_commit_records)
        assert validator_commit_records is not None

        hoster_scores = validators[i].consensus.hoster_scores
        assert hoster_scores is not None


        validators[i].reveal(validators[i].current_epoch - 1)

        # ensure score validators works
        validators[i].score_validators()
        validator_scores = validators[i].consensus.validator_scores
        print("validator_scores", validator_scores)
        assert validator_scores is not None


    validator_reveal_key = get_validator_reveal_key(validators[i].current_epoch - 1)
    validator_reveal_records = someone.get(validator_reveal_key)
    print("validator_reveal_records", validator_reveal_records)
    assert validator_reveal_records is not None

    validator_commit_records = someone.get(validator_commit_key)
    print("validator_commit_records", validator_commit_records)
    assert validator_commit_records is not None


    for path in test_paths:
        os.remove(path)


# pytest tests/test_validator.py::test_dht_reuse_get_with_validators_with_auth -rP
# pytest tests/test_validator.py::test_dht_reuse_get_with_validators_with_auth --log-cli-level=DEBUG

@pytest.mark.forked
@pytest.mark.asyncio
async def test_dht_reuse_get_with_validators_with_auth():
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

    epoch = 1
    epoch_length = 10
    hosters_length = 3
    validators_length = 3

    hosters: List[Hoster] = []
    hoster_peer_ids = []
    for i in range(0, hosters_length):
        hoster_peer_ids.append(dhts[i].peer_id)
        hoster = Hoster(
            dht=dhts[i],
            inference_protocol=DummyInferenceProtocol(),
            record_validator=record_validators[i],
            hypertensor=MockHypertensor(),
            start=False
        )
        hoster.epoch_length = epoch_length
        hosters.append(hoster)

    validators: List[Validator] = []
    validator_peer_ids = []
    for i in range(hosters_length, hosters_length + validators_length):
        validator_peer_ids.append(dhts[i].peer_id)
        validator = Validator(
            role=ServerClass.VALIDATOR,
            dht=dhts[i],
            record_validator=record_validators[i],
            consensus=DummyConsensus(),
            hypertensor=MockHypertensor(),
            start=False
        )
        validator.epoch_length = epoch_length
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
        print("loaded", loaded)
        assert loaded is not None
        assert torch.allclose(loaded, tensor)

        # --- Run inference via DummyInferenceProtocol ---
        result = await hoster.inference_protocol.run_inference_stream(loaded)
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

    for i in range(0, validators_length):
        validators[i].run_hoster_validation(epoch)

        validator_commit_key = get_validator_commit_key(validators[i].current_epoch)
        validator_commit_records = someone.get(validator_commit_key)
        print("validator_commit_records", validator_commit_records)
        assert validator_commit_records is not None

        hoster_scores = validators[i].consensus.hoster_scores
        assert hoster_scores is not None


        validators[i].reveal(validators[i].current_epoch - 1)

        # ensure score validators works
        validators[i].score_validators()
        validator_scores = validators[i].consensus.validator_scores
        print("validator_scores", validator_scores)
        assert validator_scores is not None


    validator_reveal_key = get_validator_reveal_key(validators[i].current_epoch - 1)
    validator_reveal_records = someone.get(validator_reveal_key)
    print("validator_reveal_records", validator_reveal_records)
    assert validator_reveal_records is not None

    validator_commit_records = someone.get(validator_commit_key)
    print("validator_commit_records", validator_commit_records)
    assert validator_commit_records is not None


    for path in test_paths:
        os.remove(path)
