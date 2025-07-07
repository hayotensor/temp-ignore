import asyncio
import hashlib
import io
import math
import os
from collections import namedtuple
from functools import partial
from typing import List
from unittest.mock import Mock

import pytest
import torch

from mesh import DHT, get_dht_time
from mesh.dht.crypto import RSASignatureValidator
from mesh.subnet.roles.hoster_v2 import Hoster, get_consensus_key, get_hoster_commit_key
from mesh.subnet.utils.consensus import get_consensus_subkey_rsa
from mesh.subnet.utils.dht import _get_data, _store_data, store_data
from mesh.subnet.utils.hoster import get_hoster_reveal_key
from mesh.subnet.utils.key import generate_rsa_private_key_file, get_rsa_peer_id, get_rsa_private_key
from mesh.substrate.config import BLOCK_SECS
from mesh.utils.auth import TokenRSAAuthorizerBase

from test_utils.dht_swarms import (
    launch_dht_instances_with_record_validators,
)

# pytest tests/test_hoster_commit_reveal_rsa.py -rP

class DummyInferenceProtocol:
    async def run_inference_stream(self, tensor):
        return tensor * 2  # fake inference


class MockHypertensor:
    interface = None  # only used for mock epoch length

    def get_epoch_length(self):
        return 10

    def get_block_number(self):
        return 10

    def get_elected_validator_node(self, subnet_id: int, epoch: int):
        ...


# pytest tests/test_hoster_commit_reveal_rsa.py::test_hoster_commit_and_reveal_w_validation -rP

@pytest.mark.asyncio
async def test_hoster_commit_and_reveal_w_validation():
    test_path = "rsa_test_path.key"
    private_key, public_key, public_bytes, encoded_public_key, encoded_digest, peer_id = generate_rsa_private_key_file(test_path)
    loaded_key = get_rsa_private_key(test_path)
    record_validator = RSASignatureValidator(loaded_key)
    dht = DHT(start=True, record_validators=[record_validator])

    # Validator for the validator node that submits random tensor for hosters

    # Submit a random tensor acting as this epochs validator
    tensor = torch.randn(3, 3)
    epoch = 1
    epoch_length = 10  # dummy epoch length in seconds
    consensus_key = get_consensus_key(epoch)
    consensus_subkey = get_consensus_subkey_rsa(record_validator)

    # --- Store the consensus tensor into the DHT ---
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    data = buffer.read()

    dht.run_coroutine(
        partial(
           _store_data,
           key=consensus_key,
           subkey=consensus_subkey,
           value=data,
           expiration_time=get_dht_time() + 60
        ),
        return_future=False,
    )

    data = dht.run_coroutine(
        partial(
            _get_data,
            key=consensus_key,
            expiration_time=math.inf,
            latest=True,
        ),
        return_future=False,
    )
    # print("data", data)

    # --- Instantiate Hoster ---
    hoster = Hoster(
      dht=dht,
      inference_protocol=DummyInferenceProtocol(),
      record_validator=record_validator,
      hypertensor=MockHypertensor(),
      start=False
    )
    hoster.epoch_length = epoch_length

    # --- Verify try_load_tensor retrieves the same tensor ---
    loaded = await hoster.try_load_tensor(consensus_key, epoch)
    assert loaded is not None
    assert torch.allclose(loaded, tensor)

    # --- Run inference via DummyInferenceProtocol ---
    result = await hoster.inference_protocol.run_inference_stream(loaded)
    expected_result = tensor * 2
    assert torch.allclose(result, expected_result)

    # --- COMMIT PHASE ---
    hoster.commit(epoch, result)

    commit_key = get_hoster_commit_key(epoch)
    commit_record = dht.run_coroutine(
        partial(
            _get_data,
            key=commit_key,
            expiration_time=math.inf,
            latest=True,
        ),
        return_future=False,
    )
    assert commit_record is not None
    commit_data = getattr(next(iter(commit_record[commit_key].value.values()), None), "value", None)
    assert commit_data is not None

    assert isinstance(commit_data, bytes) and len(commit_data) == hashlib.sha256().digest_size

    # --- REVEAL PHASE ---
    hoster.reveal(epoch, result)

    reveal_key = get_hoster_reveal_key(epoch)
    reveal_record = dht.run_coroutine(
        partial(
            _get_data,
            key=reveal_key,
            expiration_time=math.inf,
            latest=True,
        ),
        return_future=False,
    )
    assert reveal_record is not None
    reveal_data = getattr(next(iter(reveal_record[reveal_key].value.values()), None), "value", None)
    assert data is not None

    # # # Deserialize payload using DHT's serializer
    # # serializer = MSGPackSerializer()
    # # payload = serializer.loads(reveal_record.value)
    # # assert isinstance(payload, dict)
    # # assert "salt" in payload and "tensor" in payload

    salt = reveal_data["salt"]
    tensor_bytes = reveal_data["tensor"]
    assert isinstance(salt, bytes) and len(salt) == 16
    assert isinstance(tensor_bytes, bytes)
    tensor = torch.load(io.BytesIO(tensor_bytes), weights_only=False)

    # --- VERIFY COMMIT MATCHES REVEAL ---
    recomputed_digest = hashlib.sha256(salt + tensor_bytes).digest()
    assert recomputed_digest == commit_data

    # Load the revealed tensor and check content
    revealed_tensor = torch.load(io.BytesIO(tensor_bytes), weights_only=False)
    assert torch.allclose(revealed_tensor, expected_result)

    # Shutdown DHT
    dht.shutdown()

    os.remove(test_path)

# pytest tests/test_hoster_commit_reveal_rsa.py::test_hoster_try_load_tensor_multi_stores -rP

@pytest.mark.asyncio
async def test_hoster_try_load_tensor_multi_stores():
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
        # token_rsa_validator = TokenRSAAuthorizerBase(loaded_key)
        # token_rsa_validators.append(token_rsa_validator)
        peer_id = get_rsa_peer_id(public_bytes)


    dhts = launch_dht_instances_with_record_validators(
        record_validators=record_validators,
        identity_paths=test_paths,
    )

    hoster_dht = dhts[0]
    hoster_peer_id = dhts[0].peer_id

    node_1_dht = dhts[1]
    node_1_peer_id = dhts[1].peer_id

    node_2_dht = dhts[2]
    node_2_peer_id = dhts[2].peer_id

    node_3_dht = dhts[3]
    node_3_peer_id = dhts[3].peer_id

    node_4_dht = dhts[4]
    node_4_peer_id = dhts[4].peer_id

    hypertensor = MockHypertensor()

    # Submit a random tensor acting as this epochs validator
    tensor = torch.randn(3, 3)
    epoch = 1
    epoch_length = 10  # dummy epoch length in seconds
    consensus_key = get_consensus_key(epoch)
    consensus_subkey = get_consensus_subkey_rsa(record_validator)

    # --- Store the consensus tensor into the DHT ---
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    data = buffer.read()

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
            data,
            get_dht_time() + 999,
            subkey
        )

    # --- Instantiate Hoster ---
    hoster = Hoster(
      dht=dht,
      subnet_id=1,
      subnet_node_id=1,
      inference_protocol=DummyInferenceProtocol(),
      record_validator=record_validator,
      hypertensor=hypertensor,
      start=False
    )


    # --- Verify try_load_tensor retrieves the same tensor ---
    loaded = await hoster.try_load_tensor(consensus_key, epoch)
    assert loaded is not None
    assert torch.allclose(loaded, tensor)

    # --- Run inference via DummyInferenceProtocol ---
    result = await hoster.inference_protocol.run_inference_stream(loaded)
    expected_result = tensor * 2
    assert torch.allclose(result, expected_result)

    # --- COMMIT PHASE ---
    hoster.commit(epoch, result)

    commit_key = get_hoster_commit_key(epoch)
    commit_record = dht.run_coroutine(
        partial(
            _get_data,
            key=commit_key,
            expiration_time=math.inf,
            latest=True,
        ),
        return_future=False,
    )
    assert commit_record is not None
    commit_data = getattr(next(iter(commit_record[commit_key].value.values()), None), "value", None)
    assert commit_data is not None

    assert isinstance(commit_data, bytes) and len(commit_data) == hashlib.sha256().digest_size

    # --- REVEAL PHASE ---
    hoster.reveal(epoch, result)

    reveal_key = get_hoster_reveal_key(epoch)
    reveal_record = dht.run_coroutine(
        partial(
            _get_data,
            key=reveal_key,
            expiration_time=math.inf,
            latest=True,
        ),
        return_future=False,
    )
    assert reveal_record is not None
    reveal_data = getattr(next(iter(reveal_record[reveal_key].value.values()), None), "value", None)
    assert data is not None

    salt = reveal_data["salt"]
    tensor_bytes = reveal_data["tensor"]
    assert isinstance(salt, bytes) and len(salt) == 16
    assert isinstance(tensor_bytes, bytes)
    tensor = torch.load(io.BytesIO(tensor_bytes), weights_only=False)

    # --- VERIFY COMMIT MATCHES REVEAL ---
    recomputed_digest = hashlib.sha256(salt + tensor_bytes).digest()
    assert recomputed_digest == commit_data

    # Load the revealed tensor and check content
    revealed_tensor = torch.load(io.BytesIO(tensor_bytes), weights_only=False)
    assert torch.allclose(revealed_tensor, expected_result)

    # Shutdown DHT
    dht.shutdown()

    for path in test_paths:
        os.remove(path)

# pytest tests/test_hoster_commit_reveal_rsa.py::test_hoster_run_forever -rP

EpochInfo = namedtuple("EpochInfo", ["epoch", "percent_complete"])

@pytest.mark.asyncio
async def test_hoster_run_forever(monkeypatch):
    # ─── STEP 1: Spin up an in‐memory DHT ────────────────────────────────────────
    dht = DHT(start=True)
    # Give it a short grace time to fully start up
    await asyncio.sleep(0.1)

    # Generate a dummy RSA keypair, wrap in RSASignatureValidator for subkey use:
    test_path = "rsa_test_path.key"
    private_key, public_key, public_bytes, encoded_public_key, encoded_digest, peer_id = generate_rsa_private_key_file(test_path)
    loaded_key = get_rsa_private_key(test_path)
    record_validator = RSASignatureValidator(loaded_key)

    # ─── STEP 2: Pre‐store a “consensus” tensor under epoch 0 key ───────────────
    # Create a small 2×2 tensor
    consensus_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # Serialize it to bytes using torch.save()
    buf = io.BytesIO()
    torch.save(consensus_tensor, buf)
    buf_bytes = buf.getvalue()

    consensus_key = get_consensus_key(0)  # epoch 0
    store_data(
        dht=dht,
        key=consensus_key,
        subkey=b"",  # no subkey for the input
        data=buf_bytes,
        expiration_time=get_dht_time() + 30,
        wait=True,
    )

    # ─── STEP 3: Prepare a fixed sequence of fake blocks / epoch progress ────────
    #
    # We want the Hoster to do:
    #  - In “first half” (percent <= 0.5) of epoch 0: see consensus, run inference, commit.
    #  - Then in “second half” (percent > 0.5) of epoch 0: reveal.
    #
    # After that, we can stop the loop.
    #
    # We’ll feed get_block_number() and get_epoch_progress() calls from this queue:
    timeline = [
        # (block_number, percent_complete)  → still in epoch 0, first half
        (0, 0.0),
        (0, 0.0),  # hoster.step → commit
        # Next: still epoch 0 but now second‐half
        (0, 0.6),
        (0, 0.6),  # hoster.step → reveal
        # Move to “Epoch 1” so that hoster will go to sleep or exit run_forever
        (10, 0.0),  # Suppose epoch_length = 10 → block 10 / 10 = epoch 1
    ]
    # Make an iterator over that list
    timeline_iter = iter(timeline)

    # # Monkeypatch get_block_number(...) so it returns each block in turn
    # def fake_get_block_number(interface):
    #     try:
    #         b, _ = next(timeline_iter)
    #         return b
    #     except StopIteration:
    #         # If we run out of values, keep returning something in epoch 1
    #         return 10

    # Instead of calling next(timeline_iter) again, capture percent from the same step:
    percent_progress = [0.0]  # a mutable holder so fake_get_epoch_progress can read/write

    # Monkeypatch get_epoch_progress(block, epoch) to match our timeline values
    def fake_get_epoch_progress(block_number, epoch):
        # Our timeline pairs are (block, percent). We peek at the last returned percent
        # But get_block_number has already advanced the iterator; instead, stash the last percent
        return EpochInfo(epoch=0 if block_number < 10 else 1,
                         percent_complete=percent_progress[0])

    # But we need to advance next_percent each time get_block_number is called.
    # So wrap fake_get_block_number to also update next_percent:
    def fake_get_block_number_and_update():
        try:
            b, pct = next(timeline_iter)
            percent_progress[0] = pct
            return b
        except StopIteration:
            percent_progress[0] = 1.0
            return 10

    mock_ht = MockHypertensor()
    mock_ht.get_block_number = fake_get_block_number_and_update
    monkeypatch.setattr("mesh.subnet.roles.hoster_v2.get_epoch_progress", fake_get_epoch_progress)

    # ─── STEP 4: Instantiate and run the Hoster ────────────────────────────────
    hoster = Hoster(
        dht=dht,
        inference_protocol=DummyInferenceProtocol(),
        record_validator=record_validator,
        hypertensor=mock_ht,
    )
    # Override epoch_length to match our fake timeline (10 seconds per epoch)
    hoster.epoch_length = 10

    # Launch run_forever() but cancel it after a short timeout
    task = asyncio.create_task(hoster.run_forever())
    try:
        # Wait long enough for both commit and reveal to have happened
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        # Timeout is expected; we just want commit/reveal to happen before
        task.cancel()
        await asyncio.sleep(0.1)

    # ─── STEP 5: Check the DHT for commit (epoch 0) and reveal (epoch 0) ─────────
    # There should be a commit under get_hoster_commit_key(0)
    commit_key_0 = get_hoster_commit_key(0)
    commit_record = dht.run_coroutine(
        partial(
            _get_data,
            key=commit_key_0,
            expiration_time=math.inf,
            latest=True,
        ),
        return_future=False,
    )
    # commit_key_0 = get_hoster_commit_key(0)
    # commit_record = dht.get(commit_key_0, latest=True, return_future=False)
    # print("commit_record", commit_record)
    assert commit_record is not None, "Hoster commit for epoch 0 was not found"
    commit_record_value = getattr(next(iter(commit_record[commit_key_0].value.values()), None), "value", None)
    print("commit_record_value", commit_record_value)

    await asyncio.sleep(BLOCK_SECS*2)
    percent_progress[0] = 0.6
    await asyncio.sleep(BLOCK_SECS*2)

    reveal_key_0 = get_hoster_reveal_key(0)

    reveal_record = dht.run_coroutine(
        partial(
            _get_data,
            key=reveal_key_0,
            expiration_time=math.inf,
            latest=True,
        ),
        return_future=False,
    )
    # reveal_record = dht.get(reveal_key_0, latest=True, return_future=False)
    print("reveal_record", reveal_record)
    assert reveal_record is not None, "Hoster reveal for epoch 0 was not found"

    payload = getattr(next(iter(reveal_record[reveal_key_0].value.values()), None), "value", None)
    print("payload", payload)

    # Verify that reveal content matches the earlier commit:
    # payload = reveal_record.value
    salt = payload["salt"]
    tensor_bytes = payload["tensor"]
    assert isinstance(salt, bytes) and len(salt) == 16
    assert isinstance(tensor_bytes, bytes)
    recomputed = hashlib.sha256(salt + tensor_bytes).digest()
    assert recomputed == commit_record_value

    # Also verify that the revealed tensor is (input * 2)
    loaded_tensor = torch.load(io.BytesIO(tensor_bytes), map_location="cpu")
    expected = consensus_tensor * 2
    assert torch.allclose(loaded_tensor, expected)