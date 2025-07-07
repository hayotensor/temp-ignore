import asyncio
import hashlib
import io
import os
import pickle
from typing import Dict, List

import torch

from mesh import DHT, get_dht_time
from mesh.dht.validation import RecordValidatorBase
from mesh.subnet.utils.consensus import HosterResult
from mesh.subnet.utils.hoster import get_hoster_commit_key, get_hoster_reveal_key
from mesh.subnet.utils.key import extract_rsa_peer_id_from_subkey
from mesh.subnet.utils.validator import (
    get_validator_commit_key,
    get_validator_reveal_key,
    get_validator_subkey_rsa,
)
from mesh.substrate.chain_functions_v2 import Hypertensor
from mesh.utils.logging import get_logger

logger = get_logger(__name__)

class Validator:
    def __init__(
        self,
        dht: DHT,
        record_validator: RecordValidatorBase,
        hypertensor: Hypertensor
    ):
        self.dht = dht
        self.record_validator = record_validator
        self.current_epoch = None
        self.latest_commit = None
        self.hypertensor = hypertensor
        self.epoch_length = 1
        # self.epoch_length = int(str(self.hypertensor.get_epoch_length()))


    async def run_forever(self, poll_interval: float = 5.0):
        while True:
            current_epoch = int(get_dht_time() // self.epoch_length)
            if current_epoch != self.current_epoch:
                logger.info(f"[Validator] New epoch {current_epoch}")

                # 1) Reveal our previous-epoch commit if it exists
                self.reveal(current_epoch - 1)

                # 2) Run validation of hoster data from (current_epoch - 1)
                self.run_hoster_validation(current_epoch - 1)

                # 3) Mark that we've handled epoch N
                self.current_epoch = current_epoch
            await asyncio.sleep(poll_interval)

    def reveal(self, target_epoch: int):
        """
        Reveal our commit on the next epoch
        """
        if not self.latest_commit or self.latest_commit["target_epoch"] != target_epoch:
            return

        reveal_key = get_validator_reveal_key(target_epoch)

        reveal_payload = {
            "salt": self.latest_commit["salt"],
            "scores_bytes": self.latest_commit["scores"],
        }

        self.dht.store(
            reveal_key,
            reveal_payload,
            get_dht_time() + self.epoch_length + 999,
            get_validator_subkey_rsa(self.record_validator)
        )

        logger.info(f"[Validator] Revealed scores for epoch {target_epoch}")

        self.latest_commit = None


    def run_hoster_validation(self, target_epoch: int):
        logger.info(f"[Validator] Running validation for epoch {target_epoch}")

        # === Fetch Commits and Reveals ===
        commit_key = get_hoster_commit_key(target_epoch)
        reveal_key = get_hoster_reveal_key(target_epoch)

        commit_records = self.dht.get(commit_key, latest=True) or {}
        reveal_records = self.dht.get(reveal_key, latest=True) or {}

        if not reveal_records:
            logger.warning(f"[Validator] No reveals found for epoch {target_epoch}")
            return

        results: List[HosterResult] = []
        for public_key, reveal_data in reveal_records.value.items():
            try:
                peer_id = extract_rsa_peer_id_from_subkey(public_key)

                payload = reveal_data.value
                salt = payload["salt"]
                tensor_bytes = payload["tensor"]

                # 1) Verify hoster committed the same hash
                recomputed_digest = hashlib.sha256(salt + tensor_bytes).digest()
                committed_digest = commit_records.value[public_key].value

                if committed_digest != recomputed_digest:
                    logger.warning(f"[Validator] Host {peer_id} hash mismatch, skipping")
                    results.append(HosterResult(peer=peer_id, output=None, success=False))
                    continue

                # 3) Deserialize the tensor
                tensor = torch.load(io.BytesIO(tensor_bytes), weights_only=False)
                results.append(HosterResult(peer=peer_id, output=tensor, success=True))

            except Exception as e:
                logger.error(f"[Validator] Failed to verify or load tensor from {peer_id}: {e}")
                results.append(HosterResult(peer=peer_id, output=None, success=False))

        # 2) Score all hoster outputs
        scores = self.score_hosters(results)
        logger.info(f"[Validator] Scores for epoch {target_epoch}: {scores}")

        # 3) Commit our validator‐score hash (under current epoch, to reveal next epoch)
        scores_bytes = pickle.dumps(scores)
        salt = os.urandom(16)
        digest = hashlib.sha256(salt + scores_bytes).digest()

        validator_commit_key = get_validator_commit_key(self.current_epoch)

        self.dht.store(
            validator_commit_key,
            digest,
            get_dht_time() + self.epoch_length + 999,
            get_validator_subkey_rsa(self.record_validator)
        )

        # Store for next epochs reveal
        self.latest_commit = {
            "target_epoch": target_epoch,
            "salt": salt,
            "scores": scores_bytes,
        }

    def score_hosters(self, results: List[HosterResult]) -> Dict[str, float]:
        """
        Compute accuracy scores for each hoster based on proximity to the mean output.
        """
        valid_outputs = [r.output for r in results if r.success and r.output is not None]
        if not valid_outputs:
            return {r.peer: 0.0 for r in results}

        stacked = torch.stack(valid_outputs)
        mean_tensor = torch.mean(stacked, dim=0)

        scores = {}
        for r in results:
            if not r.success or r.output is None:
                scores[r.peer] = 0.0
                continue
            diff = torch.norm(r.output - mean_tensor)
            scores[r.peer] = float(1.0 / (1.0 + diff.item()))  # Inverse distance

        return scores
