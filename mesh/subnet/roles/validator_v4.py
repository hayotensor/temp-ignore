import asyncio
import hashlib
import io
import os
import pickle
import statistics
import threading
from collections import defaultdict
from typing import Dict, List

import torch

from mesh import DHT, get_dht_time
from mesh.dht.validation import RecordValidatorBase
from mesh.subnet.consensus_v2 import Consensus
from mesh.subnet.data_structures import ServerClass
from mesh.subnet.utils.consensus import (
    HOSTER_SCORE_RATIO,
    MAX_VALIDATOR_COMMIT_TIME,
    MAX_VALIDATOR_REVEAL_TIME,
    VALIDATOR_SCORE_RATIO,
    ConsensusScores,
    HosterResult,
)
from mesh.subnet.utils.hoster import get_hoster_commit_key, get_hoster_reveal_key
from mesh.subnet.utils.key import extract_rsa_peer_id
from mesh.subnet.utils.validator import (
    BASE_VALIDATOR_SCORE,
    EPSILON,
    get_validator_commit_key,
    get_validator_reveal_key,
    get_validator_subkey_rsa,
)
from mesh.substrate.chain_functions_v2 import Hypertensor
from mesh.utils.logging import get_logger

logger = get_logger(__name__)

# We remove 60 seconds off of the max expiration in case another node gets it late
# Max discrepency in DHT times is about 4 seconds, so 60 is plenty
_max_validator_commit_time = MAX_VALIDATOR_COMMIT_TIME - 60
_max_validator_reveal_time = MAX_VALIDATOR_REVEAL_TIME - 60

class Validator:
    """
    Note: This is ran by the hoster and validator roles

    The Validator class performs validation and scoring of hoster outputs in this container.

    This class can operate in two modes, based on the `role`:
      - As a validator: it verifies hoster outputs using a commit-reveal scheme, computes accuracy scores,
        commits its score hashes to the DHT, and later reveals them.
      - As a hoster: it performs the same validation and scoring of other hosters, but does not commit/reveal
        to the DHT. (It instead uses the scores to compare against the validator’s published consensus for attestation.)

    - In both roles, it submits the scores to the Consensus class to be used to submit scores to the blockchain if elected
      as the epoch's validator or to be used to compare to the elected validators scores to optionally attest.

    During each epoch, the Validator performs the following:
      1. Reveals any score commits from the previous epoch (validator role only).
      2. Fetches hoster commit/reveal records from the DHT.
      3. Verifies hoster reveal payloads against their commits.
      4. Loads and deserializes valid hoster tensors.
      5. Scores each hoster based on proximity to the mean tensor.
      6. Stores scores in the local `Consensus` instance.
      7. Commits a salted hash of scores to the DHT (validator role only) for later reveal.

    Note on Validator commit-reveal:
        Validators commit on one epoch and reveal on the next. The reason for this is to ensure one validator can't copy others.
        Even though the c-r can be verified by other nodes, we don't want them having access to anyone elses data until that data
        is no longer required for validating/attesting.

    Args:
        role (ServerClass): Role in the subnet, either `VALIDATOR` or `HOSTER`.
        dht (DHT): Hivemind DHT instance for storing and retrieving records.
        record_validator (RecordValidatorBase): Validator for DHT record signatures and permissions.
        consensus (Consensus): Local consensus handler to aggregate and submit scores.
        hypertensor (Hypertensor): Interface to the blockchain network for epoch/block queries.
    """

    def __init__(
        self,
        role: ServerClass,
        dht: DHT,
        record_validator: RecordValidatorBase,
        consensus: Consensus,
        hypertensor: Hypertensor,
    ):
        self.role = role
        self.dht = dht
        self.record_validator = record_validator
        self.consensus = consensus
        self.current_epoch = None
        self.latest_commit = None
        self.hypertensor = hypertensor
        self.epoch_length = int(str(self.hypertensor.get_epoch_length()))
        self.stop = threading.Event()

    def run(self, epoch: int):
        # 1) Reveal our previous-epoch commit if it exists
        self.reveal(epoch - 1)

        # 2) Run validation of hoster data from (epoch - 1) and score hosters
        self.score_hosters(epoch)

        # 3) Score validators from commit-reveal
        self.score_validators()

    def commit(self, scores: Dict[str, float], target_epoch: int):
        """
        Commit our scores to the DHT Records

        Validator role only calls this method
        """
        if self.role is not ServerClass.VALIDATOR:
            return

        scores_bytes = pickle.dumps(scores)
        salt = os.urandom(16)
        digest = hashlib.sha256(salt + scores_bytes).digest()

        validator_commit_key = get_validator_commit_key(self.current_epoch)

        self.dht.store(
            validator_commit_key,
            digest,
            get_dht_time() + _max_validator_commit_time,
            get_validator_subkey_rsa(self.record_validator)
        )

        # Store for next epochs reveal
        self.latest_commit = {
            "target_epoch": target_epoch,
            "salt": salt,
            "scores": scores_bytes,
        }

    def reveal(self, target_epoch: int):
        """
        Reveal our commit on the next epoch

        Validator role only calls this method
        """
        if (
            self.role is not ServerClass.VALIDATOR
            or not self.latest_commit
            or self.latest_commit["target_epoch"] != target_epoch
        ):
            return

        reveal_key = get_validator_reveal_key(target_epoch)

        reveal_payload = {
            "salt": self.latest_commit["salt"],
            "scores_bytes": self.latest_commit["scores"],
        }

        self.dht.store(
            reveal_key,
            reveal_payload,
            get_dht_time() + _max_validator_reveal_time,
            get_validator_subkey_rsa(self.record_validator)
        )

        logger.info(f"[Validator] Revealed scores for epoch {target_epoch}")

        self.latest_commit = None

    def score_hosters(self, current_epoch: int) -> Dict[str, float]:
        """
        Compute accuracy scores for each hoster based on the proximity of their output tensor
        to the mean tensor of all successful results.

        Each hoster's score is calculated as the inverse of their L2 distance to the mean tensor,
        i.e., `1 / (1 + distance)`. This ensures:
        - A score close to 1 if their output is close to the consensus.
        - A score approaching 0 if their output is very far or missing.

        If a hoster’s inference was unsuccessful or missing, they receive a score of 0.

        Args:
            results (List[HosterResult]): List of inference results from hosters, including their output tensors.

        Returns:
            Dict[str, float]: A mapping from peer ID to their computed accuracy score.
        """
        commit_key = get_hoster_commit_key(current_epoch - 1)
        reveal_key = get_hoster_reveal_key(current_epoch - 1)

        commit_records = self.dht.get(commit_key, latest=True) or {}
        reveal_records = self.dht.get(reveal_key, latest=True) or {}

        if not reveal_records:
            logger.warning(f"[Validator] No reveals found for epoch {current_epoch}")
            return

        results: List[HosterResult] = []
        for public_key, reveal_data in reveal_records.value.items():
            try:
                peer_id = extract_rsa_peer_id(public_key)

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

                # 2) Deserialize the tensor
                tensor = torch.load(io.BytesIO(tensor_bytes), weights_only=False)
                results.append(HosterResult(peer=peer_id, output=tensor, success=True))

            except Exception as e:
                logger.error(f"[Validator] Failed to verify or load tensor from {peer_id}: {e}")
                results.append(HosterResult(peer=peer_id, output=None, success=False))

        valid_outputs = [r.output for r in results if r.success and r.output is not None]
        if not valid_outputs:
            return {r.peer: 0.0 for r in results}

        stacked = torch.stack(valid_outputs)
        mean_tensor = torch.mean(stacked, dim=0)

        hoster_scores = {}
        for r in results:
            if not r.success or r.output is None:
                hoster_scores[r.peer] = 0.0
                continue
            diff = torch.norm(r.output - mean_tensor)
            hoster_scores[r.peer] = float(1.0 / (1.0 + diff.item()))

        hoster_scores = self.normalize_scores(hoster_scores, HOSTER_SCORE_RATIO)

        return hoster_scores

    def score_validators(self, current_epoch: int) -> Dict[str, float]:
        """
        Get the validator submitted data from the DHT Record

        We ensure the validator is submitting scores to the DHT

        - We get each Record entry by each hoster node
        - We iterate to validate and score this data and store it in the Consensus class

        Note:
            Validators with no reveal (1 epoch old validators)are not submitted as the
            reveal is the next epoch from the commit.
        """

        validator_commit_key = get_validator_commit_key(current_epoch - 2)
        validator_reveal_key = get_validator_reveal_key(current_epoch - 1)

        reveal_records = self.dht.get(validator_reveal_key) or {}
        validator_commit_records = self.dht.get(validator_commit_key) or {}

        if not reveal_records:
            logger.warning(f"[Validator] No reveals found for epoch {current_epoch}")
            return {}

        results: Dict[str, List[ConsensusScores]] = {}

        for public_key, reveal_data in reveal_records.value.items():
            try:
                peer_id = extract_rsa_peer_id(public_key)
                payload = reveal_data.value
                salt = payload["salt"]
                scores_bytes = payload["scores_bytes"]

                # 1) Verify the commit hash
                recomputed_digest = hashlib.sha256(salt + scores_bytes).digest()
                committed_digest = validator_commit_records.value[public_key].value

                if committed_digest != recomputed_digest:
                    print(f"[Validator] Hash mismatch from validator {peer_id}, skipping")
                    continue

                # 2) Deserialize the scores
                raw_scores = pickle.loads(scores_bytes)
                scores: List[ConsensusScores] = [
                    ConsensusScores(peer_id=k.to_base58(), score=v) for k, v in raw_scores.items()
                ]
                results[peer_id] = scores

            except Exception as e:
                print(f"[Validator] Failed to verify or parse scores from {peer_id}: {e}")

        # Step 1: Get scores per hoster peer_id
        peer_scores = defaultdict(list)  # hoster_peer_id -> list of scores
        for round_scores in results.values():
            for score_obj in round_scores:
                peer_scores[score_obj.peer_id].append(score_obj.score)

        # Step 2: Compute the mean score per hoster
        peer_means = {
            peer_id: statistics.mean(scores)
            for peer_id, scores in peer_scores.items()
        }

        # Step 3: Compute squared error per validator
        validator_errors: Dict[str, float] = {}
        for validator_peer_id, round_scores in results.items():
            error_sum = 0.0
            for score_obj in round_scores:
                mean = peer_means.get(score_obj.peer_id)
                if mean is not None:
                    error_sum += (score_obj.score - mean) ** 2
            validator_errors[validator_peer_id] = error_sum

        # Step 4: Normalize errors and subtract from base score
        max_error = max(validator_errors.values(), default=1.0)

        validator_scores = {
            peer_id: max(BASE_VALIDATOR_SCORE - (error / (max_error + EPSILON)), 0.0)
            for peer_id, error in validator_errors.items()
        }

        validator_scores = self.normalize_scores(validator_scores, VALIDATOR_SCORE_RATIO)

        self.consensus.store_validator_scores(validator_scores)

        return validator_scores

    def normalize_scores(self, scores: Dict[str, float], target_total: float) -> Dict[str, float]:
        total = sum(scores.values())
        if total == 0:
            return {peer_id: 0.0 for peer_id in scores}
        return {
            peer_id: (score / total) * target_total
            for peer_id, score in scores.items()
        }
