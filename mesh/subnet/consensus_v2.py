import asyncio
import sys
import threading
from dataclasses import asdict, dataclass
from typing import Dict, List

from mesh import DHT, get_dht_time
from mesh.dht.validation import RecordValidatorBase
from mesh.subnet.utils.consensus import (
    CONSENSUS_FALLBACK_STORE_DEADLINE,
    MAX_CONSENSUS_TIME,
    ConsensusScores,
    get_consensus_key,
    get_consensus_subkey_rsa,
)
from mesh.subnet.utils.key import extract_rsa_peer_id
from mesh.subnet.utils.random_prompts import RandomPrompts
from mesh.substrate.chain_functions_v2 import Hypertensor
from mesh.substrate.config import BLOCK_SECS
from mesh.utils import get_logger

logger = get_logger(__name__)

PROMPT_DEADLINE = 0.10 # After 35% of epoch, anyone can submit prompt
MIN_PROMPTS = 10

_max_consensus_time = MAX_CONSENSUS_TIME - 60

@dataclass
class ConsensusConfig:
    epoch_length: int
    subnet_node_id: str
    substrate_url: str

class Consensus(threading.Thread):
    def __init__(
        self,
        dht: DHT,
        subnet_id: int,
        subnet_node_id: int,
        record_validator: RecordValidatorBase,
        hypertensor: Hypertensor,
        converted_model_name_or_path: str,
        start: bool = True
    ):
        super().__init__(daemon=True)
        self.dht = dht
        self.peer_id = self.dht.peer_id
        self.subnet_id = subnet_id
        self.subnet_node_id = subnet_node_id
        self.hypertensor = hypertensor
        self.epoch_length = int(str(self.hypertensor.get_epoch_length()))
        self.record_validator = record_validator
        self.prompt_generator = RandomPrompts(converted_model_name_or_path)
        self.hoster_scores = None
        self.validator_scores = None
        self.previous_epoch_data = None

        self.stop = threading.Event()

        if start:
            self.run()

    def is_validator(self, epoch: int) -> bool:
        validator = self.get_validator(epoch)
        return validator == self.subnet_node_id

    def get_validator(self, epoch: int):
        validator = self.hypertensor.get_rewards_validator(self.subnet_id, epoch)
        return validator

    def get_merged_scores(self) -> List[ConsensusScores]:
        """
        Merge hoster and validator scores submitted by Consensus
        """
        if self.hoster_scores is None and self.validator_scores is None:
            return []

        merged_scores = self.hoster_scores.copy()
        merged_scores.update(self.validator_scores)

        # Step 2: Convert to List[ConsensusScores], rounding or casting score to int
        consensus_score_list = [
            ConsensusScores(peer_id=peer_id, score=int(score * 1e18))
            for peer_id, score in merged_scores.items()
        ]

        # Reset scores for node roles
        self.hoster_scores = None
        self.validator_scores = None

        return consensus_score_list

    def store_hoster_scores(self, hoster_scores: Dict[str, float]):
        """
        Stores the consensus scores of hosters to be used to validate or attest

        This is called by the Validator (used by both roles, see validator.py)
        """
        self.hoster_scores = hoster_scores

    def store_validator_scores(self, validator_scores: Dict[str, float]):
        """
        Stores the consensus scores of validators to be used to validate or attest

        This is called by the Validator (used by both roles, see validator.py)
        """
        self.validator_scores = validator_scores

    def compare_consensus_data(self, my_data, validator_data, epoch: int) -> bool:
        """
        Compare the current epochs elected validator consensus submission to our own

        Must have 100% accuracy to return True
        """
        # if validator submitted no data, and we have also found the subnet is broken
        if len(validator_data) == 0 and len(my_data) == 0:
            self.previous_epoch_data = []
            return True

        # otherwise, check the data matches
        # at this point, the

        # use ``asdict`` because data is decoded from blockchain as dataclass
        # we assume the lists are consistent across all elements
        # Convert validator_data to a set
        validator_data_set = set(frozenset(asdict(d).items()) for d in validator_data)

        # Convert my_data to a set
        my_data_set = set(frozenset(d.items()) for d in my_data)

        success = validator_data_set == my_data_set

        if not success:
            """
            The following accounts for nodes that go down or back up in the after or before validation submissions and attestations

            # Cases

            Case 1: Node leaves DHT before validator submit consensus and returns before attestation.
                    - Validator data does not include node, Attestors data will include node, creating a mismatch.
            Case 2: Node leaves DHT after validator submits consensus.
                    - Validator data does include node, Attestors data does not include node, creating a mismatch.

            # Solution

            We check our previous epochs data, if successfully attested, to find symmetry with the validators data.

            * If none of these solutions work, we assume the validator is being dishonest or faulty
            """
            if not success and self.previous_epoch_data is not None:
                dif = validator_data_set.symmetric_difference(my_data_set)
                success = dif.issubset(self.previous_epoch_data)
            elif not success and self.previous_epoch_data is None:
                """
                If this is the nodes first epoch after a restart of the node, check last epochs consensus data
                """
                previous_epoch_validator_data = self._get_validator_consensus_submission(epoch - 1)
                # This is a backup so we ensure the data was super majority attested to use it
                if previous_epoch_validator_data is not None:
                    _, attestation_percentage = self._get_reward_result(epoch)
                    if attestation_percentage / 1e18 < 0.66:
                        # TODO: Check
                        success = False
                    else:
                        previous_epoch_data_onchain = set(
                            frozenset(asdict(d).items()) for d in previous_epoch_validator_data
                        )
                        dif = validator_data_set.symmetric_difference(my_data_set)
                        success = dif.issubset(previous_epoch_data_onchain)

        self.previous_epoch_data = my_data_set

        return success

    def run(self):
        asyncio.run(self.run_forever())

    async def run_forever(self):
        """
        Loop until a new epoch to found, then run consensus logic
        """

        self._async_stop_event = asyncio.Event()
        last_epoch = None

        while not self.stop.is_set() and not self._async_stop_event.is_set():
            current_block = self.hypertensor.get_block_number()
            current_epoch = current_block // self.epoch_length

            if current_epoch != last_epoch:
                await self.run_consensus(current_epoch, current_block)
                last_epoch = current_epoch

            blocks_until_next_epoch = ((current_epoch + 1) * self.epoch_length) - current_block
            seconds_until_next_epoch = blocks_until_next_epoch * BLOCK_SECS

            try:
                await asyncio.wait_for(self._async_stop_event.wait(), timeout=seconds_until_next_epoch)
            except asyncio.TimeoutError:
                pass

    def generate_and_store_random_tensor(self, key, subkey):
        prompt = self.prompt_generator.generate_prompt_tensor()
        self.dht.store(
            key,
            prompt,
            get_dht_time() + _max_consensus_time,
            subkey
        )

    async def run_consensus(self, epoch: int, block: int):
        """
        At the start of each epoch, we check if we are validator

        We start by:
            - Query each result from the hosters Records of their inference from the previous epochs prompt task
            - Score this data for each hoster

        If elected on-chain validator:
            - Submit scores to Hypertensor

        If attestor (non-elected on-chain validator):
            - Retrieve validators score submission from Hypertensor
            - Compare to our own
            - Attest if 100% accuracy, else do nothing
        """
        consensus_key = get_consensus_key(epoch)
        consensus_subkey = get_consensus_subkey_rsa(self.record_validator)

        scores = self.get_merged_scores()
        print(f"Epoch scores: {scores}")

        if self.is_validator(epoch):
            print(f"Acting as validator for epoch {epoch}")

            if len(scores) == 0:
                self.hypertensor.validate(self.subnet_id, data=scores)
            else:
                self.hypertensor.validate(self.subnet_id, data=scores)

            """
            Generate random tensors for next epoch and store to DHT

            1. Check if we alrady stored it
                - Depending on when we get to this function, others may have assumed
                  we haven't submitted the prompt yet so we check if we have by iterating
                  all of the current results under the `consensus_key`
            2. Store it
            """
            consensus_records = self.dht.get(consensus_key) or {}
            if not consensus_records:
                self.generate_and_store_random_tensor(consensus_key, consensus_subkey)
            else:
                has_submitted = False
                for public_key, _ in consensus_records.value.items():
                    try:
                        peer_id = extract_rsa_peer_id(public_key)
                        if peer_id.__eq__(self.peer_id):
                            has_submitted = True
                            break
                    except Exception:
                        pass

                if not has_submitted:
                    self.generate_and_store_random_tensor(consensus_key, consensus_subkey)
                    print(f"Validator submitted prompt for epoch {epoch}")
        else:
            print(f"Acting as attestor for epoch {epoch}")
            while not self.stop.is_set():
                epoch_data = self.hypertensor.get_epoch_progress()
                validator_data = self.hypertensor.get_consensus_data(self.subnet_id, epoch)

                # If validator hasn't submitted by the CONSENSUS_FALLBACK_STORE_DEADLINE (10%)
                # mark of the epoch, break and don't attest
                if (
                    validator_data is None and
                    epoch_data.seconds_per_epoch * CONSENSUS_FALLBACK_STORE_DEADLINE > epoch_data.seconds_elapsed
                ):
                    continue

                """
                Get all of the hosters inference outputs they stored to the DHT
                """
                if self.compare_consensus_data(scores, validator_data, epoch):
                    print(f"Validator data matches for epoch {epoch}, attesting")
                    self.hypertensor.attest(self.subnet_id)
                else:
                    print(f"Data doesn't match validator for epoch {epoch}, moving forward with no attetation")


            """
            1. Sleep until 10% into the epoch
            2. Check if the validator submitted a prompt
            3. If they haven't, submit one
            """
            while not self.stop.is_set():
                epoch_data = self.hypertensor.get_epoch_progress()

                # Wait until 1/5 of the progress into an epoch to check if prompt has been submitted
                await asyncio.sleep(
                    max(0.0, epoch_data.seconds_per_epoch * CONSENSUS_FALLBACK_STORE_DEADLINE - epoch_data.seconds_elapsed)
                )

                epoch_data = self.hypertensor.get_epoch_progress()
                current_epoch = epoch_data.epoch

                # Break and move on, consensus likely broken
                if current_epoch > epoch:
                    break

                # Due to pydantic schema in the predicate validator, we only check if it exists,
                # not if it is a tensor.
                prompt_exists = self.dht.get(consensus_key).value is not None

                if prompt_exists:
                    print(f"Prompt already submitted for epoch {epoch}, no fallback needed.")
                    break

                if not prompt_exists and epoch_data.percent_complete >= PROMPT_DEADLINE:
                    self.generate_and_store_random_tensor(consensus_key, consensus_subkey)
                    print(f"Fallback: Submitted prompt for epoch {epoch}")
                    break

    def shutdown(self):
        self.stop.set()
