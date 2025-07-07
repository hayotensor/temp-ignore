import asyncio
import threading
from dataclasses import asdict
from typing import List, Optional

from mesh import DHT, get_dht_time
from mesh.dht.validation import RecordValidatorBase
from mesh.subnet.data_structures import ServerClass
from mesh.subnet.roles.hoster import Hoster
from mesh.subnet.roles.validator import Validator
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

_max_consensus_time = MAX_CONSENSUS_TIME - 60

class Consensus(threading.Thread):
    def __init__(
        self,
        dht: DHT,
        subnet_id: int,
        subnet_node_id: int,
        role: ServerClass,
        record_validator: RecordValidatorBase,
        hypertensor: Hypertensor,
        converted_model_name_or_path: str,
        validator: Validator,
        hoster: Optional[Hoster] = None,
        skip_activate_subnet: bool = False,
        start: bool = True
    ):
        super().__init__(daemon=True)
        self.dht = dht
        self.peer_id = self.dht.peer_id
        self.subnet_id = subnet_id
        self.subnet_node_id = subnet_node_id
        self.role = role
        self.hypertensor = hypertensor
        self.record_validator = record_validator
        self.validator = validator
        self.hoster = hoster
        self.prompt_generator = RandomPrompts(converted_model_name_or_path)
        self.hoster_scores = None
        self.validator_scores = None
        self.previous_epoch_data = None
        self.is_subnet_active = False
        self.skip_activate_subnet = skip_activate_subnet

        self.stop = threading.Event()

        if start:
            self.run()

    # def is_validator(self, epoch: int) -> bool:
    #     validator = self.get_validator(epoch)
    #     return validator == self.subnet_node_id

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

    def _get_reward_result(self, epoch: int):
        try:
            event = self.hypertensor.get_reward_result_event(self.self.subnet_id, epoch)
            subnet_id, attestation_percentage = event["event"]["attributes"]
            return subnet_id, attestation_percentage
        except Exception as e:
            logger.warning("Reward Result Error: %s" % e, exc_info=True)
            return None

    def run(self):
        self.is_subnet_active = asyncio.run(self.run_activate_subnet())
        if not self.is_subnet_active:
            return

        # self.is_node_active = asyncio.run(self.run_activate_node())
        # if not self.is_node_active:
        #     return

        self.is_node_validator = asyncio.run(self.run_is_node_validator())
        if not self.is_node_validator:
            return

        asyncio.run(self.run_forever())

    async def run_activate_subnet(self):
        """
        Verify subnet is active on-chain before starting consensus

        For initial coldkeys this will sleep until the enactment period, then proceed
        to check once per epoch after enactment starts if the owner activated the subnet
        """
        # Useful if subnet is already active and for testing
        if self.skip_activate_subnet:
            return True

        last_epoch = None
        subnet_registration_epochs = self.hypertensor.get_subnet_registration_epochs()
        subnet_active = False
        errors_count = 0
        while not self.stop.is_set():
            epoch_data = self.hypertensor.get_epoch_progress()
            current_epoch = epoch_data.epoch

            if current_epoch != last_epoch:
                seconds_per_epoch = epoch_data.seconds_per_epoch
                offset_sleep = 0
                subnet_info = self.hypertensor.get_formatted_subnet_info(
                    self.subnet_id
                )
                if subnet_info is None:
                    if errors_count > 3:
                        logger.warning("Cannot find subnet ID: %s, shutting down", self.subnet_id)
                        self.shutdown()
                        subnet_active = False
                        break
                    else:
                        errors_count = errors_count + 1
                else:
                    if subnet_info.state == "Active":
                        subnet_active = True
                        break

                    # Still in registration period
                    max_registration_epoch = subnet_registration_epochs + subnet_info.registration_epoch
                    registration_epochs_remaining = max_registration_epoch - current_epoch
                    offset_sleep = seconds_per_epoch * registration_epochs_remaining - epoch_data.seconds_elapsed
                    # Wait until enactment period
                    await asyncio.sleep(
                        max(0.0, offset_sleep)
                    )

                last_epoch = current_epoch
            else:
                await asyncio.sleep(epoch_data.seconds_remaining)
                continue

            await asyncio.sleep(
                max(0.0, seconds_per_epoch - epoch_data.seconds_elapsed - offset_sleep)
            )

        return subnet_active

    # async def run_activate_node(self):
    #     """
    #     Verify node is active on-chain before starting consensus

    #     Node must be classed as Included on-chain to to start consensus

    #     Included nodes cannot be the elected validator or attest but must take part in consensus
    #     and be included in the consensus data to graduate to a Validator classed node
    #     """
    #     subnet_data = self.hypertensor.get_formatted_subnet_node_data(self.subnet_id, self.subnet_node_id)
    #     is_activated = False

    #     if self._is_subnet_node_activated(subnet_data.classification.node_class):
    #         is_activated = True

    #     activation_grace_epochs = self.hypertensor.get_activation_grace_epochs(self.subnet_id)
    #     start_epoch = subnet_data.classification.start_epoch
    #     max_activation_epoch = activation_grace_epochs + start_epoch

    #     last_epoch = None
    #     while not self.stop.is_set() and not is_activated:
    #         epoch_data = self.hypertensor.get_epoch_progress()
    #         current_epoch = epoch_data.epoch

    #         if current_epoch > max_activation_epoch:
    #             # Before break, double check they aren't already activated
    #             subnet_data = self.hypertensor.get_formatted_subnet_node_data(self.subnet_id, self.subnet_node_id)
    #             if self._is_subnet_node_activated(subnet_data.classification.node_class):
    #                 is_activated = True
    #             break

    #         if current_epoch != last_epoch:
    #             self.hypertensor.activate_subnet_node(
    #                 self.subnet_id,
    #                 self.subnet_node_id,
    #             )


    #             last_epoch = current_epoch
    #         else:
    #             await asyncio.sleep(epoch_data.seconds_remaining)
    #             continue

    #         await asyncio.sleep(epoch_data.seconds_remaining)

    #     return True

    def _is_subnet_node_activated(self, node_class) -> bool:
        if (
            node_class == "Idle" or
            node_class == "Included" or
            node_class == "Validator"
        ):
            return True

        return False

    async def run_is_node_validator(self):
        """
        Verify node is active on-chain before starting consensus

        Node must be classed as Included on-chain to to start consensus

        Included nodes cannot be the elected validator or attest but must take part in consensus
        and be included in the consensus data to graduate to a Validator classed node
        """
        last_epoch = None
        while not self.stop.is_set():
            epoch_data = self.hypertensor.get_epoch_progress()
            current_epoch = epoch_data.epoch

            if current_epoch != last_epoch:
                nodes = self.hypertensor.get_subnet_validator_nodes(self.subnet_id)
                node_found = False
                for node in nodes:
                    if node.id == self.subnet_node_id:
                        node_found = True
                        break

                if not node_found:
                    logger.info(
                        "Subnet Node ID %s is not Validator class on epoch %s. Trying again in one epoch", self.subnet_node_id, current_epoch
                    )

                last_epoch = current_epoch
            else:
                await asyncio.sleep(epoch_data.seconds_remaining)
                continue

            await asyncio.sleep(epoch_data.seconds_remaining)

        return True

    async def run_forever(self):
        """
        Loop until a new epoch to found, then run consensus logic
        """

        self._async_stop_event = asyncio.Event()
        last_epoch = None

        while not self.stop.is_set() and not self._async_stop_event.is_set():
            epoch_data = self.hypertensor.get_epoch_progress()
            current_epoch = epoch_data.epoch

            if current_epoch != last_epoch:
                # Attest/Validate + random              ⸺ 0-15%
                await self.run_consensus(current_epoch)

                # Commit inference                      ⸺ 15-50%
                if self.role is ServerClass.HOSTER and self.hoster is not None:
                    await self.hoster.run()

                if self.role is ServerClass.VALIDATOR:
                    # Reveal inference                  ⸺ 50-60%
                    await self.validator.run_reveal(current_epoch)
                else:
                    # Reveal scores from previous epoch ⸺ 50-60%
                    await self.hoster.reveal(current_epoch)

                # Score nodes                           ⸺ 60-100%
                scores = await self.validator.score_nodes(current_epoch)

                # Commit nodes scores                   ⸺ 60-100%
                # This takes place after the Hoster reveals
                if self.role is ServerClass.VALIDATOR:
                    await self.validator.commit(scores, current_epoch)

                last_epoch = current_epoch
            else:
                await asyncio.sleep(epoch_data.seconds_remaining)
                continue

            # Keep sleep interval accurate to blockchain clock
            epoch_data = self.hypertensor.get_epoch_progress()
            epoch_data.seconds_remaining

            try:
                await asyncio.wait_for(self._async_stop_event.wait(), timeout=epoch_data.seconds_remaining)
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

    async def run_consensus(self, epoch: int):
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

        validator = None
        # Wait until validator is chosen
        while not self.stop.is_set():
            validator = self.get_validator(epoch)

            epoch_data = self.hypertensor.get_epoch_progress()

            # If validator hasn't submitted by the CONSENSUS_FALLBACK_STORE_DEADLINE (10%)
            # mark of the epoch, break and don't attest
            if epoch_data.seconds_per_epoch * CONSENSUS_FALLBACK_STORE_DEADLINE > epoch_data.seconds_elapsed:
                break

            if validator is None:
                await asyncio.sleep(BLOCK_SECS)
                continue

        if validator == self.subnet_node_id:
            print(f"Acting as validator for epoch {epoch}")

            if len(scores) == 0:
                self.hypertensor.validate(self.subnet_id, data=scores)
            else:
                self.hypertensor.validate(self.subnet_id, data=scores)

            """
            Generate random tensors for next epoch and store to DHT

            1. Check if we alrady stored it (Redundant)
            2. Store it
            """
            consensus_records = self.dht.get(consensus_key) or {}
            if not consensus_records:
                self.generate_and_store_random_tensor(consensus_key, consensus_subkey)
            else:
                # Check if we have submitted consensus prompt already
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
        elif validator is not None:
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
                    break
                elif validator_data is None:
                    await asyncio.sleep(BLOCK_SECS)
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
        This runs whether or not we are validator or attestor

        1. Sleep until 10% into the epoch
        2. Check if the validator submitted a prompt
        3. If they haven't, submit one
        """
        while not self.stop.is_set() and validator != self.subnet_node_id:
            epoch_data = self.hypertensor.get_epoch_progress()

            # Wait until 10% of the progress into an epoch to check if prompt has been submitted
            # Validator has up to the 15% of the epoch to submit random prompt, if they do not by the
            # 10%, submit our own
            await asyncio.sleep(
                max(
                    0.0,
                    epoch_data.seconds_per_epoch * CONSENSUS_FALLBACK_STORE_DEADLINE - epoch_data.seconds_elapsed
                )
            )

            # Get epoch data again to ensure we're on track with the Hypertensor clock
            epoch_data = self.hypertensor.get_epoch_progress()
            current_epoch = epoch_data.epoch

            # Break and move on, consensus likely broken
            if current_epoch > epoch:
                break

            # Due to pydantic schema in the predicate validator, we only check if it exists,
            # not if it is a tensor.
            prompt_exists = self.dht.get(consensus_key) or {}

            if prompt_exists:
                print(f"Prompt already submitted for epoch {epoch}, no fallback needed.")
                break

            # Submit fallback prompt
            # Prompt used will be the first one chosen
            if not prompt_exists and epoch_data.percent_complete >= CONSENSUS_FALLBACK_STORE_DEADLINE:
                self.generate_and_store_random_tensor(consensus_key, consensus_subkey)
                print(f"Fallback: Submitted prompt for epoch {epoch}")
                break

    def shutdown(self):
        self.stop.set()
