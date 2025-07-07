import asyncio
import os
import threading
import time
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

from mesh import PeerID
from mesh.substrate.chain_data import RewardsData
from mesh.substrate.chain_functions import (
    activate_subnet,
    attest,
    get_block_number,
    get_epoch_length,
    get_hotkey_subnet_node_id,
    get_reward_result_event,
    get_rewards_submission,
    get_rewards_validator,
    get_subnet_data,
    get_subnet_id_by_path,
    validate,
)
from mesh.substrate.chain_functions_v2 import Hypertensor
from mesh.substrate.config import BLOCK_SECS
from mesh.substrate.utils import (
    get_included_nodes,
    get_next_epoch_start_block,
    get_submittable_nodes,
)
from mesh.utils import get_logger
from mesh.utils.math import saturating_sub

logger = get_logger(__name__)

MAX_ATTEST_CHECKS = 3


class AttestReason(Enum):
    WAITING = 1
    ATTESTED = 2
    ATTEST_FAILED = 3
    SHOULD_NOT_ATTEST = 4
    NOT_VALIDATOR = 5
    SHUTDOWN = 6


class Consensus(threading.Thread):
    """
    Houses logic for validating and attesting consensus data per epochs for rewards

    This can be ran before or during a model activation.

    If before, it will wait until the subnet is successfully voted in, if the proposal to initialize the subnet fails,
    it will not stop running.

    If after, it will begin to validate and or attest epochs
    """

    def __init__(
        self,
        subnet_id: int,
        subnet_node_id: int,
        peer_id: PeerID,
        hypertensor: Hypertensor,
        identity_path: str,
    ):
        super().__init__()
        assert hypertensor is not None, "hypertensor configuration must be specified"
        self.hypertensor = hypertensor
        self.subnet_id = subnet_id
        self.subnet_node_id = subnet_node_id
        self.subnet_active = False
        self.hotkey = hypertensor.hotkey

        self.is_validator_eligible = False # Can be chosen for validator

        self.identity_path = identity_path # Private key identity path (DHTNode key)
        self.peer_id = peer_id

        self.rpc = self.hypertensor.url
        self.epoch_length = int(str(self.hypertensor.get_epoch_length()))

        self.last_validated_or_attested_epoch = 0
        self.previous_epoch_data = None

        self.stop = threading.Event()
        self.start()

    def run(self):
        """
        Iterates each epoch, runs the incentives mechanism for the SCP
        """
        while not self.stop.is_set():
            try:

                # initialize subnet node ID once we have the subnet ID
                if self.subnet_id is not None and self.subnet_node_id is None:
                    self.subnet_node_id = get_hotkey_subnet_node_id(
                        self.hypertensor.interface,
                        self.subnet_id,
                        self.hotkey,
                    )
                    logger.info(f"Subnet Node ID: {self.subnet_node_id}")

                # get block
                block_number = get_block_number(self.hypertensor.interface)
                logger.info("Block height: %s " % block_number)

                # get epoch
                epoch = int(block_number / self.epoch_length)
                logger.info("Epoch: %s " % epoch)

                next_epoch_start_block = get_next_epoch_start_block(self.epoch_length, block_number)
                remaining_blocks_until_next_epoch = next_epoch_start_block - block_number

                # sleep and skip block if already validated or attested epoch
                if epoch <= self.last_validated_or_attested_epoch and self.subnet_active:
                    time.sleep(remaining_blocks_until_next_epoch * BLOCK_SECS)
                    continue

                # Ensure subnet is activated
                if self.subnet_active is False:
                    # Attempt to activate
                    activated = self._activate_subnet()

                    if self.stop.is_set():
                        break

                    # Restart loop
                    if activated is True:
                      continue
                    else:
                      time.sleep(BLOCK_SECS)
                      continue

                """
                Is subnet node initialized and eligible to submit consensus
                """
                # subnet is eligible to accept consensus
                # check if we are submittable
                # in order to be submittable:
                # - Must stake onchain
                # - Must be Submittable subnet node class
                if self.is_validator_eligible is False:
                    if self.is_self_validator_eligible():
                        self.is_validator_eligible = True
                    else:
                        # If included, query consensus data anyway and save to self.previous_epoch_data
                        if self.is_included():
                            self.attest(epoch, attest=False)
                        time.sleep(remaining_blocks_until_next_epoch * BLOCK_SECS)
                        continue

                # is epoch submitted yet

                # is validator?
                validator = self._get_validator(epoch)

                # a validator is not chosen if there are not enough nodes, or the subnet is deactivated
                if validator is None:
                    logger.info("Validator not chosen for epoch %s yet, checking next block" % epoch)
                    time.sleep(BLOCK_SECS)
                    continue
                else:
                    logger.info("Validator for epoch %s is Subnet Node ID %s" % (epoch, validator))

                is_validator = validator == self.subnet_node_id
                if is_validator:
                    logger.info("We're the chosen validator ID for epoch %s, validating and auto-attesting..." % epoch)
                    # check if validated
                    validated = self._get_validator_consensus_submission(epoch)
                    if validated is None:
                        success = self.validate()
                        # update last validated epoch and continue (this validates and attests in one call)
                        if success:
                            self.last_validated_or_attested_epoch = epoch
                        else:
                            logger.warning("Consensus submission unsuccessful, waiting until next block to try again")
                            time.sleep(BLOCK_SECS)
                            continue
                    else:
                        # if for any reason on the last attempt it succeeded but didn't propogate
                        # because this section should only be called once per epoch and if validator until successful submission of data
                        self.last_validated_or_attested_epoch = epoch

                    # continue to next epoch, no need to attest
                    time.sleep(remaining_blocks_until_next_epoch * BLOCK_SECS)
                    continue

                # we are not validator, we must attest or not attest
                # wait until validated by epochs chosen validator

                # get epoch before waiting for validator to validate to ensure we don't get stuck
                initial_epoch = epoch
                attest_checks = 0
                logger.info("Starting attestation check")
                while True:
                    # wait for validator on every block
                    time.sleep(BLOCK_SECS)
                    block_number = get_block_number(self.hypertensor.interface)
                    logger.info("Block height: %s " % block_number)

                    epoch = int(block_number / self.epoch_length)
                    logger.info("Epoch: %s " % epoch)

                    next_epoch_start_block = get_next_epoch_start_block(self.epoch_length, block_number)
                    remaining_blocks_until_next_epoch = next_epoch_start_block - block_number

                    # If we made it to the next epoch, break
                    # This likely means the chosen validator never submitted consensus data
                    if epoch > initial_epoch:
                        logger.info(
                            "Validator didn't submit epoch %s consensus data, moving to the next epoch" % epoch
                        )
                        break

                    if attest_checks > MAX_ATTEST_CHECKS:
                        logger.info("Failed to attest after %s checks, moving to the next epoch" % attest_checks)
                        break

                    attest_result, reason = self.attest(epoch)
                    if attest_result is False:
                        attest_checks += 1
                        if reason == AttestReason.WAITING or reason == AttestReason.ATTEST_FAILED:
                            continue
                        elif reason == AttestReason.ATTESTED:
                            # redundant update on `last_validated_or_attested_epoch`
                            self.last_validated_or_attested_epoch = epoch
                            break
                        elif reason == AttestReason.SHOULD_NOT_ATTEST:
                            # sleep until end of epoch to check if we should attest

                            # sleep until latter half of the epoch to attest
                            delta = remaining_blocks_until_next_epoch / 2

                            # ensure attestor has at least 2 blocks to run compute
                            if delta / 2 < BLOCK_SECS * 2:
                                delta = 0

                            time.sleep(saturating_sub(delta * BLOCK_SECS, BLOCK_SECS))
                            continue
                        elif reason == AttestReason.NOT_VALIDATOR:
                            # Retrieved consensus data for symmetry, but not validator so skipping attestation
                            self.last_validated_or_attested_epoch = epoch
                            break
                        elif reason == AttestReason.SHUTDOWN:
                            # break and reset loop
                            self.last_validated_or_attested_epoch = epoch
                            break

                        # If False, still waiting for validator to submit data
                        continue
                    else:
                        # successful attestation, break and go to next epoch
                        self.last_validated_or_attested_epoch = epoch
                        break
            except Exception as e:
                logger.error("Consensus Error: %s" % e, exc_info=True)

    def validate(self) -> bool:
        """
        Calculate incentives data based on the scoring protocol and submit consensus

        Returns:
          bool: If successful
        """
        # TODO: Add exception handling
        consensus_data = self._get_consensus_data()
        if consensus_data is None:
            return False

        return self._do_validate(consensus_data)

    def attest(self, epoch: int, attest: Optional[bool] = True) -> Tuple[bool, AttestReason]:
        """
        1. Fetches validator incentives data from the blockchain
        2. Calculates incentives data based on the incentives protocol
        3. Compares data to see if should attest
        4. Attests if should attest

        Args:
          epoch (int): Current epoch.
          attest (bool): (
            True=will attest if should attest
            False=will run function but not perform attestation. Used to save previous_epoch_data
          )

        Returns:
          Tuple[bool, AttestReason]: If attested (True|Ralse), reason for ``attested``
        """
        validator_consensus_submission = self._get_validator_consensus_submission(epoch)

        if validator_consensus_submission is None:
            logger.info("Waiting for validator to submit")
            return False, AttestReason.WAITING

        # backup check if validator node restarts in the middle of an epoch to ensure they don't tx again
        if self._has_attested(validator_consensus_submission["attests"]):
            logger.info("Has attested already")
            return False, AttestReason.ATTESTED

        validator_consensus_data = RewardsData.list_from_scale_info(validator_consensus_submission["data"])

        logger.info("Checking if we should attest the validators submission")
        logger.info("Generating consensus data")
        consensus_data = self._get_consensus_data()  # should always return `peers` key

        # if not in validator data, check if we're still Submittable
        # this is in case we exit on-chain before shutting the node down
        in_validator_data = any(self.peer_id.__eq__(r.peer_id) for r in validator_consensus_data)
        if not in_validator_data and attest:
            is_self_validator_eligible = self.is_self_validator_eligible()
            if not is_self_validator_eligible:
                logger.warning("We are not Submittable, shutting down consensus class")
                self.shutdown()
                return False, AttestReason.SHUTDOWN

        should_attest = self.should_attest(validator_consensus_data, consensus_data, epoch)
        logger.info("Should attest is: %s", should_attest)

        if should_attest and attest:
            logger.info("Validators data is confirmed valid, attesting data...")
            attest_is_success = self._do_attest()
            if attest_is_success:
                return True, AttestReason.ATTESTED
            else:
                return False, AttestReason.ATTEST_FAILED
        elif should_attest is False and attest:
            return False, AttestReason.SHOULD_NOT_ATTEST
        else:
            logger.info("Retrieved data for symmetry but not Validator class yet, skipping attestation.")
            return False, AttestReason.NOT_VALIDATOR

    def is_self_validator_eligible(self) -> bool:
        submittable_nodes = get_submittable_nodes(
            self.hypertensor.interface,
            self.subnet_id,
        )

        _is = False
        #  wait until we are submittable
        for node_set in submittable_nodes:
            if node_set.hotkey == self.hotkey:
                _is = True
                break

        return _is

    def is_included(self) -> bool:
        included_nodes = get_included_nodes(
            self.hypertensor.interface,
            self.subnet_id,
        )

        _is = False
        #  wait until we are submittable
        for node_set in included_nodes:
            if node_set.hotkey == self.hotkey:
                _is = True
                break

        return _is

    def get_rps(self): ...

    def _do_validate(self, data) -> bool:
        try:
            receipt = validate(self.hypertensor.interface, self.hypertensor.keypair, self.subnet_id, data)
            return receipt.is_success
        except Exception as e:
            logger.error("Validation Error: %s" % e)
            return False

    def _do_attest(self) -> bool:
        try:
            receipt = attest(
                self.hypertensor.interface,
                self.hypertensor.keypair,
                self.subnet_id,
            )
            return receipt.is_success
        except Exception as e:
            logger.error("Attestation Error: %s" % e)
            return False

    def _get_consensus_data(self):
        """
        TODO:

        Get all hosters data from validators in storagae

        Score all validators based on their commits + (stake balance?)
        """
        consensus_data = []
        return consensus_data

    def _get_validator_consensus_submission(self, epoch: int):
        """Get and return the consensus data from the current validator from Hypertensor"""
        rewards_submission = get_rewards_submission(self.hypertensor.interface, self.subnet_id, epoch)
        return rewards_submission

    def _has_attested(self, attestations) -> bool:
        """Get and return the consensus data from the current validator from Hypertensor"""
        for data in attestations:
            # data = { subnet_node_id: block_number}
            if data[0] == self.subnet_node_id:
                return True
        return False

    def _get_validator(self, epoch):
        validator = get_rewards_validator(self.hypertensor.interface, self.subnet_id, epoch)
        return validator

    def _activate_subnet(self):
        """
        Activates subnet

        - If in registration period will wait for the subnet to be able to be activated
        - Subnet nodes will wait their turn to activate based on index of entry

        Returns:
          bool: If activated
        """

        return True

    def should_attest(self, validator_data, my_data, epoch):
        """Checks if two arrays of dictionaries match, regardless of order."""

        # if validator submitted no data, and we have also found the subnet is broken
        if len(validator_data) == 0 and len(my_data) == 0:
            return True

        # otherwise, check the data matches
        # at this point, the

        # use ``asdict`` because data is decoded from blockchain as dataclass
        # we assume the lists are consistent across all elements
        # Convert validator_data to a set
        set1 = set(frozenset(asdict(d).items()) for d in validator_data)

        # Convert my_data to a set
        set2 = set(frozenset(d.items()) for d in my_data)

        logger.info(f"Validator data: {set1}")
        logger.info(f"Our data:       {set2}")

        success = set1 == set2

        if not success:
            logger.debug("ATTEST L1: validator data: %s, attestor_data: %s" % (set1, set2))

        """
        The following accounts for nodes that go down or back up in the after or before validation submissions and attestations

        # Cases

        Case 1: Node leaves DHT before validator submit consensus and returns before attestation.
                - Validator data does not include node, Attestors data will include node, creating a mismatch.
        Case 2: Node leaves DHT after validator submits consensus.
                - Validator data does include node, Attestors data does not include node, creating a mismatch.

        # Solution

        We check our previous epochs data, if successfully attested, to find symmetry with the validators data.

        * If none of these solutions work, we assume the validator is being dishonest
        """
        if not success and self.previous_epoch_data is not None:
            dif = set1.symmetric_difference(set2)
            success = dif.issubset(self.previous_epoch_data)
            if not success:
                logger.debug("ATTEST L2: validator data: %s, attestor_data: %s, dif: %s" % (set1, set2, dif))
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
                    dif = set1.symmetric_difference(set2)
                    success = dif.issubset(previous_epoch_data_onchain)
        # else:
        #     # log only data
        #     intersection = set1.intersection(set2)
        #     logger.info(
        #         "Matching intersection of %s validator data" % (saturating_div(len(intersection), len(set1)) * 100)
        #     )
        #     logger.info(
        #         "Validator matching intersection of %s my data" % (saturating_div(len(intersection), len(set2)) * 100)
        #     )

        # update previous epoch data
        self.previous_epoch_data = set2

        return success

    def _get_reward_result(self, epoch: int):
        try:
            event = self.hypertensor.get_reward_result_event(self.self.subnet_id, epoch)
            subnet_id, attestation_percentage = event["event"]["attributes"]
            return subnet_id, attestation_percentage
        except Exception as e:
            logger.warning("Reward Result Error: %s" % e, exc_info=True)
            return None

    def shutdown(self):
        logger.info("Shutting down consensus")
        self.stop.set()
