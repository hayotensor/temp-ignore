import io
from typing import Callable

import torch
from pydantic import ValidationError

from mesh import get_dht_time
from mesh.dht.routing import DHTID
from mesh.dht.validation import DHTRecord
from mesh.subnet.utils.consensus import (
    CONSENSUS_STORE_DEADLINE,
    HOSTER_COMMIT_DEADLINE,
    HOSTER_REVEAL_DEADLINE,
    MAX_CONSENSUS_TIME,
    MAX_HEART_BEAT_TIME,
    MAX_HOSTER_COMMIT_TIME,
    MAX_HOSTER_REVEAL_TIME,
    MAX_VALIDATOR_COMMIT_TIME,
    MAX_VALIDATOR_REVEAL_TIME,
    VALIDATOR_COMMIT_START,
    VALIDATOR_REVEAL_DEADLINE,
    VALIDATOR_REVEAL_START,
    ConsensusTensorModel,
)
from mesh.substrate.chain_functions_v2 import EpochData
from mesh.utils.serializer import MSGPackSerializer


def hypertensor_consensus_predicate() -> Callable[[DHTRecord], bool]:
    def predicate(record: DHTRecord, epoch_data: EpochData) -> bool:
        try:
            current_epoch = epoch_data.epoch
            percent_complete = epoch_data.percent_complete
            print(f"hypertensor_consensus_predicate current_epoch {current_epoch}")
            print(f"hypertensor_consensus_predicate percent_complete {percent_complete}")

            # Ensure the keys are valid for the current allowable keys or epoch allowable keys
            valid_keys = {
                # Heartbeat
                DHTID.generate(source="hoster").to_bytes(): "hoster",
                # Heartbeat
                DHTID.generate(source="validator").to_bytes(): "validator",
                # ⸺ 0-15%
                DHTID.generate(source=f"consensus-epoch_{current_epoch}").to_bytes(): "consensus",
                # ⸺ 15-50%
                DHTID.generate(source=f"validator-reveal_epoch_{current_epoch}").to_bytes(): "validator-reveal",
                # ⸺ 15-50%
                DHTID.generate(source=f"hoster-commit_epoch_{current_epoch}").to_bytes(): "hoster-commit",
                # ⸺ 50-60%
                DHTID.generate(source=f"hoster-reveal_epoch_{current_epoch}").to_bytes(): "hoster-reveal",
                # ⸺ 60-100% - Reveals the n-1 epoch commit (stores in the current)
                DHTID.generate(source=f"validator-commit_epoch_{current_epoch}").to_bytes(): "validator-commit",
            }

            print(f"hypertensor_consensus_predicate record {record}")
            print(f"hypertensor_consensus_predicate record.key {record.key}")

            key_type = valid_keys.get(record.key, None)
            print(f"hypertensor_consensus_predicate key_type {key_type}")

            if key_type is None:
                return False

            dht_time = get_dht_time()

            if key_type == "hoster":
                max_expiration = dht_time + MAX_HEART_BEAT_TIME
                if record.expiration_time > max_expiration:
                    return False
                # TODO: validate proof-of-stake on each heartbeat (redundant)
                return True

            elif key_type == "validator":
                max_expiration = dht_time + MAX_HEART_BEAT_TIME
                if record.expiration_time > max_expiration:
                    return False
                # TODO: validate proof-of-stake on each heartbeat (redundant)
                return True

            # ⸺ 0-15%
            elif key_type == "consensus":
                # Must be submitted before deadline
                if percent_complete > CONSENSUS_STORE_DEADLINE:
                    print("Store consensus error=past deadline")
                    return False

                max_expiration = dht_time + MAX_CONSENSUS_TIME
                if record.expiration_time > max_expiration:
                    print("Store consensus error=past max_expiration")
                    return False
                try:
                    loaded = MSGPackSerializer.loads(record.value)
                    tensor = torch.load(io.BytesIO(loaded), weights_only=False)
                    ConsensusTensorModel(tensor=tensor)
                    return True
                except ValidationError:
                    print("Store consensus error=pydantic schema")
                    return False

                return False

            # ⸺ 15-50%
            elif key_type == "validator-reveal":
                max_expiration = dht_time + MAX_VALIDATOR_REVEAL_TIME
                if record.expiration_time > max_expiration:
                    print("Store validator-reveal error=past max_expiration")
                    return False
                if percent_complete > VALIDATOR_REVEAL_START and percent_complete <= VALIDATOR_REVEAL_DEADLINE:
                    print("Store validator-reveal error=past deadline")
                    return True
                return False

            # ⸺ 15-50%
            elif key_type == "hoster-commit":
                max_expiration = dht_time + MAX_HOSTER_COMMIT_TIME
                if record.expiration_time > max_expiration:
                    print("Store hoster-commit error=past max_expiration")
                    return False
                if percent_complete > CONSENSUS_STORE_DEADLINE and percent_complete <= HOSTER_COMMIT_DEADLINE:
                    print("Store hoster-commit error=past deadline")
                    return True
                return False

            # ⸺ 50-60%
            elif key_type == "hoster-reveal":
                max_expiration = dht_time + MAX_HOSTER_REVEAL_TIME
                if record.expiration_time > max_expiration:
                    print("Store hoster-reveal error=past max_expiration")
                    return False
                if percent_complete > HOSTER_COMMIT_DEADLINE and percent_complete <= HOSTER_REVEAL_DEADLINE:
                    print("Store hoster-reveal error=past deadline")
                    return True
                return False

            # ⸺ 60-100%
            elif key_type == "validator-commit":
                max_expiration = dht_time + MAX_VALIDATOR_COMMIT_TIME
                if record.expiration_time > max_expiration:
                    print("Store validator-commit error=past max_expiration")
                    return False
                if percent_complete >= VALIDATOR_COMMIT_START:
                    print("Store validator-commit error=past deadline")
                    return True
                return False

            return False  # Key doesn't match any known schema
        except Exception as e:
            print(f"Predicate Err: {e}")
            return False

    return predicate
