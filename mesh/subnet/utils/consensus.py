import io
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from pydantic import BaseModel, ConfigDict, ValidationError

from mesh import get_dht_time, get_logger
from mesh.dht.crypto import Ed25519SignatureValidator, RSASignatureValidator
from mesh.dht.routing import DHTID
from mesh.dht.validation import DHTRecord, DHTRequestType
from mesh.substrate.chain_functions_v2 import EpochData
from mesh.substrate.config import BLOCK_SECS, EPOCH_LENGTH
from mesh.utils.serializer import MSGPackSerializer

logger = get_logger(__name__)

HOSTER_SCORE_RATIO = 0.4
VALIDATOR_SCORE_RATIO = 0.6

"""
Consensus scores
"""
@dataclass
class ConsensusScores:
  peer_id: str
  score: int

"""
Hoster scores
"""
@dataclass
class HosterResult:
    peer: str
    output: Optional[torch.Tensor]
    success: bool

"""
Validator scores
"""
@dataclass
class ValidatorScores:
  peer_id: str
  score: int

"""
A default fallback tensor in case anything goes wrong
"""
DEFAULT_CONSENSUS_TENSOR = ""

def get_consensus_subkey_rsa(record_validator: RSASignatureValidator) -> bytes:
    return record_validator.local_public_key

def get_consensus_subkey_ed25519(record_validator: Ed25519SignatureValidator) -> bytes:
    return record_validator.local_public_key

def get_consensus_key(epoch: int) -> str:
    """
    Key for storing the epochs random prompt tensors for hosters to mine
    """
    return f"consensus-epoch_{epoch}"

"""
Commit-Reveal Schema using the PredicateValidator
"""

"""
Hypertensor predicate validator

Verified allowable keys, schemas, epoch relationships, and commit-reveal schemes.

- Roles
    - Validates "hoster" and "validator" role kes
- Hoster
    - Validates key relationship to epoch
    - Validates commit-reveal
- Validator
    - Validates key relationship to epoch
    - Validates commit-reveal
- Consensus
    - Validates tensors via pydantic schema
"""


class ConsensusTensorModel(BaseModel):
    tensor: torch.Tensor
    model_config = ConfigDict(arbitrary_types_allowed=True)

# Created At validations
# 0-15%
CONSENSUS_STORE_DEADLINE = 0.15
# This is a helper for nodes to store if the elected node doesn't
# submit the random prompt for consensus after the `CONSENSUS_STORE_DEADLINE` up to `CONSENSUS_STORE_DEADLINE`
CONSENSUS_FALLBACK_STORE_DEADLINE = 0.10

# hoster commit-reveal
HOSTER_COMMIT_DEADLINE = 0.5
HOSTER_REVEAL_DEADLINE = 0.6

# validator commit-reveal on scores

# commit=60-100%
VALIDATOR_COMMIT_START = 0.6
VALIDATOR_COMMIT_DEADLINE = 1.0

# reveal=50-60% (reveals previous epochs commit) ⸺ same as hoster
VALIDATOR_REVEAL_START = 0.5
VALIDATOR_REVEAL_DEADLINE = 0.6

# Expiration validations
MAX_HEART_BEAT_TIME = BLOCK_SECS * EPOCH_LENGTH * 1.1
MAX_CONSENSUS_TIME = BLOCK_SECS * EPOCH_LENGTH * 1.1 # random prompt
MAX_HOSTER_COMMIT_TIME = BLOCK_SECS * EPOCH_LENGTH
MAX_HOSTER_REVEAL_TIME = BLOCK_SECS * EPOCH_LENGTH
MAX_VALIDATOR_COMMIT_TIME = BLOCK_SECS * EPOCH_LENGTH * 2
MAX_VALIDATOR_REVEAL_TIME = BLOCK_SECS * EPOCH_LENGTH * 2

def hypertensor_consensus_predicate() -> Callable[[DHTRecord], bool]:
    def predicate(record: DHTRecord, type: DHTRequestType, epoch_data: EpochData) -> bool:
        try:
            # Enable GET data at any time
            if type is DHTRequestType.GET:
                return True

            current_epoch = epoch_data.epoch
            percent_complete = epoch_data.percent_complete

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

            key_type = valid_keys.get(record.key, None)

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
                    return False

                max_expiration = dht_time + MAX_CONSENSUS_TIME
                if record.expiration_time > max_expiration:
                    return False
                try:
                    loaded = MSGPackSerializer.loads(record.value)
                    tensor = torch.load(io.BytesIO(loaded), weights_only=False)
                    ConsensusTensorModel(tensor=tensor)
                    return True
                except ValidationError:
                    return False

            # ⸺ 15-50%
            elif key_type == "validator-reveal":
                max_expiration = dht_time + MAX_VALIDATOR_REVEAL_TIME
                if record.expiration_time > max_expiration:
                    return False
                if percent_complete <= VALIDATOR_REVEAL_START and percent_complete > VALIDATOR_REVEAL_DEADLINE:
                    return False
                return True

            # ⸺ 15-50%
            elif key_type == "hoster-commit":
                max_expiration = dht_time + MAX_HOSTER_COMMIT_TIME
                if record.expiration_time > max_expiration:
                    return False
                if percent_complete <= CONSENSUS_STORE_DEADLINE and percent_complete > HOSTER_COMMIT_DEADLINE:
                    return False

                return True

            # ⸺ 50-60%
            elif key_type == "hoster-reveal":
                max_expiration = dht_time + MAX_HOSTER_REVEAL_TIME
                if record.expiration_time > max_expiration:
                    return False
                # if percent_complete > HOSTER_COMMIT_DEADLINE and percent_complete <= HOSTER_REVEAL_DEADLINE:
                if percent_complete <= HOSTER_COMMIT_DEADLINE and percent_complete > HOSTER_REVEAL_DEADLINE:
                    return False
                return True

            # ⸺ 60-100%
            elif key_type == "validator-commit":
                max_expiration = dht_time + MAX_VALIDATOR_COMMIT_TIME
                if record.expiration_time > max_expiration:
                    return False
                # if percent_complete >= VALIDATOR_COMMIT_START:
                if percent_complete < VALIDATOR_COMMIT_START:
                    return False
                return True

            return False  # Key doesn't match any known schema
        except Exception as e:
            print(f"Predicate Err: {e}")
            return False

    return predicate
