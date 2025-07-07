from typing import Callable

from mesh.dht.crypto import Ed25519SignatureValidator, RSASignatureValidator
from mesh.dht.validation import DHTRecord, PredicateValidator
from mesh.substrate.utils import get_epoch_progress

BASE_VALIDATOR_SCORE = 0.3  # Start from this score
EPSILON = 1e-8    # To avoid division by zero

def get_validator_subkey_rsa(record_validator: RSASignatureValidator) -> bytes:
    return record_validator.local_public_key

def get_validator_subkey_ed25519(record_validator: Ed25519SignatureValidator) -> bytes:
    return record_validator.local_public_key

def get_validator_commit_key(epoch: int) -> str:
    return f"validator-commit_epoch_{epoch}"

def get_validator_reveal_key(epoch: int) -> str:
    return f"validator-reveal_epoch_{epoch}"
