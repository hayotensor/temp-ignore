from typing import Callable

from mesh.dht.crypto import Ed25519SignatureValidator, RSASignatureValidator
from mesh.dht.validation import DHTRecord, PredicateValidator
from mesh.substrate.utils import get_epoch_progress

MAX_HOSTER_TOKENS = 50

"""
Commit-reveal interval.
All commits must be submitted by this percentage of the epoch, all reveals after.
"""
HOSTER_COMMIT_CUTOFF = 0.5

def get_hoster_subkey_rsa(record_validator: RSASignatureValidator) -> bytes:
    return record_validator.local_public_key

def get_hoster_subkey_ed25519(record_validator: Ed25519SignatureValidator) -> bytes:
    return record_validator.local_public_key

"""
The shared commot-reveal keys

Uniqueness is done within the subkey
"""
def get_hoster_commit_key(epoch: int) -> str:
    return f"hoster-commit_epoch_{epoch}"

def get_hoster_reveal_key(epoch: int) -> str:
    return f"hoster-reveal_epoch_{epoch}"
