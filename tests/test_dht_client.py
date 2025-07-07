import asyncio
import concurrent.futures
import os
import random
import time
from typing import List

import pytest

from mesh.dht.crypto import RSASignatureValidator
from mesh.subnet.utils.key import generate_rsa_private_key_file, get_rsa_peer_id, get_rsa_private_key
from mesh.utils.auth import TokenRSAAuthorizerBase

from test_utils.dht_swarms import (
    launch_dht_instances_with_record_validators,
    launch_dht_instances_with_record_validators_bootstrap_no_kwargs,
)

# pytest tests/test_dht_client.py::test_dht_same_clients -rP

@pytest.mark.forked
def test_dht_same_clients(n_peers=10):
    peers_len = 5

    test_paths = []
    record_validators: List[RSASignatureValidator] = []
    token_rsa_validators: List[TokenRSAAuthorizerBase] = []
    for i in range(peers_len):
        test_path = f"rsa_test_path_{i}.key"
        root_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(root_path, test_path)

        if not os.path.exists(full_path):
            private_key, public_key, public_bytes, encoded_public_key, encoded_digest, peer_id = generate_rsa_private_key_file(test_path)

        test_paths.append(test_path)
        loaded_key = get_rsa_private_key(test_path)
        record_validator = RSASignatureValidator(loaded_key)
        record_validators.append(record_validator)

        if i > 0:
            test_paths.append(test_path)
            record_validators.append(record_validator)


    dhts = launch_dht_instances_with_record_validators_bootstrap_no_kwargs(
        record_validators=record_validators,
        identity_paths=test_paths,
        # client_mode=False,
        # check_if_identity_free=False
    )

    for path in test_paths:
        try:
            os.remove(path)
        except:  # noqa: E722
            pass

    for dht in dhts:
        dht.shutdown()
