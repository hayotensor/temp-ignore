import asyncio
import hashlib
import io
import secrets
from typing import Any, Optional, Tuple

import torch

from mesh import DHT, get_dht_time
from mesh.dht.validation import RecordValidatorBase
from mesh.subnet.protocols.inference_protocol import InferenceProtocol
from mesh.subnet.utils.consensus import get_consensus_key
from mesh.subnet.utils.dht import get_many_data, store_data
from mesh.subnet.utils.hoster import get_hoster_commit_key, get_hoster_reveal_key, get_hoster_subkey_rsa
from mesh.substrate.chain_functions_v2 import Hypertensor
from mesh.utils.logging import get_logger

logger = get_logger(__name__)

class Hoster:
    def __init__(
        self,
        dht: DHT,
        inference_protocol: InferenceProtocol,
        record_validator: RecordValidatorBase,
        hypertensor: Hypertensor,
    ):
        self.dht = dht
        self.peer_id = self.dht.peer_id
        self.inference_protocol = inference_protocol
        self.record_validator = record_validator
        self.hypertensor = hypertensor
        self.epoch_length = 0
        # self.epoch_length = int(str(self.hypertensor.get_epoch_length()))

    async def step(self):
        epoch_data = self.get_epoch_progress()
        epoch = epoch_data.epoch
        percent = epoch_data.percent_complete

        consensus_key = get_consensus_key(epoch)
        consensus_tensor = await self.try_load_tensor(consensus_key, epoch)

        if consensus_tensor is None:
            return  # No tensor available yet

        # Call inference on self
        inference_output = await self.inference_protocol.call_inference_stream(
            peer=self.peer_id,
            promt="",
            tensor=consensus_tensor
        )

        if percent <= 0.5:
            await self.commit(epoch, inference_output)
        else:
            await self.reveal(epoch, inference_output)

    async def try_load_tensor_test(self, key: bytes) -> Optional[Any]:
        result = get_many_data(
            self.dht,
            uid=key,
            latest=True,
        )
        return result

    async def try_load_tensor(self, key: bytes) -> Optional[torch.Tensor]:
        """
        Load the validators random tensor for the epoch

        We expect the data to use the recore validator so we expect it to have a
        subkey using their public key and remove that from the value to extract
        the data
        """
        result = get_many_data(
            self.dht,
            uid=key,
            latest=True,
        )
        print("try_load_tensor result", result)
        if result is None:
            return None
        try:
            data = getattr(next(iter(result[key].value.values()), None), "value", None)
            print("try_load_tensor data", data)

            # return torch.load(io.BytesIO(inner_dict))
            return torch.load(io.BytesIO(data), weights_only=False)
        except Exception as e:
            logger.warning(f"Loading tensor failed with: {e}", exc_info=True)
            return None

    def model_commit_fn(self, tensor: torch.Tensor) -> Tuple[bytes, bytes]:
        """
        Create a salted SHA256 hash of the tensor.
        Returns (salt, hash).
        """
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        tensor_bytes = buffer.getvalue()

        salt = secrets.token_bytes(16)
        digest = hashlib.sha256(salt + tensor_bytes).digest()
        return salt, digest

    def commit(self, epoch: int, result: torch.Tensor):
        salt, digest = self.model_commit_fn(result)
        self.last_salt = salt
        store_data(
            dht=self.dht,
            key=get_hoster_commit_key(epoch),
            subkey=get_hoster_subkey_rsa(self.record_validator),
            data=digest,
            expiration_time=get_dht_time() + self.epoch_length,
            wait=True
        )
        logger.info(f"[Hoster] Committed hash for epoch {epoch}")

    def reveal(self, epoch: int, result: torch.Tensor):
        buffer = io.BytesIO()
        torch.save(result, buffer)
        payload = {
            "salt": self.last_salt,
            "tensor": buffer.getvalue(),
        }
        store_data(
            dht=self.dht,
            key=get_hoster_reveal_key(epoch),
            subkey=get_hoster_subkey_rsa(self.record_validator),
            data=payload,
            expiration_time=get_dht_time() + self.epoch_length,
            wait=True
        )
        logger.info(f"[Hoster] Revealed reveal for epoch {epoch}")

    async def run_forever(self, interval: float = 10.0):
        while True:
            await self.step()
            await asyncio.sleep(interval)
