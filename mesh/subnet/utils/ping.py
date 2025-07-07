import asyncio
import math
import threading
import time
from functools import partial
from typing import Dict, Sequence

import mesh
from mesh.proto import dht_pb2
from mesh.utils.logging import get_logger

logger = get_logger(__name__)


async def ping(
    peer_id: mesh.PeerID,
    _dht: mesh.DHT,
    node: mesh.dht.DHTNode,
    *,
    wait_timeout: float = 5,
) -> float:
    try:
        ping_request = dht_pb2.PingRequest(peer=node.protocol.node_info)
        start_time = time.perf_counter()
        await node.protocol.get_stub(peer_id).rpc_ping(ping_request, timeout=wait_timeout)
        return time.perf_counter() - start_time
    except Exception as e:
        if str(e) == "protocol not supported":  # Happens on servers with client-mode DHT (e.g., reachable via relays)
            return time.perf_counter() - start_time

        logger.debug(f"Failed to ping {peer_id}:", exc_info=True)
        return math.inf

async def ping_test(
    peer_id: mesh.PeerID,
    _dht: mesh.DHT,
    node: mesh.dht.DHTNode,
    *,
    wait_timeout: float = 5,
) -> float:
    try:
        ping_request = dht_pb2.PingRequest(peer=node.protocol.node_info)
        start_time = time.perf_counter()
        await node.protocol.get_stub(peer_id).rpc_ping(ping_request, timeout=wait_timeout)
        return time.perf_counter() - start_time
    except Exception as e:
        if str(e) == "protocol not supported":  # Happens on servers with client-mode DHT (e.g., reachable via relays)
            return time.perf_counter() - start_time

        logger.debug(f"Failed to ping {peer_id}:", exc_info=True)
        return math.inf

async def ping_test_parallel(peer_ids: Sequence[mesh.PeerID], *args, **kwargs) -> Dict[mesh.PeerID, float]:
    rpc_infos = await asyncio.gather(*[ping_test(peer_id, *args, **kwargs) for peer_id in peer_ids])
    return dict(zip(peer_ids, rpc_infos))

async def ping_parallel(peer_ids: Sequence[mesh.PeerID], *args, **kwargs) -> Dict[mesh.PeerID, float]:
    rpc_infos = await asyncio.gather(*[ping(peer_id, *args, **kwargs) for peer_id in peer_ids])
    return dict(zip(peer_ids, rpc_infos))


class PingAggregator:
    def __init__(self, dht: mesh.DHT, *, ema_alpha: float = 0.2, expiration: float = 300):
        self.dht = dht
        self.ema_alpha = ema_alpha
        self.expiration = expiration
        self.ping_emas = mesh.TimedStorage()
        self.lock = threading.Lock()

    def ping(self, peer_ids: Sequence[mesh.PeerID], **kwargs) -> None:
        current_rtts = self.dht.run_coroutine(partial(ping_parallel, peer_ids, **kwargs))
        logger.debug(f"Current RTTs: {current_rtts}")

        print("ping current_rtts", current_rtts)

        with self.lock:
            expiration = mesh.get_dht_time() + self.expiration
            print("ping expiration", expiration)
            for peer_id, rtt in current_rtts.items():
                print("ping peer_id", peer_id)
                print("ping rtt", rtt)
                prev_rtt = self.ping_emas.get(peer_id)
                print("ping prev_rtt", prev_rtt)
                if prev_rtt is not None and prev_rtt.value != math.inf:
                    rtt = self.ema_alpha * rtt + (1 - self.ema_alpha) * prev_rtt.value  # Exponential smoothing
                self.ping_emas.store(peer_id, rtt, expiration)

    def to_dict(self) -> Dict[mesh.PeerID, float]:
        with self.lock, self.ping_emas.freeze():
            smoothed_rtts = {peer_id: rtt.value for peer_id, rtt in self.ping_emas.items()}
            print("ping to_dict smoothed_rtts", smoothed_rtts)
            logger.debug(f"Smothed RTTs: {smoothed_rtts}")
            return smoothed_rtts
