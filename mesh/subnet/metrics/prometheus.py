import datetime
import hashlib
import io
import pickle
import statistics
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Process
from threading import Thread
from typing import Dict, List, Optional

import torch
from prometheus_client import Gauge, start_http_server

from mesh import DHT, PeerID
from mesh.subnet.data_structures import RemoteModuleInfo, ServerState
from mesh.subnet.metrics.config import INITIAL_PEERS, UPDATE_PERIOD
from mesh.subnet.utils.consensus import HOSTER_SCORE_RATIO, VALIDATOR_SCORE_RATIO, ConsensusScores, HosterResult
from mesh.subnet.utils.dht import get_node_infos
from mesh.subnet.utils.hoster import get_hoster_commit_key, get_hoster_reveal_key
from mesh.subnet.utils.key import extract_rsa_peer_id
from mesh.subnet.utils.validator import (
    BASE_VALIDATOR_SCORE,
    EPSILON,
    get_validator_commit_key,
    get_validator_reveal_key,
)
from mesh.substrate.chain_functions_v2 import Hypertensor
from mesh.utils.multiaddr import Multiaddr
from mesh.utils.p2p_utils import check_reachability_parallel, extract_peer_ip_info, get_peers_ips


class MetricsServer:
    def __init__(
        self,
        hypertensor: Hypertensor,
        dht: DHT,
        port: int = 8000
    ):
        self.hypertensor = hypertensor
        self.dht = dht
        self.port = port
        self.process = None

        self.heartbeat = Gauge('heartbeat', 'Heartbeat')
        self.subnet_consensus_data = Gauge('subnet_consensus_data', 'Subnet Consensus')
        self.onchain_consensus_data = Gauge('onchain_consensus_data', 'Onchain Consensus')

    def start(self):
        self.process = Process(target=self._run)
        self.process.start()

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.join()

    def _run(self):
        start_http_server(self.port)
        print(f"[Metrics] Prometheus exporter started on port {self.port}")

        # Start separate threads for each interval-specific updater
        Thread(target=self._heartbeat_metrics, daemon=True).start()
        Thread(target=self._onchain_consensus, daemon=True).start()
        Thread(target=self._subnet_consensus, daemon=True).start()

        # Keep the process alive
        while self.running:
            time.sleep(1)

    def _heartbeat_metrics(self):
        while self.running:
            self.heartbeat.set(self.get_heartbeat_metrics())
            time.sleep(60)

    def _onchain_consensus(self):
        while self.running:
            self.inference_latency_gauge.set(self.get_onchain_consensus())
            time.sleep(100)

    def _subnet_consensus(self):
        while self.running:
            self.cache_hit_ratio.set(self.get_subnet_consensus())
            time.sleep(600)

    def get_heartbeat_metrics(self) -> int:
        start_time = time.perf_counter()
        bootstrap_peer_ids = []
        for addr in INITIAL_PEERS:
            peer_id = PeerID.from_base58(Multiaddr(addr)["p2p"])
            if peer_id not in bootstrap_peer_ids:
                bootstrap_peer_ids.append(peer_id)

        reach_infos = self.dht.run_coroutine(partial(check_reachability_parallel, bootstrap_peer_ids))
        bootstrap_states = ["online" if reach_infos[peer_id]["ok"] else "unreachable" for peer_id in bootstrap_peer_ids]

        all_servers: List[RemoteModuleInfo] = []
        hoster_module_infos = get_node_infos(self.dht, "hoster", latest=True)
        all_servers.append(hoster_module_infos)
        validator_module_infos = get_node_infos(self.dht, "validator", latest=True)
        all_servers.append(validator_module_infos)
        online_servers = [peer_id for peer_id, span in all_servers.items() if span.state == ServerState.ONLINE]

        reach_infos.update(self.dht.run_coroutine(partial(check_reachability_parallel, online_servers, fetch_info=True)))
        peers_info = {str(peer.peer_id): {"location": extract_peer_ip_info(str(peer.addrs[0])), "multiaddrs": [str(multiaddr) for multiaddr in peer.addrs]} for peer in self.dht.run_coroutine(get_peers_ips)}

        metrics = []
        for server in all_servers:
            peer_id = server.peer_id
            reachable = reach_infos[peer_id]["ok"] if peer_id in reach_infos else True
            state = server.server.state.name.lower() if reachable else "unreachable"
            server_info = server.server
            role = server_info.role
            public_name = server_info.public_name
            location = peers_info.get(str(peer_id), None)
            latitude = location['lat'] if location is not None else None
            longitude = location['lon'] if location is not None else None
            country = location['country'] if location is not None else None
            region = location['region'] if location is not None else None

            data = {
                "peer_id": peer_id,
                "state": state,
                "role": role.name,
                "public_name": public_name,
                "latitude": latitude,
                "longitude": longitude,
                "country": country,
                "region": region,
            }
            metrics.append(data)

        reachability_issues = [
            dict(peer_id=peer_id, err=info["error"]) for peer_id, info in sorted(reach_infos.items()) if not info["ok"]
        ]

        return dict(
            bootstrap_states=bootstrap_states,
            metrics=metrics,
            reachability_issues=reachability_issues,
            last_updated=datetime.datetime.now(datetime.timezone.utc),
            next_update=datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=UPDATE_PERIOD),
            update_period=UPDATE_PERIOD,
            update_duration=time.perf_counter() - start_time
        )

    def get_onchain_consensus(self):
        ...

    def get_subnet_consensus(self):
        epoch_data = self.hypertensor.get_epoch_progress()
        current_epoch = epoch_data.epoch

        # Get hosters data
        hoster_scores = self.score_hosters(current_epoch)

        # Get validators data
        validator_score = self.score_validators(current_epoch)

        # Merge scores
        merged_scores = self.get_merged_scores(hoster_scores, validator_score)

        # Filter scores
        filtered_scores = self.filter_merged_scores(merged_scores)

        return filtered_scores

    def score_hosters(self, current_epoch: int) -> Dict[str, float]:
        """
        Compute accuracy scores for each hoster based on the proximity of their output tensor
        to the mean tensor of all successful results.

        Each hoster's score is calculated as the inverse of their L2 distance to the mean tensor,
        i.e., `1 / (1 + distance)`. This ensures:
        - A score close to 1 if their output is close to the consensus.
        - A score approaching 0 if their output is very far or missing.

        If a hoster’s inference was unsuccessful or missing, they receive a score of 0.

        Args:
            results (List[HosterResult]): List of inference results from hosters, including their output tensors.

        Returns:
            Dict[str, float]: A mapping from peer ID to their computed accuracy score.
        """
        commit_key = get_hoster_commit_key(current_epoch)
        reveal_key = get_hoster_reveal_key(current_epoch)

        commit_records = self.dht.get(commit_key) or {}
        reveal_records = self.dht.get(reveal_key) or {}

        if not reveal_records:
            return

        results: List[HosterResult] = []
        for public_key, reveal_data in reveal_records.value.items():
            try:
                peer_id = extract_rsa_peer_id(public_key)

                payload = reveal_data.value
                salt = payload["salt"]
                tensor_bytes = payload["tensor"]

                # 1) Verify hoster committed the same hash
                recomputed_digest = hashlib.sha256(salt + tensor_bytes).digest()
                committed_digest = commit_records.value[public_key].value

                if committed_digest != recomputed_digest:
                    results.append(HosterResult(peer=peer_id, output=None, success=False))
                    continue

                # 2) Deserialize the tensor
                tensor = torch.load(io.BytesIO(tensor_bytes), weights_only=False)
                results.append(HosterResult(peer=peer_id, output=tensor, success=True))

            except Exception as e:
                results.append(HosterResult(peer=peer_id, output=None, success=False))

        valid_outputs = [r.output for r in results if r.success and r.output is not None]
        if not valid_outputs:
            return {r.peer: 0.0 for r in results}

        stacked = torch.stack(valid_outputs)
        mean_tensor = torch.mean(stacked, dim=0)

        hoster_scores = {}
        for r in results:
            if not r.success or r.output is None:
                hoster_scores[r.peer] = 0.0
                continue
            diff = torch.norm(r.output - mean_tensor)
            hoster_scores[r.peer] = float(1.0 / (1.0 + diff.item()))

        hoster_scores = self.normalize_scores(hoster_scores, HOSTER_SCORE_RATIO)

        return hoster_scores

    def score_validators(self, current_epoch: int) -> Dict[str, float]:
        """
        Get the validator submitted data from the DHT Record

        We ensure the validator is submitting scores to the DHT

        - We get each Record entry by each hoster node
        - We iterate to validate and score this data and store it in the Consensus class

        Note:
            Validators with no reveal (1 epoch old validators)are not submitted as the
            reveal is the next epoch from the commit.
        """

        validator_commit_key = get_validator_commit_key(current_epoch - 1)
        validator_reveal_key = get_validator_reveal_key(current_epoch)

        commit_records = self.dht.get(validator_commit_key) or {}
        reveal_records = self.dht.get(validator_reveal_key) or {}

        if not reveal_records:
            return {}

        results: Dict[str, List[ConsensusScores]] = {}

        for public_key, reveal_data in reveal_records.value.items():
            try:
                peer_id = extract_rsa_peer_id(public_key)

                payload = reveal_data.value
                salt = payload["salt"]
                scores_bytes = payload["scores_bytes"]

                # 1) Verify the commit hash
                recomputed_digest = hashlib.sha256(salt + scores_bytes).digest()
                committed_digest = commit_records.value[public_key].value

                if committed_digest != recomputed_digest:
                    print(f"[Validator] Hash mismatch from validator {peer_id}, skipping")
                    continue

                # 2) Deserialize the scores
                raw_scores = pickle.loads(scores_bytes)
                scores: List[ConsensusScores] = [
                    ConsensusScores(peer_id=score.peer_id.to_base58(), score=score.score)
                    for score in raw_scores
                ]
                results[peer_id] = scores

            except Exception as e:
                print(f"[Validator] Failed to verify or parse scores from {peer_id}: {e}")

        # Step 1: Get scores per hoster peer_id
        peer_scores = defaultdict(list)  # hoster_peer_id -> list of scores
        for round_scores in results.values():
            for score_obj in round_scores:
                peer_scores[score_obj.peer_id].append(score_obj.score)

        """
        TODO: Get blockchains consensus

        If super majority of attestation, score based on mean from
        validators submitted data.

        Otherwise, score based on commit-reveal auth
        """

        # Step 2: Compute the mean score per hoster
        peer_means = {
            peer_id: statistics.mean(scores)
            for peer_id, scores in peer_scores.items()
        }

        # Step 3: Compute squared error per validator
        validator_errors: Dict[str, float] = {}
        for validator_peer_id, round_scores in results.items():
            error_sum = 0.0
            for score_obj in round_scores:
                mean = peer_means.get(score_obj.peer_id)
                if mean is not None:
                    error_sum += (score_obj.score - mean) ** 2
            validator_errors[validator_peer_id] = error_sum

        # Step 4: Normalize errors and subtract from base score
        max_error = max(validator_errors.values(), default=1.0)

        validator_scores = {
            peer_id: max(BASE_VALIDATOR_SCORE - (error / (max_error + EPSILON)), 0.0)
            for peer_id, error in validator_errors.items()
        }

        validator_scores = self.normalize_scores(validator_scores, VALIDATOR_SCORE_RATIO)

        return validator_scores

    def normalize_scores(self, scores: Dict[str, float], target_total: float) -> Dict[str, float]:
        total = sum(scores.values())
        if total == 0:
            return {peer_id: 0.0 for peer_id in scores}
        return {
            peer_id: (score / total) * target_total
            for peer_id, score in scores.items()
        }

    def get_merged_scores(
        self,
        hoster_scores: Optional[Dict[str, float]],
        validator_scores: Optional[Dict[str, float]]
    ) -> List[ConsensusScores]:
        """
        Merge hoster and validator scores submitted by Consensus
        """
        if hoster_scores is None and validator_scores is None:
            return []

        merged_scores = hoster_scores.copy() if hoster_scores is not None else {}
        merged_scores.update(validator_scores)

        # Step 2: Convert to List[ConsensusScores], rounding or casting score to int
        consensus_score_list = [
            ConsensusScores(peer_id=peer_id, score=int(score * 1e18))
            for peer_id, score in merged_scores.items()
        ]

        return consensus_score_list

    def filter_merged_scores(self, scores: Dict[str, float]) -> List[ConsensusScores]:
        """
        Filter scores against the blockchain activated subnet nodes
        """
        return scores
