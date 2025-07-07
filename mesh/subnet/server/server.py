from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Dict, List, Optional, Sequence, Union

import torch
from transformers import AutoConfig, AutoTokenizer

import mesh
from mesh import DHT, MAX_DHT_TIME_DISCREPANCY_SECONDS, get_dht_time
from mesh.proto.runtime_pb2 import CompressionType  # for de/serializing protos Tensors
from mesh.subnet.constants import PUBLIC_INITIAL_PEERS
from mesh.subnet.data_structures import ModelInfo, QuantType, ServerClass, ServerInfo, ServerState
from mesh.subnet.reachability import ReachabilityProtocol, check_direct_reachability
from mesh.subnet.protocols.inference_protocol import InferenceProtocol
from mesh.subnet.utils.dht import declare_node, get_node_infos
from mesh.subnet.utils.ping import PingAggregator
from mesh.subnet.utils.random import sample_up_to
from mesh.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_NUM_WORKERS = 8

class Server:
    """
    Runs Module in thread
        - Deploys or joins DHT, starts node
        - Deploys application logic
        - Peer/node heartbeat
    """

    def __init__(
        self,
        *,
        initial_peers: List[str],
        dht_prefix: Optional[str],
        converted_model_name_or_path: str,
        public_name: Optional[str] = None,
        throughput: Union[float, str],
        num_handlers: int = 8,
        max_alloc_timeout: float = 600,
        torch_dtype: str = "auto",
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        compression=CompressionType.NONE,
        update_period: float = 60,
        expiration: Optional[float] = None,
        request_timeout: float = 3 * 60,
        session_timeout: float = 30 * 60,
        sender_threads: int = 1,
        token: Optional[Union[str, bool]] = None,
        quant_type: Optional[QuantType] = None,
        skip_reachability_check: bool = False,
        reachable_via_relay: Optional[bool] = None,
        use_relay: bool = True,
        use_auto_relay: bool = True,
        adapters: Sequence[str] = (),
        **kwargs,
    ):
        """
        Create a server
        """

        self.converted_model_name_or_path = converted_model_name_or_path

        self.num_handlers = num_handlers
        self.compression = compression
        self.update_period = update_period
        self.sender_threads = sender_threads
        self.revision, self.token = revision, token

        self.block_config = AutoConfig.from_pretrained(
            converted_model_name_or_path,
            use_auth_token=token,
            revision=revision,
        )

        self.dht_prefix = dht_prefix

        if expiration is None:
            expiration = max(2 * update_period, MAX_DHT_TIME_DISCREPANCY_SECONDS)
        self.expiration = expiration

        self.request_timeout = request_timeout
        self.session_timeout = session_timeout

        self.initial_peers = initial_peers
        self.announce_maddrs = kwargs.get('announce_maddrs')  # Returns None if 'my_key' not present
        print("announce_maddrs", self.announce_maddrs)

        self.port = int(self.announce_maddrs[0].rsplit("/", 1)[-1])
        print("port_int", self.port)

        # Connect to DHT
        if reachable_via_relay is None:
            is_reachable = check_direct_reachability(initial_peers=initial_peers, use_relay=False, **kwargs)
            reachable_via_relay = is_reachable is False  # if can't check reachability (returns None), run a full peer
            logger.info(f"This server is accessible {'via relays' if reachable_via_relay else 'directly'}")
        self.dht = DHT(
            initial_peers=initial_peers,
            start=True,
            num_workers=DEFAULT_NUM_WORKERS,
            use_relay=use_relay,
            use_auto_relay=use_auto_relay,
            client_mode=reachable_via_relay,
            **kwargs,
        )
        self.reachability_protocol = ReachabilityProtocol.attach_to_dht(self.dht) if not reachable_via_relay else None

        visible_maddrs_str = [str(a) for a in self.dht.get_visible_maddrs()]
        if initial_peers == PUBLIC_INITIAL_PEERS:
            logger.info("Connecting to the public swarm")
        else:
            logger.info(f"Connecting to a private swarm, initial peers: {initial_peers}")
        logger.info(f"Running a server on {visible_maddrs_str}")
        self.should_validate_reachability = not skip_reachability_check and initial_peers == PUBLIC_INITIAL_PEERS

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device(device.type, index=0)
        self.device = device

        self.torch_dtype = torch_dtype

        if quant_type is None:
            quant_type = QuantType.NF4 if device.type == "cuda" else QuantType.NONE
        self.quant_type = quant_type

        self.max_alloc_timeout = max_alloc_timeout

        assert isinstance(throughput, float) or throughput in ["auto", "eval", "dry_run"]

        throughput_info = {"throughput": 1.0}
        self.server_info = ServerInfo(
            state=ServerState.JOINING,
            role=ServerClass.HOSTER,
            public_name=public_name,
            version="1.0.0",
            adapters=tuple(adapters),
            torch_dtype=str(torch_dtype).replace("torch.", ""),
            quant_type=quant_type.name.lower(),
            using_relay=reachable_via_relay,
            **throughput_info,
        )
        self.model_info = ModelInfo(num_blocks=self.block_config.num_hidden_layers)
        if not os.path.isdir(converted_model_name_or_path):
            self.model_info.repository = "https://huggingface.co/" + converted_model_name_or_path

        self.module_container = None
        self.stop = threading.Event()

    def run(self):
        """
        Deploy protocols (Module) and node heartbeat (ModuleAnnouncerThread)

        Example:

            self.module_container = Module(
                dht=self.dht,
                converted_model_name_or_path=self.converted_model_name_or_path,
                dht_prefix=self.dht_prefix,
                device=self.device,
                server_info=self.server_info,
                model_info=self.model_info,
                update_period=self.update_period,
                expiration=self.expiration,
                start=True
            )
        """
        self.module_container = Module(
            dht=self.dht,
            converted_model_name_or_path=self.converted_model_name_or_path,
            dht_prefix=self.dht_prefix,
            device=self.device,
            server_info=self.server_info,
            model_info=self.model_info,
            update_period=self.update_period,
            expiration=self.expiration,
            start=True
        )

        """
        Keep server running forever
        """
        self.stop.wait()

    def shutdown(self, timeout: Optional[float] = 5):
        logger.info("Shutting down Server, wait to shutdown properly")
        self.stop.set()

        """
        if self.module_container is not None and self.module_container.is_alive():
            self.module_container.join(timeout)
        """

        if self.reachability_protocol is not None:
            self.reachability_protocol.shutdown()
        self.dht.shutdown()
        self.dht.join()

class Module(threading.Thread):
    def __init__(
        self,
        dht: DHT,
        converted_model_name_or_path: str,
        dht_prefix: Optional[str],
        server_info: ServerInfo,
        model_info: ModelInfo,
        update_period: float,
        expiration: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        start: bool = True,
    ):
        super().__init__()
        self.dht = dht
        self.converted_model_name_or_path = converted_model_name_or_path
        self.device = device

        server_info.state = ServerState.JOINING
        self.dht_announcer = ModuleAnnouncerThread(
            dht,
            server_info,
            model_info,
            dht_prefix=dht_prefix,
            update_period=update_period,
            expiration=expiration,
            daemon=True,
        )
        self.dht_announcer.start()
        logger.info("Announced to the DHT that we are joining")

        if start:
            self.run()

    def run(self):
        bootstrap = mesh.PeerID.from_base58("QmNV5G3hq2UmAck2htEgsqrmPFBff5goFZAdmKDcZLBZLX")
        if self.dht.peer_id.__eq__(bootstrap):
            inference_protocol = InferenceProtocol(
                dht=self.dht,
                converted_model_name_or_path=self.converted_model_name_or_path,
                device=self.device,
                client_mode=False,
                start=True
            )
        else:
            inference_protocol = InferenceProtocol(
                dht=self.dht,
                converted_model_name_or_path=self.converted_model_name_or_path,
                device=self.device,
                client_mode=True,
                start=True
            )

        async def run_stream_test():
            tokenizer = AutoTokenizer.from_pretrained(self.converted_model_name_or_path)
            # inputs = tokenizer("A cat sat", return_tensors="pt").to("cpu")
            inputs = tokenizer("A cat sat", return_tensors="pt").input_ids
            async for tensor in inference_protocol.call_inference_stream(bootstrap, "A cat sat", inputs):
                print(
                    "Got token tensor:", tensor,
                    "-> token id:", tensor.item(),
                    "-> char:", tokenizer.decode(tensor[0], skip_special_tokens=True)
                )

        """Testing `call_inference`"""
        if self.dht.peer_id.__eq__(bootstrap) is False:
            print("Test call_inference")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            """Call stream inference"""
            loop.run_until_complete(run_stream_test())
            loop.close()

    def shutdown(self):
        """
        Gracefully terminate the container, process-safe.
        Please note that terminating container otherwise (e.g. by killing processes) may result in zombie processes.
        If you did already cause a zombie outbreak, your only option is to kill them with -9 (SIGKILL).
        """
        self.dht_announcer.announce(ServerState.OFFLINE)
        logger.info("Announced to the DHT that we are exiting")

        self.join()

        logger.info("Module shut down successfully")

class ModuleAnnouncerThread(threading.Thread):
    """Periodically announces server is live before expiration of storage, visible to all DHT peers"""

    def __init__(
        self,
        dht: DHT,
        server_info: ServerInfo,
        model_info: ModelInfo,
        dht_prefix: Optional[str],
        *,
        update_period: float,
        expiration: float,
        max_pinged: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dht = dht
        self.server_info = server_info
        self.model_info = model_info

        self.update_period = update_period
        self.expiration = expiration
        self.trigger = threading.Event()

        self.dht_prefix = dht_prefix
        self.max_pinged = max_pinged
        self.ping_aggregator = PingAggregator(self.dht)

    def run(self) -> None:
        """
        Start heartbeat

        - Tell the network you're still hear
        - Ping other nodes
        """
        while True:
            logger.info("ModuleAnnouncerThread iteration")
            start_time = time.perf_counter()

            if self.server_info.state != ServerState.OFFLINE:
                """
                self._ping_next_servers()
                self.server_info.next_pings = {
                    peer_id.to_base58(): rtt for peer_id, rtt in self.ping_aggregator.to_dict().items()
                }
                """
            else:
                self.server_info.next_pings = None  # No need to ping if we're disconnecting

            declare_node(
                dht=self.dht,
                key="hoster",
                server_info=self.server_info,
                expiration_time=get_dht_time() + self.expiration,
            )

            module_infos = get_node_infos(
                self.dht,
                uid="hoster",
                latest=True
            )
            print("module_infos", module_infos)

            if self.server_info.state == ServerState.OFFLINE:
                break

            """
            If you want to host multiple models in one DHT or run a bootstrap node that acts as an entry 
            point to multiple subnets, you can do so in the DHTStorage mechanism.

            Without a clear understanding of how DHTs or DHTStorage, we suggest isolating subnets and not using this.

            if not self.dht_prefix.startswith("_"):
                self.dht.store(
                    key="_team_name_here.subnets",
                    subkey=self.dht_prefix,
                    value=self.model_info.to_dict(),
                    expiration_time=get_dht_time() + self.expiration,
                )
            """

            delay = self.update_period - (time.perf_counter() - start_time)
            if delay < 0:
                logger.warning(
                    f"Declaring node to DHT takes more than --update_period, consider increasing it (currently {self.update_period})"
                )
            self.trigger.wait(max(delay, 0))
            self.trigger.clear()

    def announce(self, state: ServerState) -> None:
        self.server_info.state = state
        self.trigger.set()
        if state == ServerState.OFFLINE:
            self.join()

    def _ping_next_servers(self) -> Dict[mesh.PeerID, float]:
        # module_infos = get_module_infos(self.dht, self.next_uids, latest=True)
        module_infos = get_node_infos(
            self.dht,
            uid="hoster",
            latest=True
        )
        middle_servers = {peer_id for info in module_infos[:-1] for peer_id in info.servers}
        pinged_servers = set(sample_up_to(middle_servers, self.max_pinged))
        pinged_servers.discard(self.dht.peer_id)
        # Sample servers hosting the block after the last one (most likely continuations) separately
        pinged_servers |= set(sample_up_to(module_infos[-1].servers, self.max_pinged))
        self.ping_aggregator.ping(list(pinged_servers))
