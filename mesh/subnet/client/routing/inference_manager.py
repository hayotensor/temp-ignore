# from __future__ import annotations

# import asyncio
# import dataclasses
# import itertools
# import logging
# import os
# import random
# import threading
# import time
# import warnings
# from typing import Any, Dict, List, Optional, Sequence, Set, Union
# from weakref import WeakMethod

# import numpy as np

# from hivemind import DHT, P2P, MSGPackSerializer, PeerID
# from hivemind.dht.node import Blacklist
# from hivemind.p2p.servicer import ServicerBase
# from hivemind.proto import crypto_pb2, runtime_pb2
# from hivemind.subnet.client.config import ClientConfig
# from hivemind.subnet.client.routing.remote_server_info import RemoteServerInfo
# from hivemind.subnet.data_structures import ModuleUID, RemoteInfo, ServerState
# from hivemind.subnet.utils.dht import get_node_infos
# from hivemind.subnet.utils.ping import PingAggregator
# from hivemind.utils.logging import get_logger
# from hivemind.utils.remote_worker import RemoteWorker


# logger = get_logger(__name__)



# @dataclasses.dataclass
# class SequenceManagerState:
#     p2p: P2P = None
#     remote_servers_info: Optional[RemoteServerInfo] = None
#     rpc_info: Optional[dict] = None
#     banned_peers: Optional[Blacklist] = None


# class RemoteManager:
#     """
#     Sequence manager is a thread that keeps track of remote servers that can run inference.
#     TL;DR it tells you, which peers you should ask to get inference. It is used in RemoteSequential.
#     When created, RemoteManager looks up which servers serve necessary layers by reading from DHT.
#     Using this information, sequence manager can form sequences of servers that collectively have the full sequence.
#     To form such a sequence, call .make_sequence with the appropriate optimization policy (see make_sequence docstr).

#     :note: RemoteManager takes up some CPU and network I/O to operate in background. It is recommended to avoid
#       running redundant sequence managers for the same set of layers.
#     """

#     def __init__(
#         self,
#         config: ClientConfig,
#         *,
#         dht: Optional[DHT] = None,
#         state: Optional[SequenceManagerState] = None,
#     ):
#         assert config.initial_peers or dht is not None, "Please specify `config.initial_peers` or `dht`"
#         assert config.dht_prefix, "Could not find dht_prefix in config, please create model with dht_prefix=..."

#         self.config = config
#         if state is None:
#             state = SequenceManagerState()
#         self.state = state

#         if dht is None:
#             dht = DHT(
#                 initial_peers=config.initial_peers,
#                 client_mode=True,
#                 num_workers=32,
#                 startup_timeout=config.daemon_startup_timeout,
#                 start=True,
#             )
#         assert isinstance(dht, DHT) and dht.is_alive(), "`dht` must be a running hivemind.DHT instance"
#         self.dht = dht

#         if state.p2p is None:
#             state.p2p = RemoteWorker.run_coroutine(dht.replicate_p2p())

#         self.lock_changes = threading.Lock()
#         self._thread = _SequenceManagerUpdateThread(config.update_period, WeakMethod(self._update))
#         self._thread_start_lock = threading.Lock()

#         self.allowed_servers = self._peer_ids_to_set(config.allowed_servers)
#         self.blocked_servers = self._peer_ids_to_set(config.blocked_servers)

#         self.ping_aggregator = PingAggregator(dht)

#         if state.banned_peers is None:
#             state.banned_peers = Blacklist(base_time=config.ban_timeout, backoff_rate=2.0)
#         if state.remote_servers_info is None:
#             state.remote_servers_info = RemoteServerInfo.make_empty(0)

#         if state.remote_servers_info.last_updated_time is not None:
#             self._thread.ready.set()  # no need to await the first dht fetch

#     @staticmethod
#     def _peer_ids_to_set(peer_ids: Optional[Sequence[Union[PeerID, str]]]) -> Optional[Set[PeerID]]:
#         if peer_ids is None:
#             return None

#         result = set()
#         for peer_id in peer_ids:
#             if isinstance(peer_id, PeerID):
#                 result.add(peer_id)
#             elif isinstance(peer_id, str):
#                 result.add(PeerID.from_base58(peer_id))
#             else:
#                 raise TypeError(
#                     f"`allowed_servers` and `blocked_servers` have to contain only PeerIDs or strings, but got {type(peer_id)}"
#                 )
#         return result

#     def make_sequence(
#         self,
#         *,
#         cache_tokens_needed: Optional[int] = None,
#     ) -> RemoteInfo:
#         """
#         Form a sequence of remote servers that collectively serve all consecutive layers

#         :param mode: one of ["max_throughput", "min_latency"]
#         """
#         with self._thread_start_lock:
#             if not self.is_alive():
#                 self._thread.start()
#         print("[RemoteManager] not self.ready.is_set()", not self.ready.is_set())
#         if not self.ready.is_set():
#             print("[RemoteManager] running update")
#             self.update(wait=True)  # this will await an existing update or trigger a new one (if not updating)

#         print("[RemoteManager] before _get_peer_with_max_throughput")
#         server = self._get_peer_with_max_throughput()
#         print("[RemoteManager] make_sequence server", server)

#         return server

#     @staticmethod
#     def _rtt_to_delay(
#         rtt: float,
#         *,
#         default_delay: float = 0.15,  # If network delay unknown
#         max_delay: float = 5,  # If unreachable, we don't want to discard the edge completely
#     ) -> float:
#         if rtt is None:
#             return default_delay
#         return min(rtt / 2, max_delay)

#     def _get_peer_with_max_throughput(self) -> RemoteInfo:
#         print("[RemoteManager] _get_peer_with_max_throughput")
#         client_server_rtts = self.ping_aggregator.to_dict()
#         print("[RemoteManager] client_server_rtts", client_server_rtts)

#         # TODO: Add logic for getting peer closest and available
#         # return client_server_rtts
#         return next(iter(client_server_rtts))


#     def __getitem__(self, ix: Union[int, slice]) -> RemoteManager:
#         """Get a RemoteManager for a sub-sequence of blocks"""
#         assert isinstance(ix, (int, slice))
#         if not isinstance(ix, slice):
#             ix = slice(int(ix), int(ix) + 1, 1)
#         return type(self)(self.config, dht=self.dht, state=self.state[ix])

#     def update(self, *, wait: bool):
#         """Run an asynchronous update in background as soon as possible"""
#         print("InferenceManager update")
#         logger.debug("InferenceManager update")
#         self.ready.clear()
#         self._thread.trigger.set()
#         if wait:
#             self.ready.wait()

#     def _update(self):
#         """
#         Perform an immediate and synchronous refresh, may take time
#         """

#         hoster_infos = get_node_infos(
#           self.dht,
#           uid="hoster",
#           latest=True
#         )
#         print("[RemoteManager] hoster_infos: ", hoster_infos)

#         for remote_info in hoster_infos:  # this is your list
#             peer_id = remote_info.peer_id
#             print("[RemoteManager] peer_id: ", peer_id)
#             server = remote_info.server
#             print("[RemoteManager] server: ", server)

#         # hoster_infos_dict = {hoster_info.peer_id: hoster_info for hoster_info in hoster_infos}

#         # for peer_id, server_info in hoster_infos_dict.items():
#         #     print("[RemoteManager] peer_id: ", peer_id)
#         #     print("[RemoteManager] server_info: ", server_info)

#         # # for hosters in hoster_infos:
#         # # Apply allow and block lists
#         # hosters = {
#         #     peer_id: server_info
#         #     for peer_id, server_info in hoster_infos_dict.items()
#         #     if (self.allowed_servers is None or peer_id in self.allowed_servers)
#         #     and (self.blocked_servers is None or peer_id not in self.blocked_servers)
#         # }
#         # print("[RemoteManager] hosters: ", hosters)

#         # # hoster.servers = {
#         # #     peer_id: server_info
#         # #     for peer_id, server_info in hoster.servers.items()
#         # #     if (self.allowed_servers is None or peer_id in self.allowed_servers)
#         # #     and (self.blocked_servers is None or peer_id not in self.blocked_servers)
#         # # }

#         # # Remove temporarily banned peers, unless there are no peers left
#         # valid_servers = {
#         #     peer_id: server_info
#         #     for peer_id, server_info in hosters
#         #     if peer_id not in self.state.banned_peers
#         # }
#         # print("[RemoteManager] valid_servers: ", valid_servers)

#         # if len(valid_servers) < len(hosters):
#         #     if valid_servers:
#         #         logger.debug(
#         #             f"Kept {len(valid_servers)} out of {len(hosters)} servers"
#         #         )
#         #         hosters = valid_servers
#         #     else:
#         #         # If we blacklisted all servers, the error may actually be client-caused
#         #         logger.debug("All servers are blacklisted, ignoring blacklist")

#         with self.lock_changes:
#             self.state.remote_servers_info.update_(hoster_infos)

#             # first_servers = [span.peer_id for span in self.state.sequence_info.spans_containing_block[0]]
#             # middle_servers = [
#             #     span.peer_id for spans in self.state.sequence_info.spans_containing_block[1:-1] for span in spans
#             # ]
#             # last_servers = [span.peer_id for span in self.state.sequence_info.spans_containing_block[-1]]
#             all_servers = [server.peer_id for server in hoster_infos]
#             print("[RemoteManager] all_servers: ", all_servers)

#         # pinged_servers = set(sample_up_to(first_servers, self.config.max_pinged))
#         # pinged_servers = set(sample_up_to(middle_servers, self.config.max_pinged))
#         # pinged_servers |= set(sample_up_to(last_servers, self.config.max_pinged))
#         # self.ping_aggregator.ping(list(pinged_servers), wait_timeout=self.config.ping_timeout)

#         print("[RemoteManager] ping")
#         self.ping_aggregator.ping(list(all_servers), wait_timeout=self.config.ping_timeout)
#         client_server_rtts = self.ping_aggregator.to_dict()
#         print("[RemoteManager] _update client_server_rtts", client_server_rtts)

#         print("[RemoteManager] ready.set()")

#         self.ready.set()

#     def on_request_failure(self, peer_id: Optional[PeerID]):
#         """remove a given peer from the routing table. If the routing is no longer possible, trigger an update"""
#         if peer_id is not None:
#             logger.debug(f"Peer {peer_id} did not respond, banning it temporarily")
#             self.state.banned_peers.register_failure(peer_id)
#         with self.lock_changes:
#             should_update = False
#             # for info in self.state.sequence_info.block_infos:
#             #     info.servers.pop(peer_id, None)
#             #     if not info.servers:
#             #         should_update = True
#             if should_update:
#                 self.ready.clear()
#                 self.update(wait=False)

#     def on_request_success(self, peer_id: PeerID):
#         """if peer has a failure streak, clear that streak"""
#         self.state.banned_peers.register_success(peer_id)

#     @property
#     def is_alive(self):
#         return self._thread.is_alive

#     @property
#     def ready(self) -> threading.Event:
#         return self._thread.ready

#     @property
#     def rpc_info(self):
#         """Return the rpc_info queried from one of the servers that hold the first block"""
#         if self.state.rpc_info is not None:
#             return self.state.rpc_info

#         with self._thread_start_lock:
#             if not self.is_alive():
#                 self._thread.start()

#         for attempt_no in itertools.count():
#             peer_id = None
#             try:
#                 if not self.ready.is_set():
#                     self.update(wait=True)

#                 # active_servers = [
#                 #     peer_id
#                 #     for peer_id, server in self.state.sequence_info.block_infos[0].servers.items()
#                 #     if server.state == ServerState.ONLINE
#                 # ]
#                 active_servers = [
#                     peer_id
#                     for peer_id, server in self.state.remote_servers_info.server_infos
#                     if server.state == ServerState.ONLINE
#                 ]
#                 print("rpc_info active_servers", active_servers)

#                 if not active_servers:
#                     raise MissingServersError()
#                 peer_id = random.choice(active_servers)
#                 print("rpc_info peer_id", peer_id)

#                 stub = ServicerBase.get_sub(self.state.p2p, peer_id)
#                 print("rpc_info stub", stub)

#                 # stub = TransformerConnectionHandler.get_stub(self.state.p2p, peer_id)
#                 # outputs = RemoteWorker.run_coroutine(
#                 #     stub.rpc_info(runtime_pb2.ExpertUID(uid=self.block_uids[0]), timeout=self.config.request_timeout)
#                 # )
#                 # self.state.rpc_info = MSGPackSerializer.loads(outputs.serialized_info)
#                 self.on_request_success(peer_id)
#                 break
#             except Exception as e:
#                 self.on_request_failure(peer_id)
#                 if attempt_no + 1 == self.config.max_retries:
#                     raise
#                 delay = self.get_retry_delay(attempt_no)
#                 logger.warning(
#                     f"Caught exception when gathering information from peer {peer_id} "
#                     f"(retry in {delay:.0f} sec): {repr(e)}"
#                 )
#                 maybe_log_traceback(e)
#                 time.sleep(delay)

#         return self.state.rpc_info

#     # @property
#     # def rpc_info(self):
#     #     """Return the rpc_info queried from one of the servers that hold the first block"""
#     #     if self.state.rpc_info is not None:
#     #         return self.state.rpc_info

#     #     with self._thread_start_lock:
#     #         if not self.is_alive():
#     #             self._thread.start()

#     #     for attempt_no in itertools.count():
#     #         peer_id = None
#     #         try:
#     #             if not self.ready.is_set():
#     #                 self.update(wait=True)

#     #             active_servers = [
#     #                 peer_id
#     #                 for peer_id, server in self.state.sequence_info.block_infos[0].servers.items()
#     #                 if server.state == ServerState.ONLINE
#     #             ]
#     #             if not active_servers:
#     #                 raise MissingServersError()
#     #             peer_id = random.choice(active_servers)

#     #             stub = ServicerBase.get_sub(self.state.p2p, peer_id)
#     #             # stub = TransformerConnectionHandler.get_stub(self.state.p2p, peer_id)
#     #             outputs = RemoteWorker.run_coroutine(
#     #                 stub.rpc_info(runtime_pb2.ExpertUID(uid=self.block_uids[0]), timeout=self.config.request_timeout)
#     #             )
#     #             self.state.rpc_info = MSGPackSerializer.loads(outputs.serialized_info)
#     #             self.on_request_success(peer_id)
#     #             break
#     #         except Exception as e:
#     #             self.on_request_failure(peer_id)
#     #             if attempt_no + 1 == self.config.max_retries:
#     #                 raise
#     #             delay = self.get_retry_delay(attempt_no)
#     #             logger.warning(
#     #                 f"Caught exception when gathering information from peer {peer_id} "
#     #                 f"(retry in {delay:.0f} sec): {repr(e)}"
#     #             )
#     #             maybe_log_traceback(e)
#     #             time.sleep(delay)

#     #     return self.state.rpc_info

#     def get_retry_delay(self, attempt_no: int) -> float:
#         if attempt_no == 0:
#             return 0
#         return min(self.config.min_backoff * 2 ** (attempt_no - 1), self.config.max_backoff)

#     def get_request_metadata(
#         self, protocol: str, args_structure: Any = None, *args, **kwargs
#     ) -> Optional[Dict[str, Any]]:
#         """
#         :param protocol: one of "rpc_forward", "rpc_backward" or "rpc_inference"
#         :param args_structure: the structure of flattened tensors from pack_args_kwargs in subnet.utils.packaging
#         :param args: request-specific inputs, typically block uids and input tensors
#         :param kwargs: additional request context, such as remote peer ID
#         :returns: msgpack-serialized metadata dict that will be passed alongside a given request
#         """
#         return dict(
#             active_adapter=self.config.active_adapter,
#             args_structure=args_structure,
#         )

#     def shutdown(self):
#         self._thread.shutdown()


# class _SequenceManagerUpdateThread(threading.Thread):
#     def __init__(self, update_period: float, ref_update_manager: WeakMethod):
#         super().__init__(daemon=True)
#         self.ref_update_manager = ref_update_manager
#         self.ready = threading.Event()
#         self.trigger = threading.Event()
#         self.update_period = update_period
#         self.should_shutdown = False

#     def run(self) -> None:
#         print("_SequenceManagerUpdateThread run")
#         while not self.should_shutdown:
#             print("_SequenceManagerUpdateThread b4 ref_update_manager")
#             update_manager = self.ref_update_manager()
#             print("_SequenceManagerUpdateThread update_manager", update_manager)
#             if update_manager is None:
#                 logger.debug(f"{self.__class__.__name__} exited because the sequence manager no longer exists")
#                 break

#             try:
#                 self.trigger.clear()
#                 update_manager()
#             except Exception as e:
#                 logger.exception(e)
#             finally:
#                 del update_manager

#             self.trigger.wait(self.update_period)

#         logger.debug(f"{self.__class__.__name__} thread exited")

#     def shutdown(self, timeout: Optional[float] = None):
#         self.should_shutdown = True
#         self.trigger.set()
#         if self.is_alive():
#             self.join(timeout)

#     def __del__(self):
#         self.shutdown()


# def maybe_log_traceback(exc: Exception):
#     traceback_level = logging.DEBUG if str(exc) or isinstance(exc, asyncio.TimeoutError) else logging.WARNING
#     logger.log(traceback_level, "See detailed traceback below:", exc_info=True)


# class MissingServersError(RuntimeError):
#     def __init__(self):
#         super().__init__(
#             "No servers hosting the model are online. "
#             "You can check the public swarm's state at https://dash.hypertensor.org "
#             "If there are not enough servers, please connect your GPU: "
#             "https://github.com/hypertensor-blockchain/subnet-llm-template "
#         )
