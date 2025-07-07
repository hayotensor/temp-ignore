# import dataclasses
# import time
# from typing import Dict, Iterable, List, Optional, Tuple

# from hivemind import PeerID, get_logger
# from hivemind.subnet.data_structures import ModuleUID, RemoteInfo, RemoteModuleInfo, ServerState
# from hivemind.subnet.utils.dht import compute_spans

# logger = get_logger(__name__)


# @dataclasses.dataclass
# class RemoteServerInfo:
#     """
#     A dataclass that stores general information about which servers hold any given layer;
#     - updated by RemoteSequenceManager in a background thread
#     - accessed by routing strategies in .on_update
#     :note: this class should *not* be modified by RoutingStrategy.on_update to avoid interference between strategies;
#      Any metadata specific to one routing strategy, it should be stored inside that strategy. Any information that
#      is used by most routing strategies should be moved from said strategies to this class.
#     """

#     server_infos: Tuple[RemoteModuleInfo, ...]  # note: the contents of RemoteModuleInfo can and will be updated
#     servers_by_priority: List[RemoteInfo]
#     # spans_containing_block: Tuple[List[RemoteSpanInfo], ...]
#     last_updated_time: Optional[float]

#     @classmethod
#     def make_empty(cls, block_uids: Iterable[ModuleUID]) -> "RemoteServerInfo":
#         # empty_server_infos = tuple(RemoteModuleInfo(uid, {}) for uid in block_uids)
#         empty_server_infos = tuple()
#         # empty_spans = tuple([] for _ in range(len(block_uids)))
#         return cls(empty_server_infos, [], last_updated_time=None)

#     # def __getitem__(self, ix: slice):
#     #     assert isinstance(ix, slice)
#     #     block_uids, server_infos = self.block_uids[ix], self.block_infos[ix]
#     #     servers_by_priority, spans_containing_block = self._sort_spans(server_infos)
#     #     return RemoteInfo(
#     #         block_uids, server_infos, servers_by_priority, spans_containing_block, self.last_updated_time
#     #     )

#     def __getitem__(self, ix: slice):
#         assert isinstance(ix, slice)
#         server_infos = self.server_infos[ix]
#         servers_by_priority = self._sort_spans(server_infos)
#         return RemoteServerInfo(
#             server_infos, servers_by_priority, self.last_updated_time
#         )

#     def __len__(self):
#         return len(self.block_uids)

#     # def update_(self, new_block_infos: List[RemoteModuleInfo]):
#     #     assert len(new_block_infos) == len(self.block_uids)
#     #     for block_index, (uid, info) in enumerate(zip(self.block_uids, new_block_infos)):
#     #         assert uid == info.uid, f"The DHT entry for {uid} actually points to {info.uid}"
#     #         self.block_infos[block_index].servers = info.servers

#     #     self.servers_by_priority, self.spans_containing_block = self._sort_spans(self.block_infos)
#     #     self.last_updated_time = time.perf_counter()

#     def update_(self, new_server_infos: List[RemoteModuleInfo]):
#         for block_index, (uid, info) in enumerate(zip(self.block_uids, new_server_infos)):
#             assert uid == info.uid, f"The DHT entry for {uid} actually points to {info.uid}"
#             self.server_infos[block_index].servers = info.servers

#         self.servers_by_priority = self._sort_spans(self.server_infos)
#         self.last_updated_time = time.perf_counter()

#     @staticmethod
#     def _sort_spans(server_infos: List[RemoteModuleInfo]):
#         servers_by_priority = list(compute_spans(server_infos, min_state=ServerState.ONLINE).values())
#         servers_by_priority.sort(key=lambda span: span.length, reverse=True)

#         return servers_by_priority
