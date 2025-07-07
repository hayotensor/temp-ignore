from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Optional

from mesh import get_logger
from mesh.subnet.client.inference_session_v2 import InferenceSession
from mesh.subnet.client.routing.inference_manager_v2 import RemoteManager
from mesh.utils.auth import AuthorizerBase

logger = get_logger(__name__)

# class InferenceSessionManager:
#     def __init__(self, remote_manager: RemoteManager):
#         self.remote_manager = remote_manager
#         self.sessions: Dict[str, InferenceSession] = {}

#     def get_session(self, session_id: str) -> InferenceSession | None:
#         """Return existing session or None if missing."""
#         return self.sessions.get(session_id)

#     def create_session(self, session_id: str) -> InferenceSession:
#         """Create and store a new session."""
#         session = InferenceSession(self.remote_manager)
#         self.sessions[session_id] = session
#         return session

#     async def get_or_create(self, session_id: str) -> InferenceSession:
#         async with self.lock:
#             if session_id not in self.sessions:
#                 self.sessions[session_id] = InferenceSession(self.remote_manager)
#             return self.sessions[session_id]

#     async def enter_session(self, session_id: str):
#         """Async context manager that yields an active session."""
#         session = self.get_or_create(session_id)
#         # If your session supports async context:
#         await session.__aenter__()
#         try:
#             yield session
#         finally:
#             await session.__aexit__(None, None, None)

#     async def close(self, session_id: str):
#         session = self.sessions.pop(session_id, None)
#         if session:
#             await session.close()

class InferenceSessionManager:
    """
    Stateless manager: just creates InferenceSession instances on demand.
    Does not keep track of sessions or history.
    """

    def __init__(
        self,
        remote_manager: RemoteManager,
        max_length: int,
        authorizer: Optional[AuthorizerBase] = None,
    ):
        self._remote_manager = remote_manager
        self._max_length = max_length
        self._authorizer = authorizer

    @asynccontextmanager
    async def session(self) -> AsyncIterator[InferenceSession]:
        logger.debug("session yielding")
        session = InferenceSession(self._remote_manager, self._max_length, self._authorizer)
        try:
            yield session
        finally:
            await session.close()
