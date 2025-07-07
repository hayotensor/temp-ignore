from dataclasses import dataclass, field
from typing import List, Optional
import uuid

@dataclass
class UserSession:
    session_id: str
    user_id: str
    chat_history: List[str] = field(default_factory=list)
    host_peer: Optional[str] = None  # peer ID of model host
    # past_key_values: Optional[...] = None  # Add later

    @staticmethod
    def new(user_id: str) -> "UserSession":
        return UserSession(session_id=str(uuid.uuid4()), user_id=user_id)
