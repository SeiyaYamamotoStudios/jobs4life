"""Per-request context.

The API key and user travel down from the entry point as an explicit argument.
Nothing below the CLI reads os.environ -- that is what makes this deployable
later without unpicking module-level state.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RequestContext:
    user_id: str
    anthropic_api_key: str | None
    database_url: str
    embedding_device: str = "cuda"
    trace_id: uuid.UUID = field(default_factory=uuid.uuid4)

    @classmethod
    def from_env(cls) -> RequestContext:
        """The ONLY place environment is read. Call this at the CLI boundary."""
        return cls(
            user_id=os.environ.get("JFL_USER_ID", "local"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            database_url=os.environ["JFL_DATABASE_URL"],
            embedding_device=os.environ.get("JFL_EMBEDDING_DEVICE", "cuda"),
        )
