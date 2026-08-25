"""Per-request context.

The API key and user travel down from the entry point as an explicit argument.
Nothing below the CLI reads os.environ -- that is what makes this deployable
later without unpicking module-level state.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

from jfl_core.db.tables import LOCAL_USER_ID


@dataclass(frozen=True, slots=True)
class RequestContext:
    user_id: uuid.UUID
    anthropic_api_key: str | None
    database_url: str
    embedding_device: str = "cuda"
    trace_id: uuid.UUID = field(default_factory=uuid.uuid4)

    @classmethod
    def from_env(cls) -> RequestContext:
        """The ONLY place environment is read. Call this at the CLI boundary."""
        return cls(
            user_id=uuid.UUID(os.environ["JFL_USER_ID"])
            if "JFL_USER_ID" in os.environ
            else LOCAL_USER_ID,
            # None is meaningful: it tells the client to fall back to an
            # `ant auth login` profile rather than failing.
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            database_url=os.environ["JFL_DATABASE_URL"],
            embedding_device=os.environ.get("JFL_EMBEDDING_DEVICE", "cuda"),
        )
