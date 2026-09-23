"""Per-request context.

The API key and user travel down from the entry point as an explicit argument.
Nothing below the CLI reads os.environ -- that is what makes this deployable
later without unpicking module-level state.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Literal

from jfl_core.db.tables import LOCAL_USER_ID

# The model defaults, defined once. Core is the one package every other package may
# import, so `jfl_gate.pricing`, the worker's settings and the CLI read these rather
# than each repeating the literal.
#
# The product model: every generation call (extraction, coverage, CV facts, scoring,
# drafting, application answers) runs on this unless `$JFL_MODEL` / `--model` names
# another. Opus 5.5 at $4 / $20 per MTok against Opus 5's $5 / $25. Model choice is a
# product option (CLAUDE.md, 2026-09-05), so Opus 5 stays selectable.
PRODUCT_MODEL = "claude-opus-5-5"

# The claim gate's model, configured independently (`$JFL_GATE_MODEL`, `--gate-model`)
# and deliberately NOT following the product model. The published over-claim 0.7%
# (1/140) and over-flag 2.9% (2/69) were measured on Opus 5; Opus 5.5 widens the
# `reasoning_extraction` safety classifier that once refused every gate call
# (CLAUDE.md, 2026-09-02); so the gate moves only after the paired eval
# (`packages/evals/scripts/compare_eval_runs.py`) shows the two models agree on the
# 210-item tier-1 set -- the same evidence that put the gate on Opus 5 in the first
# place. Until then, switching it would quietly change what the published number
# describes.
GATE_MODEL = "claude-opus-5"

# Every Opus call pins its effort explicitly. Opus 5 defaulted to `high`, which is
# what every measured number and every shipped prompt ran at; Opus 5.5 defaults to
# `medium`, so leaving it unset would make the switch change the depth of reasoning
# as well as the price, silently. Pinned so the switch changes the price only.
# (Haiku 4.5 rejects `effort`, so the deliberately-cheap Haiku calls do not send it.)
MODEL_EFFORT: Literal["low", "medium", "high", "xhigh", "max"] = "high"


@dataclass(frozen=True, slots=True)
class RequestContext:
    user_id: uuid.UUID
    anthropic_api_key: str | None
    database_url: str
    embedding_device: str = "cuda"
    # The product model; see PRODUCT_MODEL above.
    model: str = PRODUCT_MODEL
    # The claim gate's model, independent of `model`; see GATE_MODEL above.
    gate_model: str = GATE_MODEL
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
            model=os.environ.get("JFL_MODEL") or PRODUCT_MODEL,
            gate_model=os.environ.get("JFL_GATE_MODEL") or GATE_MODEL,
        )
