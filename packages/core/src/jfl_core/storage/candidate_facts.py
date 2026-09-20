"""PLACEHOLDER -- CV-derived candidate facts (NEXT.md task 1, CV onboarding).

**This module is a stand-in and is expected to be replaced wholesale.** The
real `candidate_facts` table, its extraction pass and its per-role confirmation
screen are being built alongside this slice; scoring (PLAN.md B4) needs only to
read the **unconfirmed** facts so it can name them as levers -- "your CVs claim
X; confirm it and this moves from 5 to 7" -- and that read is the only surface
it depends on:

    PostgresCandidateFactRepository(conn, user_id).list_facts(state="proposed")

returning objects with `fact_text`, `role_label` and `source_line`. Nothing in
the scoring path imports the model class by name or touches the table, so
swapping the real module in over this one changes nothing above it. Until then
this returns nothing, and a score simply has no levers -- which is the correct
behaviour for a user who has uploaded no CVs anyway.

The standing rule this exists to serve: a proposed fact is **not evidence**.
Only confirmed corpus facts ground anything (CLAUDE.md, 2026-09-18), so these
never reach a grounding query, never reach the claim gate, and never raise a
score by themselves -- they can only be named as something the user could
confirm.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from jfl_core.storage.tenancy import TenantScopedRepository

# confirmed: the user said it is true as written, and it is a corpus span.
# proposed: a model read it off one of their CVs and nobody has confirmed it.
# rejected: the user said it is not true, or not theirs.
CandidateFactState = Literal["confirmed", "proposed", "rejected"]


class CandidateFact(BaseModel):
    """One fact a model read off a CV, with the line it came from.

    `source_line` is the CV's own words, kept so the confirmation screen can
    show the user what is being claimed on their behalf rather than a
    paraphrase of it.
    """

    fact_text: str
    role_label: str = ""
    source_line: str = ""
    state: CandidateFactState = "proposed"


class PostgresCandidateFactRepository(TenantScopedRepository):
    """This user's CV-derived candidate facts, and no one else's.

    Placeholder: see the module docstring. `list_facts` returns nothing until
    the real implementation lands.
    """

    def list_facts(self, state: CandidateFactState = "proposed") -> list[CandidateFact]:
        return []
