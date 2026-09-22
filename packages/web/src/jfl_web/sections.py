"""Collapsible sections: one component, one set of rules, applied everywhere.

The contract is written out in `docs/ui-sections.md`. This module is the half of
it that is code, and it is deliberately pure -- every function here takes what a
route already loaded and returns a `Section`, so the rules are unit-testable
without a database, a browser or a model.

Three rules decide whether a section starts open, in this order:

  1. **Forced.** Something is pending or needs attention -- a running
     extraction, an unanswered question, a failed run, a prerequisite not met.
     A forced section is open whatever anyone has chosen before, because the
     choice was made about a different situation and this one is transient.
  2. **The user's own choice**, if they have ever toggled this section.
  3. **The agreed default** otherwise (`docs/ui-sections.md` lists them).

The change marker is derived from timestamps the data already carries -- a
draft's `created_at`, a score's `updated_at`, an extraction's `extracted_at` --
measured against `last_opened_at`, which is written when the user toggles the
section and at no other time.

**No watermark means no marker.** A section nobody has ever opened or closed has
no `last_opened_at`, and the honest answer to "what is new since you last
looked" is then "we have never seen you look". Inventing a baseline would
announce a year of old drafts as news on someone's first visit -- the same
failure a newly watched job board's first check avoids by being a baseline
rather than news (CLAUDE.md, 2026-09-10).

**A marker never appears on an open section.** The marker exists to say what
changed inside something folded away; on an open section the content is already
on screen, and a badge counting it would be noise.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from jfl_core.storage.ui_sections import SectionState

from jfl_web.drafts import kind_label
from jfl_web.timeformat import humanize

# The marker's words when something changed but there is nothing countable to
# count -- an ad that was re-read, a score that was re-run. "3 new" is better
# where it is true; this is what is left when it is not.
UPDATED = "updated"


@dataclasses.dataclass(frozen=True)
class Section:
    """One rendered section, ready for the `_sections.html` macro.

    `default_open` is not what the *spec* asked for but what this page would
    have shown had the user never touched it -- forcing included. It is posted
    back on toggle, and it is what makes "this user went against the default"
    mean something later.
    """

    key: str
    title: str
    summary: str = ""
    marker: str = ""
    open: bool = True
    default_open: bool = True
    level: int = 2


def resolve(
    key: str,
    title: str,
    *,
    state: SectionState | None,
    default_open: bool,
    forced_open: bool = False,
    summary: str = "",
    changed_at: dt.datetime | None = None,
    item_times: Iterable[dt.datetime | None] = (),
    level: int = 2,
) -> Section:
    """Apply the three rules above and work out the marker. See the module docstring."""
    effective_default = default_open or forced_open
    if forced_open:
        is_open = True
    elif state is not None:
        is_open = state.is_open
    else:
        is_open = default_open
    return Section(
        key=key,
        title=title,
        summary=summary,
        marker=("" if is_open else marker(state, changed_at=changed_at, item_times=item_times)),
        open=is_open,
        default_open=effective_default,
        level=level,
    )


def marker(
    state: SectionState | None,
    *,
    changed_at: dt.datetime | None = None,
    item_times: Iterable[dt.datetime | None] = (),
) -> str:
    """ "2 new", "updated", or nothing at all.

    `item_times` wins where it is given, because a count is more use than a
    word: "Drafts 1 new" says what to expect behind the fold. `changed_at` is
    the fallback for a section that holds one thing that was re-made rather than
    a list that grew.
    """
    if state is None or state.last_opened_at is None:
        return ""
    seen = state.last_opened_at
    times = [when for when in item_times if when is not None]
    if times:
        fresh = sum(1 for when in times if when > seen)
        return f"{fresh} new" if fresh else ""
    if changed_at is not None and changed_at > seen:
        return UPDATED
    return ""


def counted(number: int, singular: str, plural: str | None = None) -> str:
    """`3 requirements` / `1 requirement`. Used to build the count summaries."""
    word = singular if number == 1 else (plural if plural is not None else f"{singular}s")
    return f"{number} {word}"


def joined(*parts: str) -> str:
    """The count summary's separator, in one place so every screen uses the same one."""
    return " · ".join(part for part in parts if part)


# --------------------------------------------------------------------------
# The application detail page
# --------------------------------------------------------------------------


def ad_section(states: dict[str, SectionState], extraction: object | None) -> Section:
    """ "The ad" -- the employer's own words, read into requirements.

    This is the section the complaint was about: an application whose essentials
    and desirables fill the screen before anything you can act on. Once the ad
    has been read it collapses behind its own count, and it forces itself back
    open for every state that is not "read": nothing stored, a fetch in flight,
    a failed read, an ad waiting to be pasted.
    """
    status = getattr(extraction, "status", None)
    done = status == "done"
    requirements = list(getattr(extraction, "requirements", None) or [])
    essential = sum(1 for item in requirements if getattr(item, "necessity", "") == "essential")
    return resolve(
        "application.ad",
        "The ad",
        state=states.get("application.ad"),
        default_open=False,
        forced_open=not done,
        summary=joined(
            counted(len(requirements), "requirement"),
            f"{essential} essential" if essential else "",
        )
        if done
        else "",
        changed_at=getattr(extraction, "extracted_at", None) if done else None,
    )


def score_section(states: dict[str, SectionState], score: object | None) -> Section:
    """ "Score" -- open by default, because the current score is the current score.

    The summary counts constraints and objectives and never repeats either
    number. Two axes that are never composited must not be turned into one line
    of shorthand on the way into a fold.
    """
    breaches = list(getattr(score, "hard_gate_breaches", None) or [])
    constraints = list(getattr(score, "constraint_verdicts", None) or [])
    objectives = list(getattr(score, "objective_verdicts", None) or [])
    status = getattr(score, "status", None)
    return resolve(
        "application.score",
        "Score",
        state=states.get("application.score"),
        default_open=True,
        forced_open=score is None or status in ("pending", "failed"),
        summary=joined(
            counted(len(breaches), "must-have broken", "must-haves broken") if breaches else "",
            counted(len(constraints), "constraint") if constraints else "",
            counted(len(objectives), "objective") if objectives else "",
        ),
        changed_at=getattr(score, "updated_at", None),
    )


def score_detail_sections(
    states: dict[str, SectionState], score: object | None
) -> dict[str, Section]:
    """The long lists inside the score panel.

    The constraint and objective verdicts stay open: the silences are the
    product, and folding them away by default would hide the thing the panel is
    for. The levers and the not-stated list fold, because both are prompts to go
    and do something elsewhere rather than findings about this job.
    """
    constraints = list(getattr(score, "constraint_verdicts", None) or [])
    objectives = list(getattr(score, "objective_verdicts", None) or [])
    levers = list(getattr(score, "levers", None) or [])
    not_stated = list(getattr(score, "not_stated", None) or [])
    return {
        "constraints": resolve(
            "score.constraints",
            "What the ad says about what you asked for",
            state=states.get("score.constraints"),
            default_open=True,
            summary=counted(len(constraints), "constraint"),
            level=3,
        ),
        "objectives": resolve(
            "score.objectives",
            "Your objectives, each judged on its own",
            state=states.get("score.objectives"),
            default_open=True,
            summary=counted(len(objectives), "objective"),
            level=3,
        ),
        "levers": resolve(
            "score.levers",
            "What confirming a claim would change",
            state=states.get("score.levers"),
            default_open=False,
            summary=counted(len(levers), "claim"),
            level=3,
        ),
        "not_stated": resolve(
            "score.not_stated",
            "Not stated",
            state=states.get("score.not_stated"),
            default_open=False,
            summary=counted(len(not_stated), "question"),
            level=3,
        ),
    }


def pushbacks_section(states: dict[str, SectionState], pushbacks: Sequence[object]) -> Section:
    """ "What you have said about this score".

    Forced open while any of them is still waiting on the user to confirm what
    kind of statement it was: a pushback nobody has applied has done nothing
    yet, and a loop that quietly ignores people is the failure this feature
    exists to avoid.
    """
    # `pushback_context` hands each row over as {"pushback": row, ...}; the
    # tests and any future caller may hand the row itself. Both read the same.
    rows = [item["pushback"] if isinstance(item, dict) else item for item in pushbacks]
    waiting = sum(1 for row in rows if getattr(row, "status", None) != "applied")
    return resolve(
        "score.pushbacks",
        "What you have said about this score",
        state=states.get("score.pushbacks"),
        default_open=False,
        forced_open=waiting > 0,
        summary=counted(len(rows), "correction"),
        item_times=[getattr(row, "created_at", None) for row in rows],
        level=3,
    )


def questions_section(states: dict[str, SectionState], question_views: Sequence[object]) -> Section:
    """ "Application questions" -- forced open while one is unanswered or in flight."""
    latest = [getattr(view, "latest", None) for view in question_views]
    answered = [item for item in latest if getattr(item, "status", None) == "done"]
    needs_attention = any(
        item is None or getattr(item, "status", None) in ("pending", "failed") for item in latest
    )
    return resolve(
        "application.questions",
        "Application questions",
        state=states.get("application.questions"),
        default_open=False,
        forced_open=needs_attention or not question_views,
        # An empty section counts nothing: "0 questions" beside a panel that
        # already says "no questions added yet" is the same sentence twice.
        summary=joined(
            counted(len(question_views), "question") if question_views else "",
            f"{len(answered)} answered" if answered else "",
        ),
        item_times=[getattr(item, "updated_at", None) for item in answered],
    )


def status_section(states: dict[str, SectionState]) -> Section:
    """ "Status" -- always open by default. It is the one control this page exists for."""
    return resolve(
        "application.status",
        "Status",
        state=states.get("application.status"),
        default_open=True,
    )


def notes_section(states: dict[str, SectionState], notes: str | None) -> Section:
    """ "Notes" -- open when there is something written, folded when it is an empty box."""
    written = bool((notes or "").strip())
    return resolve(
        "application.notes",
        "Notes",
        state=states.get("application.notes"),
        default_open=written,
        summary="" if written else "empty",
    )


def timeline_section(states: dict[str, SectionState], events: Sequence[object]) -> Section:
    """ "Timeline" -- always open by default, alongside status. Agreed with the owner."""
    return resolve(
        "application.timeline",
        "Timeline",
        state=states.get("application.timeline"),
        default_open=True,
        summary=counted(len(events), "event"),
        item_times=[getattr(event, "occurred_at", None) for event in events],
    )


# --------------------------------------------------------------------------
# The drafting screen
# --------------------------------------------------------------------------


def requirements_section(
    states: dict[str, SectionState], requirements: Sequence[object], coverage: Sequence[object]
) -> Section:
    """ "Requirements", with each one's coverage status.

    Folded behind its counts once coverage has been checked; forced open before
    that, because an unchecked list is a prompt to press the button rather than
    a result to read.
    """
    statuses = [getattr(row, "status", None) for row in coverage]
    evidenced = sum(1 for status in statuses if status == "evidenced")
    return resolve(
        "drafts.requirements",
        "Requirements",
        state=states.get("drafts.requirements"),
        default_open=False,
        forced_open=not coverage,
        summary=joined(
            counted(len(requirements), "requirement"),
            f"{evidenced} evidenced" if coverage else "",
        ),
    )


def generate_section(states: dict[str, SectionState]) -> Section:
    """ "Generate" -- the action, so it is open."""
    return resolve(
        "drafts.generate",
        "Generate",
        state=states.get("drafts.generate"),
        default_open=True,
    )


def draft_history_section(states: dict[str, SectionState], entries: Sequence[object]) -> Section:
    """ "Draft history" -- open, with the newest draft inside it open and the rest folded."""
    times = [getattr(entry["draft"], "created_at", None) for entry in entries]  # type: ignore[index]
    return resolve(
        "drafts.history",
        "Draft history",
        state=states.get("drafts.history"),
        default_open=True,
        summary=counted(len(entries), "draft"),
        item_times=times,
    )


def draft_section(
    states: dict[str, SectionState], entry: Mapping[str, Any], *, newest: bool
) -> Section:
    """One stored draft. The newest stays open; older ones fold behind their date.

    Keyed by the draft's own id, which is why `ui_section_states.section_key`
    carries no closed value list -- see that table's comment.

    The summary carries the absolute date *and* the relative one, because A6's
    rule holds wherever a record's timestamp is shown and a folded draft is
    still showing one.
    """
    draft = entry["draft"]
    key = f"draft.{draft.id}"
    created_at = draft.created_at
    sentences = (draft.gate_result or {}).get("sentences", [])
    return resolve(
        key,
        kind_label(draft.kind).capitalize(),
        state=states.get(key),
        default_open=newest,
        summary=joined(
            humanize(created_at) if created_at is not None else "",
            counted(len(sentences), "sentence"),
        ),
        level=3,
    )


# --------------------------------------------------------------------------
# The profile
# --------------------------------------------------------------------------


def profile_sections(
    states: dict[str, SectionState],
    *,
    profile: object,
    capabilities: Sequence[object],
    saved_capability_keys: set[str],
    cluster_created_at: dt.datetime | None,
    cluster_pending: bool,
    force_open: str | None = None,
) -> dict[str, Section]:
    """The five sections of `/profile`, folded once they have been filled in.

    The rule across all five is the same, and it is the "anything pending is
    open" default read the other way round: a section you have not stated
    anything in is the one that still wants you, so it stays open; a section
    already answered folds behind a count of what it holds. A new user's profile
    therefore looks exactly as it does today, and a filled-in one is short.

    Nothing here carries a change marker except capabilities. Every other
    section changes only when the user types in it, and telling someone their
    own typing is new would be noise.

    `force_open` is the anchor a save has just redirected to. The fragment in a
    URL never reaches the server, so the redirect carries it as a query
    parameter too -- without it, pressing Save on a section that has just become
    "filled in" would land the user on a folded panel and read as the save
    having been lost.
    """
    constraints = list(getattr(profile, "constraints", None) or [])
    disciplines = getattr(profile, "disciplines", None)
    practises = list(getattr(disciplines, "practises", None) or [])
    not_practised = list(getattr(disciplines, "not_practised", None) or [])
    objectives = list(getattr(profile, "objectives", None) or [])
    self_assessment = getattr(profile, "self_assessment", None)
    stated_self = bool(
        (getattr(self_assessment, "depth_genuine", "") or "").strip()
        or (getattr(self_assessment, "recurring_gaps", "") or "").strip()
    )
    unevidenced = sum(1 for item in capabilities if not getattr(item, "evidence", None))
    proposed = sum(
        1 for item in capabilities if getattr(item, "key", "") not in saved_capability_keys
    )
    return {
        "constraints": resolve(
            "profile.constraints",
            "What you will and will not take",
            state=states.get("profile.constraints"),
            forced_open=force_open == "constraints",
            default_open=not constraints,
            summary=f"{len(constraints)} stated" if constraints else "not stated",
        ),
        "capabilities": resolve(
            "profile.capabilities",
            "What you can actually do",
            state=states.get("profile.capabilities"),
            forced_open=cluster_pending or force_open == "capabilities",
            default_open=not capabilities,
            summary=joined(
                counted(len(capabilities), "capability", "capabilities") if capabilities else "",
                f"{unevidenced} unevidenced" if unevidenced else "",
                f"{proposed} proposed" if proposed else "",
            ),
            changed_at=cluster_created_at,
        ),
        "disciplines": resolve(
            "profile.disciplines",
            "What you practise",
            state=states.get("profile.disciplines"),
            forced_open=force_open == "disciplines",
            default_open=not practises and not not_practised,
            summary=joined(
                f"{len(practises)} practised" if practises else "",
                f"{len(not_practised)} ruled out" if not_practised else "",
            )
            or "not stated",
        ),
        "objectives": resolve(
            "profile.objectives",
            "What this move is for",
            state=states.get("profile.objectives"),
            forced_open=force_open == "objectives",
            default_open=not objectives,
            summary=f"{len(objectives)} of 4 stated" if objectives else "not stated",
        ),
        "self_assessment": resolve(
            "profile.self-assessment",
            "How you would put your own depth",
            state=states.get("profile.self-assessment"),
            forced_open=force_open == "self-assessment",
            default_open=not stated_self,
            summary="on the record" if stated_self else "not stated",
        ),
    }


__all__ = [
    "UPDATED",
    "Section",
    "ad_section",
    "counted",
    "draft_history_section",
    "draft_section",
    "generate_section",
    "joined",
    "marker",
    "notes_section",
    "profile_sections",
    "pushbacks_section",
    "questions_section",
    "requirements_section",
    "resolve",
    "score_detail_sections",
    "score_section",
    "status_section",
    "timeline_section",
]
