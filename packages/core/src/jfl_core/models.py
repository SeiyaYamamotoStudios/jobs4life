"""Domain schemas. Pydantic, no HTTP or framework types anywhere in core."""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field

Provenance = Literal["document", "adjudicated"]
SpanKind = Literal["bullet", "paragraph", "heading"]
JobSource = Literal["paste", "file"]
Necessity = Literal["essential", "desirable", "unstated"]
# Deliberately not met/unmet -- coverage is measured against the corpus, never
# against the candidate. See CLAUDE.md's decisions log.
CoverageStatus = Literal["evidenced", "partial", "absent", "contradicted"]
QuestionStatus = Literal["open", "answered", "dismissed"]


class Sentence(BaseModel):
    idx: int
    start_offset: int
    end_offset: int


class Span(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    document_id: uuid.UUID | None = None
    provenance: Provenance
    kind: SpanKind
    section_path: str | None = None
    ordinal: int | None = None
    text: str
    content_hash: str
    char_start: int | None = None  # offsets into the source document; None for adjudicated spans
    char_end: int | None = None
    sentences: list[Sentence] = Field(default_factory=list)


class SpanCandidate(BaseModel):
    span: Span
    score: float


class RunRecord(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    trace_id: uuid.UUID
    parent_run_id: uuid.UUID | None = None
    component: Literal["gate", "evals", "ingest", "generate"]
    stage: str
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: Decimal | None = None
    latency_ms: int | None = None
    outcome: Literal["ok", "error", "refused", "skipped"] = "ok"
    error: str | None = None
    attributes: dict[str, object] | None = None
    started_at: dt.datetime


# --------------------------------------------------------------------------
# Generation (domain 2a): a job anchors extracted requirements, which anchor
# per-run coverage rows and gap questions. Server-defaulted timestamps
# (`created_at`) are left off these models, same convention as `Span` above --
# they are not known until the row is written, and no caller here needs them
# back. `JobSummary` is the one exception, built for `jfl job list`'s display.
# --------------------------------------------------------------------------


class Job(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    source: JobSource
    employer: str | None = None
    title: str | None = None
    location: str | None = None
    url: str | None = None
    raw_text: str
    content_hash: str


class JobRequirement(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    job_id: uuid.UUID
    ordinal: int  # order in the ad; ordering only, NOT part of the id
    text: str
    necessity: Necessity


class JobSummary(BaseModel):
    """One row of `jfl job list` -- not a table, just what that listing needs."""

    id: uuid.UUID
    employer: str | None
    title: str | None
    requirement_count: int
    created_at: dt.datetime


class RequirementCoverage(BaseModel):
    """One row of an append-only history: a fresh row per coverage run, never an
    update. See `requirement_coverage` in tables.py for why.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    requirement_id: uuid.UUID
    trace_id: uuid.UUID  # groups every requirement checked in one coverage run
    status: CoverageStatus
    cited_span_ids: list[uuid.UUID]
    evidence_note: str


class GapQuestion(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    requirement_id: uuid.UUID
    question: str
    status: QuestionStatus = "open"
    answer_text: str | None = None
    answered_at: dt.datetime | None = None
    resulting_span_id: uuid.UUID | None = None


# --------------------------------------------------------------------------
# Generation (domain 2b-core): a draft anchored on a job, gated automatically.
# See CLAUDE.md's decisions log, "The claim gate runs automatically on
# generated text." `gate_result` is `jfl_gate.schema.GateOutput.model_dump()`
# -- kept as a plain dict here rather than typed against that model, since
# `jfl_core` has no dependency on `jfl_gate` (core holds no HTTP/framework/
# other-package types; see CLAUDE.md's architectural constraints).
# --------------------------------------------------------------------------

DraftKind = Literal["cv_bullets", "cover_letter"]


class Draft(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    job_id: uuid.UUID
    kind: DraftKind
    text: str
    gate_result: dict[str, object]
    trace_id: uuid.UUID


# --------------------------------------------------------------------------
# Application tracker (slice A5). No model call anywhere in this slice --
# see CLAUDE.md's build order. `created_at`/`updated_at`/`occurred_at` are
# NOT left off these models the way `Job`'s are: the list and detail screens
# these exist for show timestamps on every row, so the repository always
# populates them from what Postgres actually wrote (via `RETURNING`), never
# guessed client-side.
# --------------------------------------------------------------------------

ApplicationStatus = Literal[
    "interested", "applied", "screening", "interviewing", "offer", "rejected", "withdrawn"
]

# Slice B3. Where the background read of the pasted ad has got to -- deliberately
# a separate axis from `ApplicationStatus`, which is where the application is in
# the world. An extraction that failed says nothing about whether the user has
# applied, and collapsing the two would make one lie about the other.
ExtractionStatus = Literal["none", "pending", "done", "failed"]

# A closed set, and never a message. The worker writes this while holding the
# user's decrypted API key, and a free-text error column is exactly where a
# careless `str(exc)` from the SDK ends up. Wording lives in the web layer, where
# it can change without a migration.
ExtractionErrorCode = Literal[
    "no_api_key",
    "api_key_rejected",
    "no_job_ad",
    "ad_too_long",
    "model_refused",
    "model_error",
    "credential_unreadable",
]


class Application(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    job_id: uuid.UUID | None = None
    title: str
    employer: str | None = None
    url: str | None = None
    status: ApplicationStatus
    source: str | None = None
    notes: str | None = None
    extraction_status: ExtractionStatus = "none"
    extraction_error_code: ExtractionErrorCode | None = None
    extracted_at: dt.datetime | None = None
    # True while `title` is a placeholder taken from the ad's first line. See
    # `jfl_core.db.tables.applications` for why this is a column and not a guess.
    title_is_provisional: bool = False
    # Soft delete: set when archived, cleared when restored. See tables.py.
    archived_at: dt.datetime | None = None
    created_at: dt.datetime
    updated_at: dt.datetime


class ApplicationEvent(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    application_id: uuid.UUID
    from_status: ApplicationStatus | None = None
    to_status: ApplicationStatus
    note: str | None = None
    occurred_at: dt.datetime
    created_at: dt.datetime


class ApplicationDetail(BaseModel):
    """What the detail screen needs: the row, plus its full timeline in
    chronological order (oldest first -- how a timeline reads).
    """

    application: Application
    events: list[ApplicationEvent]


class ExtractionInput(BaseModel):
    """What the `extract_job_ad` handler needs to do its work, read out of the
    database under the task's own `user_id` rather than carried in the payload.

    The ad text is deliberately NOT in the task payload. It is already stored
    once, verbatim, in `jobs.raw_text`; a second copy in `tasks.payload` would
    be a second place a user's job ad lives, read back by admin queries and
    quoted into log lines, for no gain over a scoped read of the row.
    """

    application_id: uuid.UUID
    job_id: uuid.UUID
    raw_text: str


class ApplicationExtraction(BaseModel):
    """The extraction panel's whole state, in one read.

    `status` is the only thing the UI needs while work is in flight; the rest is
    the result, and is empty until `status == "done"`.
    """

    application_id: uuid.UUID
    status: ExtractionStatus
    error_code: ExtractionErrorCode | None = None
    extracted_at: dt.datetime | None = None
    has_job_ad: bool = False
    employer: str | None = None
    title: str | None = None
    location: str | None = None
    requirements: list[JobRequirement] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Background work (slice B1). A row in `tasks`, as the queue and the worker see
# it -- see `jfl_core.storage.tasks`.
# --------------------------------------------------------------------------

TaskStatus = Literal["pending", "running", "succeeded", "failed"]


class Task(BaseModel):
    """One unit of background work.

    `payload` carries `repr=False` deliberately. This object reaches log lines
    and tracebacks, and while a payload is only ever meant to hold arguments
    (see `tables.py`), "only ever meant to" is not a guarantee -- so the default
    repr shows the id, kind and status and leaves the arguments out. Code that
    genuinely needs the payload asks for `task.payload`, which is a decision
    someone made rather than a field that came along for the ride.
    """

    id: uuid.UUID
    user_id: uuid.UUID
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict, repr=False)
    status: TaskStatus
    attempts: int
    max_attempts: int
    last_error: str | None = None
    scheduled_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    created_at: dt.datetime
    updated_at: dt.datetime


class ReclaimResult(BaseModel):
    """What one sweep of the stale-`running` reclaim did.

    Two lists, not one count, because they mean different things to whoever is
    reading the logs: `requeued` is a worker that died and a task that will run
    again, `failed` is a task that died for the last time and now needs a human.
    """

    requeued: list[uuid.UUID] = Field(default_factory=list)
    failed: list[uuid.UUID] = Field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.requeued or self.failed)


# --------------------------------------------------------------------------
# Watched job boards (domain 3, intake). See `jfl_core.db.tables.watched_boards`
# for the schema and `jfl_intake.engine` for the rule these rest on: only a
# complete check may close a presence interval.
# --------------------------------------------------------------------------

BoardPlatform = Literal[
    "greenhouse",
    "ashby",
    "lever",
    "workday",
    "smartrecruiters",
    "rippling",
    "breezy",
    "teamtailor",
    "personio",
    "recruitee",
    "pinpoint",
    "workable",
]
# These must stay equal, by construction, to `_BOARD_PLATFORMS` in
# `jfl_core.db.tables` and to `default_registry()`'s platforms in
# `jfl_intake.adapters` -- see `test_board_platform_sources_agree` in
# `packages/intake/tests/test_intake_adapters.py`, which fails the build if
# the three ever drift again the way they did between 2026-09-10 and
# 2026-09-11 (models.py listed twelve; the CHECK constraint and the table
# constant still listed the original four; migration 3c540957b0d2 fixed it).
BoardCheckStatus = Literal["complete", "incomplete", "truncated", "unreachable", "failed", "held"]
# What an adapter can report. `held` is not in it: holding is the check engine's
# decision about a complete fetch, never something a fetch can say about itself.
FetchStatus = Literal["complete", "incomplete", "truncated", "unreachable", "failed"]
BoardCheckErrorCode = Literal[
    "not_found",
    "http_client_error",
    "rate_limited",
    "server_error",
    "timeout",
    "connection_error",
    "malformed_response",
    "unidentifiable_job",
    "count_mismatch",
    "page_cap_reached",
    # A token-paged listing (Workable) handed back a posting id already
    # collected in this check without exhausting its pages -- a failure to
    # make progress, never a legitimate way to reach `complete`.
    "duplicate_posting",
    "request_budget_exhausted",
    "deadline_exceeded",
    "listing_ceiling",
    "unsupported_board",
    "drop_guard",
]
# `reposted` is reported INSTEAD of `new`, not as well: a reposted job is a new
# external id, and saying both would count it twice. `returned` is the same
# external id reopening, and never shares a code path with `reposted`.
BoardJobEventKind = Literal["new", "reposted", "returned", "gone"]
# How a posting says the work is done. `unknown` is a real answer, not a gap in
# the data model: many platforms (Greenhouse, Workday, Rippling, Personio) carry
# no structured field at all, and a job whose workplace is not stated must never
# be presented as on-site -- or silently hidden by a "remote only" filter. How
# each platform's field maps onto these is `jfl_intake.workplace`'s docstring.
Workplace = Literal["remote", "hybrid", "onsite", "unknown"]


class WatchedBoard(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    platform: BoardPlatform
    board_url: str
    board_key: dict[str, str]
    label: str | None = None
    created_at: dt.datetime
    next_check_at: dt.datetime
    last_check_id: uuid.UUID | None = None
    consecutive_failures: int = 0
    baseline_check_id: uuid.UUID | None = None
    held_check_id: uuid.UUID | None = None
    drop_accepted: bool = False
    # None = the platform default (`jfl_intake.workplace.include_unstated_by_default`).
    include_unstated_workplace: bool | None = None


class BoardCheck(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    board_id: uuid.UUID
    started_at: dt.datetime
    finished_at: dt.datetime
    status: BoardCheckStatus
    jobs_seen: int
    expected_total: int | None = None
    error_code: BoardCheckErrorCode | None = None
    is_baseline: bool = False


class BoardJob(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    board_id: uuid.UUID
    external_id: str
    requisition_id: str | None = None
    title: str
    location: str | None = None
    url: str | None = None
    fingerprint: str
    # Descriptive, refreshed on every applied sighting like `requisition_id`;
    # never identity and never part of the fingerprint. See `ObservedJob`.
    workplace: Workplace = "unknown"
    workplace_label: str | None = None
    locations: list[str] = Field(default_factory=list)
    first_seen_check_id: uuid.UUID
    first_seen_at: dt.datetime
    last_seen_at: dt.datetime
    reposted_from_job_id: uuid.UUID | None = None


class BoardJobPresence(BaseModel):
    """One interval of continuous presence. Open while `closed_at` is None."""

    id: uuid.UUID
    user_id: uuid.UUID
    job_id: uuid.UUID
    opened_check_id: uuid.UUID
    opened_at: dt.datetime
    closed_check_id: uuid.UUID | None = None
    closed_at: dt.datetime | None = None


class BoardJobEvent(BaseModel):
    """One thing that happened to a job, derived from the intervals -- there is
    no events table. `new`/`reposted` come from `board_jobs.first_seen_check_id`
    (never a baseline check), `returned` from an interval opened by a later check
    than the job's first, and `gone` from an interval's close.
    """

    kind: BoardJobEventKind
    board_id: uuid.UUID
    check_id: uuid.UUID
    at: dt.datetime
    job: BoardJob


class ObservedJob(BaseModel):
    """One job as an adapter saw it, normalised, before any history is applied.

    `fingerprint` is computed when the record is built
    (`jfl_intake.normalise.fingerprint`), so every consumer compares the same
    string rather than re-deriving it.
    """

    model_config = {"frozen": True}

    # The platform's id for this POSTING -- the finest grain the platform
    # exposes, and the identity every diff keys on. For Workday that is the
    # `externalPath` suffix (`R171808-1`), not the requisition (`R171808`):
    # a role re-listed as `R171808-2` must read as a new posting, or reposts
    # are invisible. History kept at a coarser grain can never be re-keyed.
    external_id: str
    title: str
    location: str | None = None
    url: str | None = None
    fingerprint: str
    # The employer's requisition, where the platform exposes one (Workday's
    # `bulletFields[0]`, Greenhouse's `requisition_id`); None elsewhere.
    # STORED ONLY. No rule reads it: a requisition re-listed under a new posting
    # id has not been observed yet, and rules come from observed patterns. It is
    # recorded so that pattern can be learned from real history later.
    requisition_id: str | None = None
    # How the posting says the work is done, from the platform's structured
    # field where one exists -- see `jfl_intake.workplace` for the mapping and
    # the rule for platforms without one. Descriptive data, refreshed on every
    # sighting; deliberately NOT part of the fingerprint, which stays
    # `title|location` so repost detection over existing history is unchanged.
    workplace: Workplace = "unknown"
    # The employer's own words for it, verbatim, where they wrote some -- a
    # Greenhouse custom field value such as "On-Site". None when the workplace
    # came from a platform enum or boolean, which are not the employer's words.
    workplace_label: str | None = None
    # Every location the posting lists (primary first where the platform marks
    # one), display strings, deduplicated in a stable order. `location` above is
    # kept exactly as it was because the fingerprint is built from it.
    locations: tuple[str, ...] = ()


class KnownBoardJob(BaseModel):
    """What the check engine needs to know about a job already on record."""

    job_id: uuid.UUID
    external_id: str
    fingerprint: str
    is_open: bool
    last_closed_at: dt.datetime | None = None
    # Already the source of a repost. A closed posting can be reposted once;
    # after that the repost is the thing that may be reposted again.
    has_repost_successor: bool = False


class BoardCheckState(BaseModel):
    """A board's history as of a check, read under a row lock on the board.

    `known_jobs` is not every job ever seen: it is every job with an open
    interval, every job whose external id was just observed, and every job that
    closed inside the repost window -- exactly the set the diff can touch.
    """

    board_id: uuid.UUID
    baseline_check_id: uuid.UUID | None = None
    held_check_id: uuid.UUID | None = None
    drop_accepted: bool = False
    known_jobs: list[KnownBoardJob] = Field(default_factory=list)


class PlannedNewJob(BaseModel):
    job: ObservedJob
    reposted_from_job_id: uuid.UUID | None = None


class PlannedSighting(BaseModel):
    """An observed job that is already on record: `job_id` is its row."""

    job_id: uuid.UUID
    job: ObservedJob


class CheckPlan(BaseModel):
    """What one check means for a board's history. Produced by the pure
    `jfl_intake.engine.plan_check`, applied by
    `PostgresBoardRepository.apply_check_plan`. Every job list is empty unless
    `status == "complete"`.
    """

    board_id: uuid.UUID
    status: BoardCheckStatus
    error_code: BoardCheckErrorCode | None = None
    jobs_seen: int
    expected_total: int | None = None
    is_baseline: bool = False
    new_jobs: list[PlannedNewJob] = Field(default_factory=list)
    returned: list[PlannedSighting] = Field(default_factory=list)
    still_open: list[PlannedSighting] = Field(default_factory=list)
    gone_job_ids: list[uuid.UUID] = Field(default_factory=list)

    @property
    def changes_job_state(self) -> bool:
        return self.status == "complete"

    def summary(self) -> dict[str, int]:
        """Counts for a log line. `new` excludes baseline jobs and reposts."""
        reposted = sum(1 for n in self.new_jobs if n.reposted_from_job_id is not None)
        return {
            "baseline": len(self.new_jobs) if self.is_baseline else 0,
            "new": 0 if self.is_baseline else len(self.new_jobs) - reposted,
            "reposted": reposted,
            "returned": len(self.returned),
            "gone": len(self.gone_job_ids),
        }


class JobFilter(BaseModel):
    """A user's saved lens over their watched boards' open jobs.

    A lens, never a fetch parameter: boards are always watched whole and
    filtered locally, so changing this can never make a job appear to vanish
    from a board's history. Matching lives in `jfl_intake.filtering` (pure);
    this is only what is stored. Empty text and an empty workplace set each
    mean "no constraint".
    """

    workplaces: list[Workplace] = Field(default_factory=list)
    title_includes: str = ""
    title_excludes: str = ""
    location: str = ""
    updated_at: dt.datetime | None = None


class BoardFilterException(BaseModel):
    """A per-board "also include" rule, OR'd with the saved `JobFilter`.

    Widens workplace and location only -- the saved filter's title includes and
    excludes still apply to every job it lets through. `note` is the owner's own
    words, verbatim. Empty `workplaces` / `location` mean "any".
    """

    id: uuid.UUID
    board_id: uuid.UUID
    workplaces: list[Workplace] = Field(default_factory=list)
    location: str = ""
    note: str = ""
    created_at: dt.datetime
    updated_at: dt.datetime


TitleSuggestionStatus = Literal["pending", "done", "failed"]

# A subset of ExtractionErrorCode's codes -- the ones that can actually happen
# on this call. No `no_job_ad` / `ad_too_long` (there is no ad here), see
# `jfl_generate.titles.suggest_titles` and `jfl_worker.handlers.title_suggestions`.
TitleSuggestionErrorCode = Literal[
    "no_api_key",
    "api_key_rejected",
    "model_refused",
    "model_error",
    "credential_unreadable",
]


class SuggestedTitle(BaseModel):
    """One adjacent title the model proposed, plus a very short note on how it
    differs -- never a property named `reason`. See CLAUDE.md's 2026-09-02
    decision and PLAN.md's C7a.
    """

    title: str
    gloss: str = ""


class TitleSuggestion(BaseModel):
    """One saved filter phrase's suggestion state -- slice C7a.

    One row per (user, phrase_key): the call runs once per phrase, ever, and is
    cached here. `suggestions` is empty until `status == "done"`. Suggestions
    are never added to the filter by this row existing -- see
    `jfl_web.routes.title_suggestions`, where a tickbox is the only path onto
    `JobFilter.title_includes`.
    """

    id: uuid.UUID
    phrase: str
    phrase_key: str
    status: TitleSuggestionStatus
    suggestions: list[SuggestedTitle] = Field(default_factory=list)
    error_code: TitleSuggestionErrorCode | None = None
    dismissed_at: dt.datetime | None = None
    created_at: dt.datetime
    updated_at: dt.datetime


class DueBoard(BaseModel):
    """A board the scheduler found due, and whose it is. Ids only, by design --
    see `jfl_core.storage.boards.PostgresBoardScheduler`.
    """

    board_id: uuid.UUID
    user_id: uuid.UUID
