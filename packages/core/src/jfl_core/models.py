"""Domain schemas. Pydantic, no HTTP or framework types anywhere in core."""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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

# What produced a `cv_documents` version -- see `jfl_core.db.tables.cv_documents`.
# Each is a real event that writes a new row: the worker `generated` it, the user
# `edited` its wording, switched its `template`, refreshed its `header` (and
# interests) from the profile, or had their edits `checked` by the claim gate.
CvDocumentStatus = Literal["generated", "edited", "template", "header", "checked"]


class Draft(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    job_id: uuid.UUID
    kind: DraftKind
    text: str
    gate_result: dict[str, object]
    trace_id: uuid.UUID
    # None on the object `generate_draft` builds before it is stored -- Postgres
    # sets it (`drafts.created_at`, server-defaulted). Populated whenever a draft
    # is read back (`list_drafts`), which is where B5's screen gets the
    # timestamp it shows alongside each draft.
    created_at: dt.datetime | None = None


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
    # Slice C7: "Track as application" could not read the posting's description
    # off the board -- unsupported platform, a 404, or a fetch that never came
    # back after retrying. Distinct from `no_job_ad`, which is a genuinely empty
    # `raw_job_text`; here there is no ad at all yet, and the fix is the same
    # paste box a manual application starts from (`POST /applications/{id}/ad`).
    "description_unavailable",
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
    # Slice C7: the watched-board job this application was created from, or
    # None for one added by paste. ON DELETE SET NULL -- losing the board (or
    # the job falling off it) must never take the tracked application with it,
    # so this is a provenance pointer, never something the application's own
    # life depends on.
    board_job_id: uuid.UUID | None = None
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
# Application questions -- two equal paths, "check my answer" and "draft one
# for me". See CLAUDE.md's 2026-09-18 decision and NEXT.md's task 4.
# --------------------------------------------------------------------------


class ApplicationQuestion(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    application_id: uuid.UUID
    question_text: str
    created_at: dt.datetime


AnswerKind = Literal["user", "draft"]
AnswerStatus = Literal["pending", "done", "failed"]

# A closed set, never a message -- same reasoning as `ExtractionErrorCode`: the
# worker writes this while holding the user's decrypted API key.
# `no_requirements` is `draft_application_answer`'s own precondition failure
# (see `jfl_worker.handlers.application_questions`), not an SDK error.
AnswerErrorCode = Literal[
    "no_api_key",
    "api_key_rejected",
    "model_refused",
    "model_error",
    "credential_unreadable",
    "no_requirements",
]


class ApplicationQuestionAnswer(BaseModel):
    """One attempt to answer a question -- append-only: a second attempt is a
    new row, never an UPDATE, so what was tried first stays readable.
    `kind='user'` is the user's own words, checked by the claim gate;
    `kind='draft'` is generated from the corpus and gated automatically, with
    `answer_text` empty until the draft call finishes. `gate_result` is
    `jfl_gate.schema.GateOutput.model_dump()`, kept as a plain dict here for
    the same reason `Draft.gate_result` is (jfl_core has no dependency on
    jfl_gate). `assessment` is only ever populated for `kind='user'` -- see
    `jfl_core.db.tables.application_question_answers`'s docstring for why a
    draft is never asked to assess itself.
    """

    id: uuid.UUID
    user_id: uuid.UUID
    question_id: uuid.UUID
    kind: AnswerKind
    answer_text: str = ""
    status: AnswerStatus = "pending"
    error_code: AnswerErrorCode | None = None
    gate_result: dict[str, object] | None = None
    assessment: dict[str, object] | None = None
    model: str | None = None
    trace_id: uuid.UUID | None = None
    created_at: dt.datetime
    updated_at: dt.datetime


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
# The saved job filter's main workplace control. `remote_only` is strict: the
# employer states remote and nothing in the posting calls it remote-friendly.
# `remote_friendly` adds low-commitment hybrid -- and because platforms say
# "Hybrid" without a day count, hybrid is shown there badged "days not stated",
# never presented as confirmed low commitment. `custom` is the workplace
# checkboxes, exactly as they behaved before the presets existed, so filters
# saved before then keep matching what they matched. Semantics:
# `jfl_intake.filtering`.
WorkplaceMode = Literal["remote_only", "remote_friendly", "custom"]


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
    # The owner's judgement that this employer's hybrid is more than about a day
    # a week: its hybrid and remote-friendly jobs are left out of the
    # `remote_friendly` preset. Plain remote jobs still pass.
    hybrid_too_heavy: bool = False


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

    workplace_mode: WorkplaceMode = "custom"
    # Consulted only under `workplace_mode == "custom"`, but kept whatever the
    # mode, so switching back to Custom restores what was ticked.
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


# -- capability clustering ---------------------------------------------------
#
# One cheap model call that groups a user's **confirmed** facts into capability
# labels. A role is not a capability ("FX pricing platforms" spans several
# roles) and neither is a single fact (thirty-three overlapping CVs produce
# dozens of near-duplicates), so the grouping is the one thing here worth a
# model.
#
# Nothing in this table is a profile row. A proposal becomes a
# `jfl_core.profile.Capability` only when the user accepts it, and their own
# label wins permanently if they rename it -- see
# `jfl_web.routes.profile.accept_capability_proposal`.

CapabilityClusterStatus = Literal["pending", "done", "failed"]

# The same subset `TitleSuggestionErrorCode` takes, and for the same reason:
# there is no document here to be missing or too long, so the only failures are
# about the credential, the model, or the call.
CapabilityClusterErrorCode = Literal[
    "no_api_key",
    "api_key_rejected",
    "model_refused",
    "model_error",
    "credential_unreadable",
]

# Inside JSONB, so no CHECK constraint holds it -- the Pydantic model below is
# the only write path, the same trade the profile itself makes.
CapabilityProposalState = Literal["open", "accepted", "rejected"]


class ProposedCapability(BaseModel):
    """One capability the model proposed, and the confirmed facts it covers.

    `label` is the model's words until the user renames it, at which point it
    is theirs and is never regenerated over. `span_ids` are corpus spans and
    become the accepted `Capability`'s `evidence`, which is what makes the row
    citable rather than merely asserted; `fact_ids` name the candidate facts
    those spans came from, so the screen can say which facts a proposal would
    account for and which are still unplaced.

    No property named `reason` here or in the schema that produces it -- see
    CLAUDE.md's 2026-09-02 decision.
    """

    label: str
    fact_ids: list[uuid.UUID] = Field(default_factory=list)
    span_ids: list[uuid.UUID] = Field(default_factory=list)
    state: CapabilityProposalState = "open"


class CapabilityCluster(BaseModel):
    """One clustering run: what was sent, what came back, and what it left over.

    `unclustered_fact_ids` are facts the model was given and placed in no
    capability; `omitted_fact_ids` are facts that did not fit in one bounded
    call at all. Both are kept and both are shown -- a fact is never silently
    dropped, and the next run picks up whatever this one did not place.

    `trace_id` is how the screen prices the run (`RunRepository.cost_for_trace`)
    without this table ever holding a cost of its own.
    """

    id: uuid.UUID
    status: CapabilityClusterStatus
    trace_id: uuid.UUID
    proposals: list[ProposedCapability] = Field(default_factory=list)
    fact_count: int = 0
    unclustered_fact_ids: list[uuid.UUID] = Field(default_factory=list)
    omitted_fact_ids: list[uuid.UUID] = Field(default_factory=list)
    error_code: CapabilityClusterErrorCode | None = None
    dismissed_at: dt.datetime | None = None
    created_at: dt.datetime
    updated_at: dt.datetime

    @property
    def open_proposals(self) -> list[ProposedCapability]:
        """The ones still awaiting an answer. Accepted and rejected rows stay in
        the row so that re-running is a decision about new material rather than
        a chance to be asked the same question twice.
        """
        return [p for p in self.proposals if p.state == "open"]


# -- profile suggestions from uploaded CVs -----------------------------------
#
# A CV states claims about the world, which become `candidate_facts` and are
# confirmed one at a time. It also states plain **settings** -- which
# disciplines someone practises, where they have worked, what level they have
# been operating at -- and those are not claims to be measured against a corpus.
# One cheap model call reads the user's own CVs and proposes them.
#
# Nothing here is a profile row. A proposal reaches `profiles.data` only when
# the user accepts it, and a setting they have already stated always wins --
# see `jfl_web.profilesuggestions`.

ProfileSuggestionStatus = Literal["pending", "done", "failed"]

# The same subset `CapabilityClusterErrorCode` takes, and for the same reason:
# the CVs are already stored, so the only failures are about the credential,
# the model, or the call.
ProfileSuggestionErrorCode = Literal[
    "no_api_key",
    "api_key_rejected",
    "model_refused",
    "model_error",
    "credential_unreadable",
]

# **The whole allow-list, and the exclusions are its complement.** A CV records
# what someone has done; it does not state what they now require. Comp, contract
# type, right to work, notice, a categorical no and a workplace preference are
# therefore absent by construction rather than by a rule someone has to
# remember -- a guessed constraint of any of those kinds would be read by
# scoring as the user's own requirement.
#
# `level` is an **observation** ("has been operating at engineering-manager
# level"), never a demand: `level_floor` is a choice the user makes, so
# accepting one still asks them for a stance and for their own words.
ProfileSuggestionKind = Literal["discipline", "not_discipline", "location", "level"]
PROFILE_SUGGESTION_KINDS: tuple[ProfileSuggestionKind, ...] = (
    "discipline",
    "not_discipline",
    "location",
    "level",
)

# Inside JSONB, so no CHECK constraint holds it -- the Pydantic model below is
# the only write path, the same trade the profile itself makes.
SuggestionState = Literal["open", "accepted", "rejected"]


class ProposedSetting(BaseModel):
    """One profile setting a CV suggests, and the CV's own words behind it.

    `values` is a list for every kind so that one shape serves all four: a
    discipline and a level observation carry one entry, a location carries the
    places in the order the model read them, most recent first.
    `source_lines` runs in step with it -- **every proposal shows what in the CV
    suggested it**, which is what makes a wrong one obvious rather than
    plausible.

    `key` is `jfl_core.ids.setting_key`, derived from the kind and the folded
    values: it names the proposal in a form action, and it is what makes a
    rejection stick, since a later run's identical suggestion folds to the same
    key and is never offered again.

    No property named `reason` here or in the schema that produces it -- see
    CLAUDE.md's 2026-09-02 decision.
    """

    kind: ProfileSuggestionKind
    key: str
    values: list[str] = Field(default_factory=list)
    source_lines: list[str] = Field(default_factory=list)
    state: SuggestionState = "open"


class ProfileSuggestionRun(BaseModel):
    """One run of the CV-reading call: what it read and what it proposed.

    `cv_count` is how many stored CVs went into the call, so the screen can say
    what the answer was drawn from. `trace_id` is how that screen prices the run
    (`RunRepository.cost_for_trace`) without this table holding a cost of its
    own.
    """

    id: uuid.UUID
    status: ProfileSuggestionStatus
    trace_id: uuid.UUID
    proposals: list[ProposedSetting] = Field(default_factory=list)
    cv_count: int = 0
    error_code: ProfileSuggestionErrorCode | None = None
    created_at: dt.datetime
    updated_at: dt.datetime

    @property
    def open_proposals(self) -> list[ProposedSetting]:
        """The ones still awaiting an answer. Answered rows stay on the run, so
        re-running is a decision about new material rather than a chance to be
        asked the same question twice.
        """
        return [p for p in self.proposals if p.state == "open"]


class DueBoard(BaseModel):
    """A board the scheduler found due, and whose it is. Ids only, by design --
    see `jfl_core.storage.boards.PostgresBoardScheduler`.
    """

    board_id: uuid.UUID
    user_id: uuid.UUID


class JobFeedMark(BaseModel):
    """One user's record of one derived `BoardJobEvent` in the "what changed"
    feed: when they were first shown it, and whether they dismissed it. The event
    itself is not stored -- `(job_id, kind, check_id)` names it. See
    `jfl_intake.feed` for how marks decide what the feed shows.
    """

    id: uuid.UUID
    job_id: uuid.UUID
    check_id: uuid.UUID
    kind: BoardJobEventKind
    event_at: dt.datetime
    first_seen_at: dt.datetime
    dismissed_at: dt.datetime | None = None


# -- the profile, docs/profile-schema.md -------------------------------------
#
# One denormalised JSONB row per save. The models live in `jfl_core.profile`,
# not here: that module is the only write path into `profiles.data`, and JSONB
# carries no CHECK constraint, so keeping the shape and its validation in one
# file is what stands in for one. `ProfileAnswer`, `ProfileObjective`,
# `ProfileRuledOut` and `ProfileQuestionKey` were retired with the three tables
# they described (2026-09-21).


# -- CV intake, PLAN.md slice B6 ---------------------------------------------
#
# A new user's corpus is too small to score or draft against, so onboarding
# starts from their CVs. But a CV cannot simply *become* the corpus: grounding
# on a CV makes every later CV "supported" and switches the over-claim
# measurement off silently (CLAUDE.md, 2026-09-18). So a CV is stored verbatim
# in the sent-document store -- form, never truth -- a model proposes candidate
# facts from it, and only what the user confirms, fact by fact, becomes a
# corpus span.

# Where the markdown behind a corpus document actually lives. `local_file` is
# the CLI's `corpus/*.md` on the owner's disk; `hosted` is markdown this
# deployment holds in `documents.text`, because a hosted user has no file --
# see `jfl_core.corpus_source`.
DocumentStorageKind = Literal["local_file", "upload", "paste", "hosted"]

# Three states, and the middle one is the point: a fact a CV claims but the
# user has not confirmed is *kept* and never grounds anything (CLAUDE.md's
# "confirmed, claimed-unconfirmed, absent"). `proposed` is that middle state.
CandidateFactState = Literal["proposed", "confirmed", "rejected"]

# Joins a confirmed fact to its probe answer in the single corpus line the two
# become together. See `CandidateFact.corpus_text`.
PROBE_JOIN = " -- "

CvExtractionStatus = Literal["pending", "done", "failed"]

# A closed set, never free text -- same reasoning as `ExtractionErrorCode`:
# the worker writes this column while holding the user's decrypted API key, and
# a formatted exception is where a credential leaks. Wording lives in the web
# layer.
CvExtractionErrorCode = Literal[
    "no_cv_text",
    "no_api_key",
    "api_key_rejected",
    "credential_unreadable",
    "cv_too_long",
    "model_refused",
    "model_error",
]


class StoredCv(BaseModel):
    """One CV in the sent-document store, plus how its extraction went.

    Deliberately carries no span ids and no corpus anything: this is the store
    that must never be reachable from a grounding query, and the type that
    describes it should not tempt anyone to join it to one.
    """

    id: uuid.UUID
    path: str
    title: str | None = None
    content_hash: str
    created_at: dt.datetime
    extraction_status: CvExtractionStatus = "pending"
    extraction_error_code: CvExtractionErrorCode | None = None
    facts_proposed: int = 0
    # Character count, not the text: a CV listing page has no business
    # rendering the CV, and `text` on this model would invite it.
    length_chars: int = 0


class ProposedFact(BaseModel):
    """One candidate fact on its way into `candidate_facts`, before it has an
    id or a state. Built by `jfl_generate.cv_facts.to_proposed_facts` from what
    the model returned; written by
    `PostgresCandidateFactRepository.add_proposed`, which supplies the user.

    No `user_id` field on purpose -- the repository is constructed with the one
    user it may act on, and a caller that could name a user is a caller that
    could name the wrong one.
    """

    sent_document_id: uuid.UUID | None = None
    role_label: str
    role_key: str
    source_line: str
    fact_text: str
    probe: str | None = None
    fingerprint: str
    # Position within the CV it came from, used only to order roles the way the
    # CV ordered them. Never part of any identity -- see `jfl_core.ids`.
    ordinal: int = 0


class CandidateFact(BaseModel):
    """A fact a CV claims, in one of three states.

    `confirmed_text` is the user's own words where they edited the model's
    proposal, and it -- not `fact_text` -- is what reaches the corpus. A model
    tidying a user's answer into a neater corpus fact is the ratchet in
    miniature (CLAUDE.md, "A gap answer is stored verbatim"), so nothing on the
    path from `confirmed_text` to a span involves a model.

    `span_id` is set once the fact is in the corpus, and is the only link from
    here into the grounding store. It points at an ordinary
    `provenance='document'` span, because the write-back goes through markdown
    -- see `jfl_core.corpus_source`.
    """

    id: uuid.UUID
    user_id: uuid.UUID
    sent_document_id: uuid.UUID | None = None
    role_label: str
    role_key: str
    source_line: str
    fact_text: str
    probe: str | None = None
    probe_answer: str | None = None
    state: CandidateFactState = "proposed"
    confirmed_text: str | None = None
    span_id: uuid.UUID | None = None
    fingerprint: str
    created_at: dt.datetime
    updated_at: dt.datetime

    @property
    def corpus_text(self) -> str:
        """What would go to the corpus if this were confirmed as it stands: the
        user's edit where there is one, the proposal otherwise, with the probe's
        answer joined on.

        The answer is the half that matters -- "led how many?" -> "nine
        engineers" -- so it belongs in the span the claim gate reads, not only
        in this row. `PROBE_JOIN` is the one character sequence this system adds
        to the user's words; nothing is reworded and no model is on this path.
        """
        edited = (self.confirmed_text or "").strip()
        statement = edited or self.fact_text
        answer = (self.probe_answer or "").strip()
        return f"{statement}{PROBE_JOIN}{answer}" if answer else statement

    @property
    def needs_probe_answer(self) -> bool:
        """True while a fact that asserts a number or ownership has no answer to
        its probe -- the confirmation screen blocks confirming it until then.
        """
        return bool(self.probe) and not self.probe_answer


class RoleGroup(BaseModel):
    """One role's worth of candidate facts, for the confirmation screen.

    Grouped by `role_key` rather than by CV: the same role appears in every CV
    that mentions it, and confirming it thirty-three times is exactly the
    friction that gets a tool abandoned.
    """

    role_key: str
    role_label: str
    proposed: int = 0
    confirmed: int = 0
    rejected: int = 0

    @property
    def total(self) -> int:
        return self.proposed + self.confirmed + self.rejected


class FactCounts(BaseModel):
    proposed: int = 0
    confirmed: int = 0
    rejected: int = 0

    @property
    def total(self) -> int:
        return self.proposed + self.confirmed + self.rejected


# -- two scores for one application, PLAN.md slice B4 ------------------------
#
# **Two axes, 1-10 each, never composited.** CLAUDE.md's standing decision and
# PLAN.md's B4: "do I want this" and "could I get this" diverge constantly and
# averaging them destroys exactly the signal that makes them worth having, so
# there is deliberately no third number anywhere in this file, in the table, in
# the repository or on the page. Both ship **unmeasured and labelled as such**
# -- there is no golden set for fit, and the measured over-claim rate is a
# different number about a different thing.
#
# Nothing here is named `reason`: see CLAUDE.md's 2026-09-02 decision. The
# paragraph behind each number is an `assessment`.

ScoreStatus = Literal["pending", "done", "failed"]

# A subset of ExtractionErrorCode's codes plus one of this call's own. Closed
# set, never a message: the worker writes this column while holding the user's
# decrypted API key, and a free-text column is where a careless `str(exc)` from
# the SDK ends up.
ScoreErrorCode = Literal[
    "no_api_key",
    "api_key_rejected",
    "model_refused",
    "model_error",
    "credential_unreadable",
    # The ad has not been read yet, so there are no requirements to score
    # "could I get this" against. Not a model failure -- nothing was called.
    "no_requirements",
]


# What the **ad** evidences about one thing the user said matters. Not a rating
# of the job and not a prediction that they would enjoy it: `docs/profile-schema.md`
# is explicit that computed person-job fit predicts satisfaction at rho ~= .28,
# so the model is asked what the ad says, and `jfl_core.fit` derives the number
# from these four words. `silent` is the one that earns the feature: it is the
# question to ask at interview, and it is deliberately distinguishable from
# `contradicted` -- the claim gate's corpus-silence lesson, applied to job ads.
FitVerdict = Literal["evidenced", "partial", "silent", "contradicted"]
FIT_VERDICTS: tuple[str, ...] = ("evidenced", "partial", "silent", "contradicted")


class ConstraintVerdict(BaseModel):
    """What the ad evidences about one constraint the user recorded.

    `kind` and `stance` are copied from the stored constraint and typed `str`
    rather than against `jfl_core.profile`'s Literals: this is JSONB read back
    out of `application_scores`, and a run recorded under a vocabulary that has
    since changed must still parse rather than fail the whole page. The write
    path is validated where it belongs, on the profile.

    `label` is the constraint in the user's own words, stored beside the
    verdict so the panel never has to re-read the profile -- and so a verdict
    stays readable after the constraint it judged has been edited away.
    """

    kind: str
    stance: str
    label: str = ""
    verdict: FitVerdict
    note: str = ""


class ObjectiveVerdict(BaseModel):
    """One of the user's objectives, judged on its own against the ad.

    `rank` is the objective's own rank, which is how `jfl_core.fit` weights it;
    `objective` is the user's own words echoed back so the page never has to
    re-read the profile to label a verdict. Deliberately no number: objectives
    are ranked and scored separately and never blended, and inventing a
    per-objective scale is the first step towards something that gets averaged.
    """

    rank: int
    objective: str = ""
    verdict: FitVerdict
    note: str = ""

    @model_validator(mode="before")
    @classmethod
    def _read_a_row_from_before_this_vocabulary(cls, data: Any) -> Any:
        """Rows stored before the four-word verdict still parse.

        `application_scores.objective_verdicts` is JSONB written by an earlier
        shape: the slot was `ordinal` and `verdict` held a free sentence. A
        score panel that raises on one of those is a worse failure than one
        that reports the ad as silent about that objective -- which is also the
        truthful reading, since nothing in an old row states what the ad
        evidenced in these terms. The sentence is not thrown away: it moves to
        `note`, where the page still shows it.

        Only reading is lenient. The write path is the Literal, so nothing new
        can be stored outside the four words.
        """
        if not isinstance(data, dict):
            return data
        row = dict(data)
        if "rank" not in row and "ordinal" in row:
            row["rank"] = row.pop("ordinal")
        verdict = row.get("verdict")
        if isinstance(verdict, str) and verdict not in FIT_VERDICTS:
            row["verdict"] = "silent"
            if not row.get("note"):
                row["note"] = verdict
        return row


class HardGateBreach(BaseModel):
    """A hard gate the ad breaks, stated in plain words rather than folded
    silently into a number. `gate` names which one (location, workplace, comp
    floor, contract, right to work, a categorical no); `breach` says what the
    ad does about it.
    """

    gate: str
    breach: str


class ScoreLever(BaseModel):
    """Something **claimed and not evidenced** that would move "could I get this".

    Two things have that status and they are treated identically: an
    unconfirmed fact from the user's own CV, and a capability they tiered on
    the profile without a corpus span behind it. Neither is evidence -- only
    confirmed corpus facts are (CLAUDE.md, 2026-09-18) -- so a lever is the
    honest way to say "you claim X; confirm it and this moves from 5 to 7"
    without quietly crediting the claim.

    `claim_kind` says which of the two this is (`cv_fact` or `capability`) and
    `tier` carries the depth a capability was claimed at, so the panel can say
    "claimed at working level, no evidence" rather than presenting a profile
    row as though a CV had said it. `fact_text` and `role_label` are copied
    verbatim from the stored claim, never from the model's paraphrase of it.
    """

    fact_text: str
    role_label: str = ""
    would_move_to: int | None = None
    note: str = ""
    # Free `str` for the same reason `NotStated.question_key` is: this is
    # stored JSONB, and a row written before a name changed must still read.
    claim_kind: str = "cv_fact"
    tier: str = ""


class NotStated(BaseModel):
    """A part of the profile this user has not filled in. Reported as "not
    stated" and never guessed at. `question_key` names the section or field, as
    free `str`: this is stored JSONB, and a row written before a name was
    retired must still parse back -- which is exactly what happened when the
    eighteen B3a questions gave way to `jfl_core.profile.Profile`.
    """

    question_key: str
    wording: str = ""


class ApplicationScore(BaseModel):
    """One scoring run against one application. Append-only across runs: a
    re-score inserts a new row, the page shows the latest, and the history
    stays readable. A row's own `status` moves `pending` -> `done`/`failed`
    once, which is the run's state, not a rewrite of an earlier score.

    `cost_usd` is the whole run's cost -- the scoring call, plus the coverage
    call when this run had to make one -- because that is what the user was
    billed for pressing the button. `trace_id` ties it to the `runs` rows.
    """

    id: uuid.UUID
    application_id: uuid.UUID
    status: ScoreStatus
    error_code: ScoreErrorCode | None = None
    could_get_score: int | None = None
    could_get_assessment: str = ""
    want_it_score: int | None = None
    want_it_assessment: str = ""
    # What `want_it_score` was derived from. Stored so the panel can show the
    # verdicts under the number and recompute the tally from them, which is
    # what stops the number saying something the list below it does not.
    constraint_verdicts: list[ConstraintVerdict] = Field(default_factory=list)
    objective_verdicts: list[ObjectiveVerdict] = Field(default_factory=list)
    hard_gate_breaches: list[HardGateBreach] = Field(default_factory=list)
    levers: list[ScoreLever] = Field(default_factory=list)
    not_stated: list[NotStated] = Field(default_factory=list)
    model: str | None = None
    cost_usd: Decimal | None = None
    trace_id: uuid.UUID | None = None
    created_at: dt.datetime
    updated_at: dt.datetime


# --------------------------------------------------------------------------
# Pushback: the user disagreeing with a score, recorded whether or not it
# changes anything. The arithmetic is `jfl_core.pushback`; these are the
# stored shapes.
#
# Nothing here is named `reason` -- see CLAUDE.md's 2026-09-02 decision. The
# model's one-line account of why it classified a pushback the way it did is a
# `classification_note`.
# --------------------------------------------------------------------------

PushbackStatus = Literal["awaiting_classification", "classified", "applied"]

# Same closed-set discipline as `ScoreErrorCode`: the worker writes this column
# holding the user's decrypted API key, and a free-text column is where a
# careless `str(exc)` ends up.
PushbackErrorCode = Literal[
    "no_api_key",
    "api_key_rejected",
    "model_refused",
    "model_error",
    "credential_unreadable",
]

ClassificationSource = Literal["none", "model", "user"]


class Pushback(BaseModel):
    """One disagreement with one score, and exactly what it did.

    `user_text` is the user's own words, byte for byte. `shown_score` and
    `shown_explanation` are the stimulus they were answering -- preferences are
    constructed at the moment of elicitation rather than retrieved, so a
    pushback typed after reading our sentence is partly a response to our
    sentence, and the sentence is part of the record.

    `applied_delta` is None until the pushback has been read and applied, and
    0.0 is a normal, frequent and honest value afterwards: every capability
    claim that the number should go up lands on it, by design.
    """

    id: uuid.UUID
    application_id: uuid.UUID
    score_id: uuid.UUID
    axis: str
    dimension: str
    target_dimension: str = ""
    shown_score: int | None = None
    shown_explanation: str = ""
    user_text: str
    asserted_direction: str
    asserted_points: float = 1.0
    status: PushbackStatus = "awaiting_classification"
    # Free `str` rather than the Literal for the same reason `ScoreLever.claim_kind`
    # is: this is read back out of a table that outlives a vocabulary change.
    classification: str | None = None
    classification_source: ClassificationSource = "none"
    classification_note: str = ""
    new_information: bool | None = None
    error_code: PushbackErrorCode | None = None
    trace_id: uuid.UUID | None = None
    applied_delta: float | None = None
    prior_observations: int | None = None
    disposition: str | None = None
    effect: dict[str, Any] = Field(default_factory=dict)
    evidence_question: str = ""
    resulting_span_id: uuid.UUID | None = None
    created_at: dt.datetime
    updated_at: dt.datetime
    applied_at: dt.datetime | None = None
    # Set when the user said "Not what I meant". The row keeps what it did, for
    # the record; nothing that sums the log counts it any more.
    withdrawn_at: dt.datetime | None = None

    @property
    def awaiting_user(self) -> bool:
        """Whether this has not been applied yet -- still being read, or the
        reading failed and the user has to say which they meant.
        """
        return self.status != "applied"

    @property
    def withdrawn(self) -> bool:
        return self.withdrawn_at is not None

    @property
    def moved(self) -> bool:
        return bool(self.applied_delta)


class ScoreOverride(BaseModel):
    """A displayed number the user set by hand, for one application only.

    The escape hatch, and honest about being one: shown as an override wherever
    it appears, scoped to this application, feeding no dimension's displacement,
    reaching no other job's score, and changing nothing about what the claim
    gate will say about a CV bullet that asserts the same thing. `value` of None
    is the override taken off, kept as a row because the record of what you
    believed in March stays readable.
    """

    id: uuid.UUID
    application_id: uuid.UUID
    axis: str
    value: int | None = None
    note: str = ""
    created_at: dt.datetime
