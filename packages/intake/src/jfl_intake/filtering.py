"""Matching open jobs against a saved filter. Pure: no database, no network, no clock.

**A filter is a lens, never a fetch parameter.** Boards are always watched whole
and filtered here, over stored data. Filtering at the source would make jobs
appear to vanish from a board's history whenever the filter or a category
changed -- exactly the false signal the watching engine exists to keep out.

**Text is matched the way fingerprints are normalised** (`jfl_intake.normalise`:
NFKC, casefold, punctuation to spaces, whitespace collapsed), on both sides.
Input is **comma-separated alternatives**; an alternative matches a target when
**every one of its words appears as a whole word** in the target, in any order.
So `engineering manager` matches "Manager, Engineering" and "Engineering
Manager, Platform" but not "Engineer"; `engineer` does not match "Engineering".
No stemming and no synonyms, for the reason `normalise` gives: every one would be
a guess, and a guess here silently hides or shows a role.

  * **Title includes** -- any alternative matching is enough; empty = no constraint.
  * **Title excludes** -- any alternative matching excludes the job, and wins.
  * **Location** -- any alternative matching **any one** of the job's locations
    (all of the alternative's words within that one location string). A job
    with no `locations` recorded (a row from before they were captured) is
    matched against its single `location`.
  * **Workplace** -- decided by the filter's `workplace_mode`:
      - `custom`: the job's workplace must be in the selected set; empty = any.
        Exactly the behaviour from before the presets, so an old filter matches
        what it matched.
      - `remote_only`: the employer's structured field states `remote`. A posting
        that also calls itself remote-friendly (`jfl_intake.workplace.says_remote_friendly`)
        stays in, with the conflict on the row: the structured field is the
        employer's own statement, and hiding it would resolve the conflict silently
        (owner ruling, 2026-09-16 -- before that, the words excluded it, which on
        the captured Anthropic board hid both of its remote jobs). A job whose
        field is not `remote` never gets in on the words alone.
      - `remote_friendly`: `remote` or `hybrid`, or the posting says
        remote-friendly **even where the structured field says on-site** --
        Anthropic's 38 `On-Site` jobs whose location reads
        "Remote-Friendly (Travel-Required)" (owner ruling, 2026-09-15). A board
        the owner marks `hybrid_too_heavy` contributes only the jobs remote only
        would, so remote friendly is never narrower than remote only.

**Hybrid is flagged, never promoted.** Platforms say "Hybrid" without a day
count, so each match carries a `WorkplaceNote` the page turns into words:
`hybrid_days_not_stated`, `says_remote_friendly`, or
`listed_onsite_says_remote_friendly` -- the conflict shown on the row rather than
silently resolved in either direction. Where a hybrid job's board has an
exception that matches it and carries the owner's note, that note is surfaced
instead of "days not stated": the owner's words are evidence the platform lacks.

**Unknown workplace is never silently hidden.** A job whose workplace is not
stated either passes (its board is set to include such jobs, which is the
default for platforms with no workplace field -- `jfl_intake.workplace`), or is
counted in `FilterResult.hidden_unstated`, which the page states with a way to
show them.

**Board exceptions** are per-board "also include" rules, OR'd with the filter.
An exception widens **workplace and location only**: the title includes and
excludes still apply to anything it lets through, so it can never widen the
role. A job's reason for matching is kept -- the filter itself, a board
exception (with the owner's note), or its board's include-unstated setting -- so
the page can say why a job the employer calls On-Site is on a "remote" list
without rewriting what the employer published.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from jfl_core.models import BoardFilterException, JobFilter, Workplace, WorkplaceMode

from jfl_intake.normalise import normalise
from jfl_intake.workplace import says_remote_friendly

# Each alternative is the set of words that must all appear.
Terms = tuple[frozenset[str], ...]

MatchReason = Literal["filter", "exception", "unstated"]

# What a matched job's workplace evidence says, for the page to put into words.
# Codes, not prose: the employer's own label is rendered verbatim beside them.
WorkplaceNote = Literal[
    "hybrid_days_not_stated",  # hybrid, and nothing says how many days
    "says_remote_friendly",  # the posting says remote-friendly; not listed on-site
    "listed_onsite_says_remote_friendly",  # listed on-site, yet says remote-friendly
]


class FilterableJob(Protocol):
    @property
    def board_id(self) -> uuid.UUID: ...
    @property
    def title(self) -> str: ...
    @property
    def location(self) -> str | None: ...
    @property
    def locations(self) -> Sequence[str]: ...
    @property
    def workplace(self) -> Workplace: ...
    @property
    def workplace_label(self) -> str | None: ...


def parse_terms(text: str | None) -> Terms:
    """Comma-separated alternatives, each normalised to its set of words.
    Alternatives that normalise to nothing (`", ,"`, a lone `-`) are dropped.
    """
    if not text:
        return ()
    alternatives: list[frozenset[str]] = []
    for part in text.split(","):
        words = frozenset(normalise(part).split())
        if words and words not in alternatives:
            alternatives.append(words)
    return tuple(alternatives)


def already_covered(alternative: frozenset[str], terms: Terms) -> bool:
    """True when adding `alternative` to `terms` would match nothing new.

    An alternative matches a title when *all* of its words are in it
    (`any_alternative_matches`), so an existing alternative whose words are a
    subset of the candidate's already matches every title the candidate would:
    with "technical lead" in the filter, "Technical Lead Manager" adds nothing.
    Exact equality is the special case. This is the test for "already in the
    filter" -- comparing keys for equality alone let a suggestion through that
    the filter already covered.
    """
    return any(existing <= alternative for existing in terms)


def _words(text: str | None) -> frozenset[str]:
    return frozenset(normalise(text).split())


def any_alternative_matches(terms: Terms, targets: Iterable[str | None]) -> bool:
    """True if some alternative has all its words in some single target.
    Callers decide what empty terms mean; this returns False for them.
    """
    target_words = [_words(t) for t in targets]
    return any(alt <= words for alt in terms for words in target_words)


@dataclass(frozen=True, slots=True)
class CompiledFilter:
    workplaces: frozenset[Workplace] = frozenset()
    includes: Terms = ()
    excludes: Terms = ()
    location: Terms = ()
    mode: WorkplaceMode = "custom"
    # A preset's equivalent of adding `unknown` to the custom set: jobs that
    # state nothing at all pass the workplace test, for one view.
    unknown_shown: bool = False

    @classmethod
    def from_saved(cls, saved: JobFilter) -> CompiledFilter:
        return cls(
            mode=saved.workplace_mode,
            workplaces=frozenset(saved.workplaces),
            includes=parse_terms(saved.title_includes),
            excludes=parse_terms(saved.title_excludes),
            location=parse_terms(saved.location),
        )

    @property
    def is_empty(self) -> bool:
        workplace_open = self.mode == "custom" and not self.workplaces
        return workplace_open and not (self.includes or self.excludes or self.location)

    def with_unknown_workplace(self) -> CompiledFilter:
        """The same filter with jobs of unstated workplace let through -- the
        page's one-click "show them" for jobs hidden only for that reason.
        Under `custom`, `unknown` joins a non-empty workplace set; a preset
        always constrains workplace, so it takes the flag instead.
        """
        if self.mode != "custom":
            return replace(self, unknown_shown=True)
        if not self.workplaces:
            return self
        return replace(self, workplaces=self.workplaces | {"unknown"})


@dataclass(frozen=True, slots=True)
class CompiledException:
    exception: BoardFilterException
    workplaces: frozenset[Workplace]
    location: Terms

    @classmethod
    def from_saved(cls, exception: BoardFilterException) -> CompiledException:
        return cls(
            exception=exception,
            workplaces=frozenset(exception.workplaces),
            location=parse_terms(exception.location),
        )


def _job_locations(job: FilterableJob) -> Sequence[str | None]:
    return job.locations if job.locations else [job.location]


def title_passes(f: CompiledFilter, job: FilterableJob) -> bool:
    if f.excludes and any_alternative_matches(f.excludes, [job.title]):
        return False
    return not f.includes or any_alternative_matches(f.includes, [job.title])


def _location_passes(terms: Terms, job: FilterableJob) -> bool:
    return not terms or any_alternative_matches(terms, _job_locations(job))


def workplace_passes(f: CompiledFilter, job: FilterableJob, *, hybrid_too_heavy: bool) -> bool:
    """The filter's own workplace test, per `f.mode` (see the module docstring).
    `hybrid_too_heavy` is the job's board's setting; only `remote_friendly`
    reads it.
    """
    if f.mode == "custom":
        return not f.workplaces or job.workplace in f.workplaces
    evidence = says_remote_friendly(job)
    if f.unknown_shown and job.workplace == "unknown" and not evidence:
        return True
    if f.mode == "remote_only" or hybrid_too_heavy:
        return job.workplace == "remote"
    return job.workplace in ("remote", "hybrid") or evidence


def _is_unstated(f: CompiledFilter, job: FilterableJob) -> bool:
    """Is this job's workplace genuinely not stated, for the include-unstated
    rule? Under a preset, a posting whose words say remote-friendly has stated
    something the preset has already judged -- so it is not "unstated" and
    cannot slip back in through that rule, or be counted as hidden by it.
    """
    if job.workplace != "unknown":
        return False
    return f.mode == "custom" or not says_remote_friendly(job)


def _exception_matches(rule: CompiledException, job: FilterableJob) -> bool:
    workplace_ok = not rule.workplaces or job.workplace in rule.workplaces
    return workplace_ok and _location_passes(rule.location, job)


Verdict = Literal["filter", "exception", "unstated", "hidden_unstated", "excluded"]


def evaluate(
    job: FilterableJob,
    f: CompiledFilter,
    *,
    include_unstated: bool,
    exceptions: Sequence[CompiledException] = (),
    hybrid_too_heavy: bool = False,
) -> tuple[Verdict, BoardFilterException | None]:
    """One job's fate, and the exception that let it through if one did.

    Precedence: the filter itself; then the board's exceptions, in order; then
    the board's include-unstated setting. A job that fails only because its
    workplace is not stated, on a board that does not include such jobs, is
    `hidden_unstated` -- counted, never silently dropped.
    """
    if not title_passes(f, job):
        return "excluded", None

    location_ok = _location_passes(f.location, job)
    workplace_ok = workplace_passes(f, job, hybrid_too_heavy=hybrid_too_heavy)
    if location_ok and workplace_ok:
        return "filter", None

    for rule in exceptions:
        if _exception_matches(rule, job):
            return "exception", rule.exception

    if location_ok and _is_unstated(f, job):
        return ("unstated", None) if include_unstated else ("hidden_unstated", None)
    return "excluded", None


def workplace_note(
    job: FilterableJob, exceptions: Sequence[CompiledException] = ()
) -> tuple[WorkplaceNote | None, BoardFilterException | None]:
    """What the page should say about a matched job's workplace, in every mode:
    the evidence is a fact about the posting, not about the preset chosen.

    Returns a note code, or -- for a hybrid job that one of its board's
    exceptions matches and whose note is not blank -- that exception instead,
    so the owner's words ("~1 day a week") replace "days not stated". An
    exception with no note has no words to offer, so the badge stays.
    """
    if says_remote_friendly(job):
        if job.workplace == "onsite":
            return "listed_onsite_says_remote_friendly", None
        return "says_remote_friendly", None
    if job.workplace != "hybrid":
        return None, None
    for rule in exceptions:
        if rule.exception.note.strip() and _exception_matches(rule, job):
            return None, rule.exception
    return "hybrid_days_not_stated", None


@dataclass(frozen=True, slots=True)
class Match[J: FilterableJob]:
    job: J
    reason: MatchReason
    # The exception that let the job through; set only when `reason == "exception"`.
    exception: BoardFilterException | None = None
    workplace_note: WorkplaceNote | None = None
    # The exception whose owner's note describes this hybrid job's workplace, in
    # place of `hybrid_days_not_stated`. May be set whatever the `reason`.
    workplace_exception: BoardFilterException | None = None


@dataclass(slots=True)
class FilterResult[J: FilterableJob]:
    matches: list[Match[J]] = field(default_factory=list)
    open_total: int = 0
    hidden_unstated: int = 0

    @property
    def via_exception(self) -> int:
        return sum(1 for m in self.matches if m.reason == "exception")

    @property
    def via_unstated(self) -> int:
        return sum(1 for m in self.matches if m.reason == "unstated")

    @property
    def hybrid_days_not_stated(self) -> int:
        return sum(1 for m in self.matches if m.workplace_note == "hybrid_days_not_stated")


def apply_filter[J: FilterableJob](
    jobs: Iterable[J],
    saved: JobFilter,
    *,
    include_unstated: Mapping[uuid.UUID, bool],
    exceptions: Iterable[BoardFilterException] = (),
    show_hidden_unstated: bool = False,
    hybrid_too_heavy: Collection[uuid.UUID] = frozenset(),
) -> FilterResult[J]:
    """Filter `jobs`, keeping their order. `include_unstated` is each board's
    effective setting (a board missing from it does not include unstated jobs).
    `show_hidden_unstated` is the page's one-view "show them": jobs whose
    workplace is not stated match the filter itself. `hybrid_too_heavy` is the
    ids of boards whose hybrid the owner has left out of remote friendly.
    """
    compiled = CompiledFilter.from_saved(saved)
    if show_hidden_unstated:
        compiled = compiled.with_unknown_workplace()
    by_board: dict[uuid.UUID, list[CompiledException]] = {}
    for exception in exceptions:
        by_board.setdefault(exception.board_id, []).append(CompiledException.from_saved(exception))

    result: FilterResult[J] = FilterResult()
    for job in jobs:
        result.open_total += 1
        board_exceptions = by_board.get(job.board_id, ())
        verdict, via = evaluate(
            job,
            compiled,
            include_unstated=include_unstated.get(job.board_id, False),
            exceptions=board_exceptions,
            hybrid_too_heavy=job.board_id in hybrid_too_heavy,
        )
        if verdict == "hidden_unstated":
            result.hidden_unstated += 1
        elif verdict != "excluded":
            note, described_by = workplace_note(job, board_exceptions)
            result.matches.append(
                Match(
                    job=job,
                    reason=verdict,
                    exception=via,
                    workplace_note=note,
                    workplace_exception=described_by,
                )
            )
    return result
