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
  * **Workplace** -- the job's workplace must be in the selected set; empty = any.

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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from jfl_core.models import BoardFilterException, JobFilter, Workplace

from jfl_intake.normalise import normalise

# Each alternative is the set of words that must all appear.
Terms = tuple[frozenset[str], ...]

MatchReason = Literal["filter", "exception", "unstated"]


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

    @classmethod
    def from_saved(cls, saved: JobFilter) -> CompiledFilter:
        return cls(
            workplaces=frozenset(saved.workplaces),
            includes=parse_terms(saved.title_includes),
            excludes=parse_terms(saved.title_excludes),
            location=parse_terms(saved.location),
        )

    @property
    def is_empty(self) -> bool:
        return not (self.workplaces or self.includes or self.excludes or self.location)

    def with_unknown_workplace(self) -> CompiledFilter:
        """The same filter with `unknown` added to a non-empty workplace set --
        the page's one-click "show them" for jobs hidden only for that reason.
        """
        if not self.workplaces:
            return self
        return CompiledFilter(
            workplaces=self.workplaces | {"unknown"},
            includes=self.includes,
            excludes=self.excludes,
            location=self.location,
        )


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


Verdict = Literal["filter", "exception", "unstated", "hidden_unstated", "excluded"]


def evaluate(
    job: FilterableJob,
    f: CompiledFilter,
    *,
    include_unstated: bool,
    exceptions: Sequence[CompiledException] = (),
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
    workplace_ok = not f.workplaces or job.workplace in f.workplaces
    if location_ok and workplace_ok:
        return "filter", None

    for rule in exceptions:
        if (not rule.workplaces or job.workplace in rule.workplaces) and _location_passes(
            rule.location, job
        ):
            return "exception", rule.exception

    if location_ok and job.workplace == "unknown":
        return ("unstated", None) if include_unstated else ("hidden_unstated", None)
    return "excluded", None


@dataclass(frozen=True, slots=True)
class Match[J: FilterableJob]:
    job: J
    reason: MatchReason
    exception: BoardFilterException | None = None


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


def apply_filter[J: FilterableJob](
    jobs: Iterable[J],
    saved: JobFilter,
    *,
    include_unstated: Mapping[uuid.UUID, bool],
    exceptions: Iterable[BoardFilterException] = (),
    show_hidden_unstated: bool = False,
) -> FilterResult[J]:
    """Filter `jobs`, keeping their order. `include_unstated` is each board's
    effective setting (a board missing from it does not include unstated jobs).
    `show_hidden_unstated` is the page's one-view "show them": `unknown` is
    added to the workplace set, so those jobs match the filter itself.
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
        verdict, via = evaluate(
            job,
            compiled,
            include_unstated=include_unstated.get(job.board_id, False),
            exceptions=by_board.get(job.board_id, ()),
        )
        if verdict == "hidden_unstated":
            result.hidden_unstated += 1
        elif verdict != "excluded":
            result.matches.append(Match(job=job, reason=verdict, exception=via))
    return result
