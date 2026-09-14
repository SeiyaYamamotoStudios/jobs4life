"""Where the work is done, and where the posting says it is: workplace and locations.

Both are **descriptive data** refreshed on every sighting, like `requisition_id`.
Neither is identity, and neither feeds the repost fingerprint -- that stays
`title|location` (`jfl_intake.normalise`), so adding this changed nothing about
how existing history is keyed or compared.

**The honesty rule for workplace.** A job's workplace is one of `remote`,
`hybrid`, `onsite`, `unknown`, and it is only ever what the employer or the
platform *stated*:

  1. **The platform's structured field wins wherever one exists.** A value that
     literally means remote, hybrid or on-site maps to that. A value that is
     explicitly *not an answer* -- Teamtailor's `none` and `temporary`, a
     boolean saying "not remote" -- maps to `unknown` and stops there: "not
     remote" does not say hybrid or on-site, and reading the location text to
     overrule a structured answer would be us second-guessing the employer.
  2. **Only where there is no structured value** (the platform has no such
     field, or this posting's is absent, null, `unspecified` or unrecognised)
     may the employer's own location text decide -- and only when it literally
     contains the whole word `remote` or `hybrid` after normalisation, never
     both, and never preceded by `no`/`non`/`not`. That is still the employer
     stating it.
  3. Everything else is `unknown`.

**Never infer `remote` from a country or region name.** Cohere's live Ashby
board (2026-09-15) disproves it directly: a job whose primary location is
`United Kingdom` is `Hybrid`, and several `London` jobs are `Remote`. Nor is
Ashby's `isRemote` used: on that same board it was `true` for 134 of 144 jobs,
hybrid ones included. `onsite` is never inferred from text at all; it is set
only by a platform or employer value that literally means on-site.

**Per platform**, as implemented (fixtures captured 2026-09-10; live checks
2026-09-15 where noted):

  * Ashby -- `workplaceType` `Remote`/`Hybrid`/`OnSite`; null -> text rule.
  * Lever -- `workplaceType` `remote`/`hybrid`/`onsite`; `unspecified` -> text rule.
  * Workable -- `workplace` `remote`/`hybrid`/`on_site` (live: Devsinc, all
    `on_site`); if absent, `remote: true` -> remote, `remote: false` -> unknown.
  * Pinpoint -- `workplace_type` `onsite`/`remote`/`hybrid`.
  * Teamtailor -- `<remoteStatus>` `fully` -> remote, `hybrid` -> hybrid;
    `none` and `temporary` -> unknown. Live (career.teamtailor.com, 12 items):
    10 `hybrid`, 2 `none`. `none` is Teamtailor's "not remote", which is also
    what a job nobody set carries, so it is not a statement of on-site.
  * Recruitee -- exactly one of the booleans `remote`/`hybrid`/`on_site` true
    -> that; none or several true -> unknown.
  * SmartRecruiters -- `location.remote` true -> remote, `location.hybrid` true
    -> hybrid; **both false -> unknown**. The API models this as two
    independent booleans with no on-site value, so "neither" is the absence of
    a remote or hybrid claim, not a claim of on-site. Live (Bosch, 100
    postings): 62 both false, 36 hybrid, 2 remote -- one both-false posting's
    description did say "fully onsite", which is consistent with on-site being
    common but is one observation, in prose.
  * Breezy -- `location.is_remote` true -> remote; false -> unknown, because
    Breezy does not distinguish hybrid from on-site.
  * Greenhouse -- no standard field, but employers set custom `metadata`. A
    field whose normalised **name** is one of `GREENHOUSE_WORKPLACE_FIELD_NAMES`
    is read first, its **value** mapped by whole words (`remote` -> remote,
    `hybrid` -> hybrid, `on site`/`onsite`/`office` -> onsite; `office` covers
    "in office"); null, unrecognised or self-contradictory values fall through
    to the text rule. Live (Anthropic, 592 jobs): one field, `Location Type`:
    `On-Site` 503, `Remote` 21, `Hybrid (Travel-Required)` 7, null 61. The value
    is also kept verbatim as `workplace_label`, so the page shows "On-Site", not
    our word for it. Note the precedence bites: 38 Anthropic jobs are
    `On-Site` while their multi-location text includes "Remote-Friendly".
  * Rippling, Workday, Personio -- no structured field: text rule only.

**Locations** are every location the posting lists, as display strings,
deduplicated (by normalised form, first spelling kept) in a stable order --
the platform's own order with the primary first, except Rippling, whose rows
arrive in no promised order and are sorted. Workday collapses multi-location
postings to `"2 Locations"`; that string is stored as given and the individual
locations are never invented.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from jfl_core.models import BoardPlatform, Workplace

from jfl_intake.normalise import clean_text, normalise

# Platforms with no structured workplace field at all. Their boards default to
# letting unstated-workplace jobs through a workplace filter (see
# `include_unstated_by_default`); Greenhouse is here because its only source is
# optional employer-defined metadata, which many employers do not set.
PLATFORMS_WITHOUT_WORKPLACE_FIELD: frozenset[BoardPlatform] = frozenset(
    {"greenhouse", "rippling", "workday", "personio"}
)

# Greenhouse custom-field names read as workplace, compared after `normalise`.
# Observed: `Location Type` (Anthropic). The others are the obvious spellings of
# the same field; a name not listed here is simply not read.
GREENHOUSE_WORKPLACE_FIELD_NAMES = frozenset(
    {"location type", "workplace type", "work type", "remote"}
)

_NEGATIONS = frozenset({"no", "non", "not"})


def include_unstated_by_default(platform: BoardPlatform) -> bool:
    """A board's default for "include jobs whose workplace isn't stated": on
    where the platform cannot state it, off where it can. A "remote only" filter
    must not reduce a board that states nothing to nothing.
    """
    return platform in PLATFORMS_WITHOUT_WORKPLACE_FIELD


def effective_include_unstated(platform: BoardPlatform, setting: bool | None) -> bool:
    return include_unstated_by_default(platform) if setting is None else setting


def _named(text: str, *, allow_onsite: bool) -> set[Workplace] | None:
    """Every workplace a piece of text names by whole word; None if any of them
    is negated ("non-remote"), because then the text is not a plain statement.
    `allow_onsite` is for employer *workplace values* only -- location text
    never yields on-site.
    """
    words = normalise(text).split()
    found: set[Workplace] = set()
    for i, word in enumerate(words):
        kind: Workplace | None = None
        if word == "remote":
            kind = "remote"
        elif word == "hybrid":
            kind = "hybrid"
        elif allow_onsite and (
            word in ("onsite", "office") or (word == "on" and words[i + 1 : i + 2] == ["site"])
        ):
            kind = "onsite"
        if kind is None:
            continue
        if i > 0 and words[i - 1] in _NEGATIONS:
            return None
        found.add(kind)
    return found


def from_location_text(texts: Iterable[str | None]) -> Workplace:
    """Rule 2: the employer's location text, literally saying remote or hybrid
    -- exactly one of them, un-negated, across all of the posting's locations.
    """
    found: set[Workplace] = set()
    for text in texts:
        if not text:
            continue
        named = _named(text, allow_onsite=False)
        if named is None:
            return "unknown"
        found |= named
    return found.pop() if len(found) == 1 else "unknown"


def from_enum(value: object) -> Workplace | None:
    """A platform enum value that literally means one workplace (`Remote`,
    `hybrid`, `OnSite`, `on_site`, `onsite`), or None for anything else --
    including null and `unspecified`, which carry no answer.
    """
    if not isinstance(value, str):
        return None
    squashed = normalise(value).replace(" ", "")
    if squashed in ("remote", "hybrid", "onsite"):
        return squashed  # type: ignore[return-value]
    return None


def resolve(structured: Workplace | None, texts: Iterable[str | None]) -> Workplace:
    """Rules 1-3 for a platform whose structured value is `structured`: None
    means "no structured answer here", so the text rule may decide. Platforms
    with an explicit non-answer return `unknown` themselves and never call this.
    """
    return structured if structured is not None else from_location_text(texts)


def greenhouse_metadata(metadata: object) -> tuple[Workplace, str] | None:
    """The workplace an employer's Greenhouse custom fields state, and the
    verbatim value that stated it. None when no recognised field gives exactly
    one answer -- the caller then falls through to the location text.
    """
    if not isinstance(metadata, list):
        return None
    answers: dict[Workplace, str] = {}
    for field in metadata:
        if not isinstance(field, Mapping):
            continue
        name = field.get("name")
        if not isinstance(name, str) or normalise(name) not in GREENHOUSE_WORKPLACE_FIELD_NAMES:
            continue
        value = clean_text(field.get("value"))
        if value is None:
            continue
        named = _named(value, allow_onsite=True)
        if named is not None and len(named) == 1:
            answers.setdefault(next(iter(named)), value)
    if len(answers) != 1:
        return None
    ((kind, label),) = answers.items()
    return kind, label


def dedupe_locations(values: Iterable[object]) -> tuple[str, ...]:
    """Display strings, blanks and non-strings dropped, deduplicated by
    normalised form with the first spelling and position kept.
    """
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = clean_text(value)
        if text is None:
            continue
        key = normalise(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return tuple(out)


def split_location_text(text: object) -> tuple[str, ...]:
    """One employer-written string listing several places (Greenhouse's
    `"New York City, NY; San Francisco, CA | New York City, NY"`), split on `;`
    and `|` only. Commas are kept: they separate a city from its region, not
    one place from another.
    """
    if not isinstance(text, str):
        return ()
    parts = [p for chunk in text.split(";") for p in chunk.split("|")]
    return dedupe_locations(parts)


def as_bool(mapping: Mapping[str, Any], key: str) -> bool | None:
    """A strictly boolean field, or None if absent or any other type."""
    value = mapping.get(key)
    return value if isinstance(value, bool) else None
