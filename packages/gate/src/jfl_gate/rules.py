"""The deterministic rule tier: runs after the model, never before it.

See CLAUDE.md, "The deterministic tier buys guarantees, not savings." The tier does
not save a model call -- deciding claim-versus-framing needs the model on every
sentence regardless, and the corpus is already cached, so a marginal check is cheap.
What it buys instead is a guarantee no prompt can give: a rule either fires or it
doesn't, with no sampling variance.

**A deterministic rule can prove contact, never crossing.** Term overlap with a
corpus line like "Rust: no experience" shows a claim *touches* a boundary; only the
model can judge whether it *crosses* one. So every rule here can only ever move a
sentence from `supported` to `review` -- never to `unsupported`, never a downgrade
of an existing `review`/`unsupported`, never an upgrade to `supported`. And a rule
never touches `drift_label`: the taxonomy is the model's classification, not a
heuristic's. What fired is recorded in `SentenceResult.rule_flags` instead.

Rules apply only to `kind == "claim"` sentences. Framing is never touched -- see
CLAUDE.md's repeated point that over-flagging framing is the failure that gets this
tool switched off; a deterministic rule with no judgement at all is the last place
that discipline should slip.

## Rule 1 -- unsourced numbers

Extracts every numeric token from a claim and from the whole corpus, normalises
both the same way, and flags any claim number absent from the corpus set.

Why this only ever escalates to `review`, never fails the claim outright: an
analysis of 2,347 units of the owner's real material found `invented_quantity`
never fired once -- every number traced back to the corpus. So the realistic case
this rule catches is not a fabricated figure; it is a number legitimately
*derived* from the corpus but not present in it verbatim -- a tenure summed from
two dates, a total rolled up from several bullets. Hard-failing those would be
exactly the over-flagging CLAUDE.md warns against, so the rule can only ask a
human (or the model, next pass) to look, never assert a contradiction it cannot
actually see.

## Rule 2 -- boundary contact

Some corpus spans record things stated explicitly as *not* true about the person,
or boundaries to hold. That section is identified by a marker substring in its
`section_path` -- "Things stated explicitly as NOT true, or as boundaries to
hold" is the heading text seen in the owner's own corpus, but the match is against
substrings of that phrase (`_BOUNDARY_SECTION_MARKERS`, below) precisely so this
is a corpus-*format* convention, not a hardcoded fact about one user's corpus.

From those spans, a set of *distinctive* terms is built: not every word in the
boundary section, only ones that are rare elsewhere in the corpus, so a term
every span uses (e.g. a company name repeated throughout) never fires the rule
just for being present. Unigrams only -- multi-word boundary phrases ("computer
vision", "self-managed Kafka") are not handled as phrases, only as their
constituent distinctive words. Nothing in the real material examined so far has
shown that phrase-level matching is needed; add it if and when it is.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from jfl_core.models import Span

from jfl_gate.schema import GateOutput, SentenceResult

# --- rule 1: unsourced numbers ------------------------------------------------

# Optional leading currency symbol, digits optionally thousands-grouped, an
# optional decimal tail, an optional trailing percent sign. Two alternatives,
# tried in this order deliberately: the first requires at least one real
# ",ddd" group, so a plain ungrouped run of more than three digits ("1200")
# never partially matches it (Python's re takes the first alternative that
# matches *at all*, not the longest one, so a `*` here would silently truncate
# "1200" to "120" and leave a stray "0"). Anything without a qualifying comma
# group falls through to the second, plain-digit alternative. Deliberately
# does not swallow a trailing comma/period from surrounding prose ("...raised
# $40, then...") -- thousands grouping only matches when a full group of
# exactly three digits follows the comma.
_NUMBER_TOKEN = re.compile(r"[$£€¥]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|[$£€¥]?\d+(?:\.\d+)?%?")
_CURRENCY_SYMBOLS = "$£€¥"


def _normalise_number(raw: str) -> str:
    """Strip thousands separators, a trailing `%`, currency symbols, and trailing
    punctuation; keep the decimal point and its digits. Compared as strings, so
    "40" and "40.0" are deliberately distinct -- no reason yet to consider them
    the same number.
    """
    token = raw.strip().rstrip(".,;:!?")
    for symbol in _CURRENCY_SYMBOLS:
        token = token.replace(symbol, "")
    if token.endswith("%"):
        token = token[:-1]
    return token.replace(",", "")


def _numbers_in(text: str) -> set[str]:
    return {_normalise_number(m.group(0)) for m in _NUMBER_TOKEN.finditer(text)}


def _corpus_numbers(spans: Sequence[Span]) -> frozenset[str]:
    numbers: set[str] = set()
    for span in spans:
        numbers.update(_numbers_in(span.text))
    return frozenset(numbers)


# --- rule 2: boundary contact -------------------------------------------------

# Corpus-format convention, not a fact about any one user's corpus: matched
# case-insensitively as substrings of `section_path`, so different phrasing of
# the same heading ("boundaries to hold", "explicitly NOT true") both hit.
_BOUNDARY_SECTION_MARKERS = ("not true", "boundaries to hold")

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_MIN_TOKEN_LEN = 4

# Short and obvious -- common connective words long enough to pass the length
# filter but carrying no distinguishing content of their own.
_STOPWORDS = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "have",
        "they",
        "been",
        "were",
        "does",
        "such",
        "into",
        "than",
        "then",
        "some",
        "most",
        "much",
        "many",
        "about",
        "also",
        "only",
        "very",
        "just",
        "what",
        "when",
        "which",
        "while",
        "each",
        "these",
        "those",
        "there",
        "their",
        "would",
        "could",
        "should",
        "will",
        "your",
        "over",
        "here",
        "stated",
        "described",
        "confirmed",
    }
)

# A term used in more than this fraction of non-boundary spans is treated as
# corpus-wide vocabulary, not a distinctive boundary term. A guess, not a
# measured value -- tune against real behaviour once the rule has fired on
# enough real material to say whether it is too loose or too tight.
_DISTINCTIVE_FREQUENCY_CEILING = 0.2


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN_SPLIT.split(text.lower()) if len(t) >= _MIN_TOKEN_LEN)


def _is_boundary_span(span: Span) -> bool:
    if not span.section_path:
        return False
    path = span.section_path.lower()
    return any(marker in path for marker in _BOUNDARY_SECTION_MARKERS)


def _distinctive_boundary_terms(spans: Sequence[Span]) -> frozenset[str]:
    boundary_spans = [s for s in spans if _is_boundary_span(s)]
    if not boundary_spans:
        return frozenset()

    candidates: set[str] = set()
    for span in boundary_spans:
        candidates.update(_tokenize(span.text))
    candidates -= _STOPWORDS

    non_boundary_token_sets = [_tokenize(s.text) for s in spans if not _is_boundary_span(s)]
    if not non_boundary_token_sets:
        return frozenset(candidates)

    total = len(non_boundary_token_sets)
    return frozenset(
        term
        for term in candidates
        if sum(1 for tokens in non_boundary_token_sets if term in tokens) / total
        <= _DISTINCTIVE_FREQUENCY_CEILING
    )


# --- wiring --------------------------------------------------------------------


def _escalate(
    sentence: SentenceResult, corpus_numbers: frozenset[str], boundary_terms: frozenset[str]
) -> SentenceResult:
    flags: list[str] = []
    clauses: list[str] = []

    unsourced = sorted(n for n in _numbers_in(sentence.text) if n not in corpus_numbers)
    if unsourced:
        flags.append("unsourced-number")
        clauses.append(f"rule: number(s) not found in corpus ({', '.join(unsourced)})")

    matched_terms = sorted(boundary_terms & _tokenize(sentence.text))
    for term in matched_terms:
        flags.append(f"boundary-contact:{term}")
    if matched_terms:
        clauses.append(f"rule: touches boundary term(s) ({', '.join(matched_terms)})")

    if not flags:
        return sentence

    return sentence.model_copy(
        update={
            "verdict": "review",
            "rule_flags": [*sentence.rule_flags, *flags],
            "reason": f"{sentence.reason} {'; '.join(clauses)}".strip(),
        }
    )


def apply_rules(output: GateOutput, spans: Sequence[Span]) -> GateOutput:
    """Run the deterministic rule tier over an already-parsed `GateOutput`.

    Pure and side-effect free: no model call, no I/O, returns a new `GateOutput`
    rather than mutating `output` or anything reachable from it. Only claim
    sentences currently at `supported` are eligible to move -- see the module
    docstring for why that is the only direction a rule may ever move a verdict.
    """
    corpus_numbers = _corpus_numbers(spans)
    boundary_terms = _distinctive_boundary_terms(spans)

    sentences = [
        _escalate(sentence, corpus_numbers, boundary_terms)
        if sentence.kind == "claim" and sentence.verdict == "supported"
        else sentence
        for sentence in output.sentences
    ]
    return GateOutput(sentences=sentences)
