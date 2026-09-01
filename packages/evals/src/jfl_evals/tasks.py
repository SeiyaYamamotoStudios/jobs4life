"""The Inspect (AISI) evaluation task for the claim gate, against the tier-1
FEVER-derived golden set. See `packages/evals/README.md` for how to actually run
this -- including the measured per-item cost, read that before raising the
default `limit` -- and `packages/evals/DATASET.md` for the dataset itself.

The solver below calls `jfl_gate.gate.check_text` directly: the real code path,
not a re-implementation of it, against the in-memory repositories in
`jfl_evals.repos` so no Postgres is needed anywhere (CLAUDE.md: "Inspect eval logs
stay as Inspect's own files on disk, never in Postgres"). `check_text` makes its
own Anthropic call and Inspect's own model machinery is never used, so this must
be run with `--model none/none` -- see README.md.

Credentials: `RequestContext.from_env()` is CLAUDE.md's one sanctioned place to
read the environment ("The API key is per-request context passed from the entry
point -- never a module-level environment read"), so this solver calls it rather
than reading `ANTHROPIC_API_KEY` itself. `from_env()` also requires
`JFL_DATABASE_URL` to be set even though nothing here touches a database -- rather
than add a second environment-reading path to work around that, README.md
documents exporting a dummy value before running this eval. That keeps
`from_env()` the single entry point CLAUDE.md asks for, at the cost of one
harmless-but-required env var for a command that never opens a connection.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, Value, metric, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from jfl_core.context import RequestContext
from jfl_gate.gate import GateError, check_text
from jfl_gate.schema import GateOutput

from jfl_evals.dataset import GoldenItem, load_golden_set
from jfl_evals.repos import InMemoryGroundingRepository, InMemoryRunRepository
from jfl_evals.scoring import ItemResult, Verdict, aggregate, is_over_claim, is_over_flag
from jfl_evals.spans import build_spans

# A smoke test, not a full run -- see README.md's cost section for why. Passing
# `-T limit=<N>` (up to the dataset's size; DATASET.md records how many items are
# cached) is a deliberate act, not the default.
DEFAULT_LIMIT = 5

DEFAULT_DATASET_PATH = str(Path(__file__).resolve().parents[2] / "data" / "fever_slice.jsonl")


def _sample_for(item: GoldenItem) -> Sample:
    return Sample(
        input=item.claim,
        target=item.expected_verdict,
        id=item.id,
        metadata={"item": item.model_dump(mode="json")},
    )


@solver
def run_claim_gate() -> Solver:
    """Runs `check_text` for real, once per sample, against an in-memory corpus
    built from that item's FEVER evidence sentences (`jfl_evals.spans.build_spans`).

    `check_text` is synchronous and makes a blocking network call; it is pushed to
    a worker thread with `asyncio.to_thread` so Inspect can still run samples
    concurrently instead of serializing every call behind one event loop turn.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        item = GoldenItem.model_validate(state.metadata["item"])
        ctx = RequestContext.from_env()
        spans = build_spans(item, ctx.user_id)
        grounding_repo = InMemoryGroundingRepository(spans)
        run_repo = InMemoryRunRepository()

        result: GateOutput | None
        error: str | None
        try:
            result = await asyncio.to_thread(check_text, ctx, grounding_repo, run_repo, item.claim)
            error = None
        except GateError as e:
            result = None
            error = str(e)

        state.metadata["gate_result"] = result.model_dump(mode="json") if result else None
        state.metadata["gate_error"] = error
        # RunRecords for this one call -- almost always exactly one, `check_text`
        # writes exactly one `runs` row per invocation, success or failure alike.
        state.metadata["runs"] = [r.model_dump(mode="json") for r in run_repo.records]
        return state

    return solve


def _item_results(scores: list[SampleScore]) -> list[ItemResult]:
    """Rebuilds the pure `ItemResult` list `jfl_evals.scoring.aggregate` needs from
    the metadata `gate_grounding_scorer` attached to each `Score`. Every metric
    below goes through this and then `aggregate` -- the arithmetic lives in
    exactly one place, tested without Inspect in `tests/test_scoring.py`.
    """
    results = []
    for s in scores:
        md = s.score.metadata or {}
        results.append(
            ItemResult(
                item_id=str(s.sample_id),
                expected=md["expected"],
                actual=md.get("actual"),
                kind=md.get("kind"),
                error=md.get("error"),
            )
        )
    return results


@metric
def over_claim_rate() -> Metric:
    """The headline metric. See `jfl_evals.scoring` for why it is never reported alone."""

    def calc(scores: list[SampleScore]) -> Value:
        s = aggregate(_item_results(scores))
        return {
            "rate": s.over_claim_rate if s.over_claim_rate is not None else 0.0,
            "count": s.over_claim_count,
            "denominator": s.over_claim_denominator,
        }

    return calc


@metric
def over_flag_rate() -> Metric:
    """The headline metric's mandatory partner. See `jfl_evals.scoring`."""

    def calc(scores: list[SampleScore]) -> Value:
        s = aggregate(_item_results(scores))
        return {
            "rate": s.over_flag_rate if s.over_flag_rate is not None else 0.0,
            "count": s.over_flag_count,
            "denominator": s.over_flag_denominator,
        }

    return calc


@metric
def confusion_matrix() -> Metric:
    """The raw expected-verdict x actual-verdict counts, flattened to
    "expected->actual" keys (`Score.value` cannot hold a nested mapping). Grouping
    these keys by their expected half is the per-expected-label breakdown.
    """

    def calc(scores: list[SampleScore]) -> Value:
        s = aggregate(_item_results(scores))
        out: dict[str, Any] = {
            f"{expected}->{actual}": count for (expected, actual), count in s.confusion.items()
        }
        out["n_items"] = s.n_items
        out["n_scored"] = s.n_scored
        out["n_errors"] = s.n_errors
        return out

    return calc


@metric
def framing_rate() -> Metric:
    """Diagnostic only -- never a headline number. FEVER claims are all factual
    assertions by construction, so a nonzero rate here means the gate is
    mis-classifying claim sentences as framing, not that it correctly spotted any.

    Carries `framing_over_claim_rate`/`_count` alongside the plain framing rate,
    not folded into either headline metric: framing is the one path in the gate
    with no check anywhere in it (see `jfl_evals.scoring`'s module docstring for
    the full argument), so how often a framing misclassification specifically
    causes an over-claim is its own question, worth its own number, separate from
    "how often does the gate call something framing" and from the general
    `over_claim_rate`.
    """

    def calc(scores: list[SampleScore]) -> Value:
        s = aggregate(_item_results(scores))
        return {
            "rate": s.framing_rate if s.framing_rate is not None else 0.0,
            "count": s.framing_count,
            "over_claim_rate": (
                s.framing_over_claim_rate if s.framing_over_claim_rate is not None else 0.0
            ),
            "over_claim_count": s.framing_over_claim_count,
        }

    return calc


@scorer(metrics=[over_claim_rate(), over_flag_rate(), confusion_matrix(), framing_rate()])
def gate_grounding_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # target.text is plain str -- it round-trips a GoldenItem.expected_verdict
        # via Sample.target (jfl_evals.tasks._sample_for), so this cast just
        # reasserts what the sample construction already guarantees.
        expected = cast(Verdict, target.text)
        gate_result = state.metadata.get("gate_result")
        error = state.metadata.get("gate_error")

        if gate_result is None:
            return Score(
                value={"expected": expected, "actual": "error", "kind": "error"},
                explanation=error or "gate call failed",
                metadata={"expected": expected, "actual": None, "kind": None, "error": error},
            )

        sentences = gate_result["sentences"]
        # Every golden item is exactly one FEVER claim, so `check_text` (given
        # exactly one sentence) returns exactly one SentenceResult. If that is
        # ever not true it is a data problem worth seeing loudly, not a case to
        # silently pick sentences[0] for.
        if len(sentences) != 1:
            raise ValueError(
                f"sample {state.sample_id}: expected exactly one sentence result, "
                f"got {len(sentences)}"
            )
        sentence = sentences[0]
        actual = sentence["verdict"]
        kind = sentence["kind"]

        if is_over_claim(expected, actual):
            outcome = "over_claim"
        elif is_over_flag(expected, actual):
            outcome = "over_flag"
        elif actual == expected:
            outcome = "match"
        else:
            outcome = "other_mismatch"  # e.g. expected "review", gate said "unsupported"

        return Score(
            value={"expected": expected, "actual": actual, "kind": kind, "outcome": outcome},
            answer=actual,
            explanation=sentence.get("reason"),
            metadata={
                "expected": expected,
                "actual": actual,
                "kind": kind,
                "drift_label": sentence.get("drift_label"),
                "error": None,
            },
        )

    return score


@task
def claim_gate_fever(
    dataset_path: str = DEFAULT_DATASET_PATH,
    limit: int = DEFAULT_LIMIT,
) -> Task:
    """The tier-1 claim gate eval, against the FEVER-derived golden set.

    `limit` (default 5) truncates the *loaded* dataset before Inspect ever sees
    it -- deliberately, so omitting `--limit` on the CLI still only spends money
    on a handful of items. A full run over the whole cached set is a deliberate
    act: pass `-T limit=<N>` up to the dataset's size (DATASET.md records how many
    items are cached). Inspect's own `--limit` flag still works on top of this but
    is not what keeps a forgotten invocation cheap -- this default is.

    Run with `--model none/none`: the solver calls `jfl_gate.gate.check_text`
    directly and never asks Inspect to generate anything, so Inspect's model
    machinery is unused. See README.md for the exact command and the measured
    cost of the default 5-item run.
    """
    items = load_golden_set(Path(dataset_path))
    samples = [_sample_for(item) for item in items[:limit]]
    return Task(
        dataset=MemoryDataset(samples=samples, name="jfl-claim-gate-fever-tier1"),
        solver=run_claim_gate(),
        scorer=gate_grounding_scorer(),
        model="none/none",
    )
