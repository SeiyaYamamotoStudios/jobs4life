"""Compare two Inspect eval logs of the same golden set, item by item.

    uv run python packages/evals/scripts/compare_eval_runs.py A.eval B.eval

Reads `.eval` logs straight off disk -- CLAUDE.md: "Inspect eval logs stay as
Inspect's own files on disk, never in Postgres" -- and reports, for each run and
then for the pair:

  * over-claim rate and over-flag rate, each with a Wilson 95% interval, never
    combined into one number (see `jfl_evals.scoring`'s module docstring for why);
  * framing rate and the framing x over-claim count, which is the one failure mode
    with no defence anywhere in the gate;
  * cost, read from the RunRecords each sample's solver collected;
  * a **paired** comparison over the items both runs scored.

The paired part is the reason this script exists. Compared as two independent
proportions, each rate on 210 items carries roughly +/-3.6pp and the difference
between them about +/-5.1pp, so only a large gap between two models is
resolvable. Run on identical items, the informative quantity is instead the
*discordant* pairs -- items where one run over-claimed and the other did not --
and McNemar's exact test on those has far more power for the same spend. Items
either run scored as an error are dropped from the paired comparison and counted,
never imputed.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import cast

from inspect_ai.log import read_eval_log


@dataclass(frozen=True)
class ItemOutcome:
    item_id: str
    expected: str
    actual: str | None  # None when the run errored on this item
    kind: str | None
    outcome: str | None


@dataclass(frozen=True)
class RunLog:
    path: Path
    model: str
    items: dict[str, ItemOutcome]
    cost_usd: Decimal
    n_calls: int
    tokens_in: int
    tokens_out: int
    cache_read: int
    cache_write: int


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval. Preferred over the normal approximation because the
    rates here are small and n is modest, exactly where the normal interval
    misbehaves (it can run below zero).
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: a binomial test on the discordant pairs
    against p=0.5. Exact rather than the chi-squared approximation because the
    discordant count here is expected to be small, which is where chi-squared is
    unreliable.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)
    return float(min(1.0, 2 * tail))


def load_run(path: Path) -> RunLog:
    """Read one `.eval` log through Inspect's own reader.

    Not stdlib `zipfile`: Inspect writes these with ZSTD (compression method 93),
    which `zipfile` cannot decompress before Python 3.14. `read_eval_log` also
    keeps this script honest about the log format if Inspect changes it.

    The model id is read off the RunRecords the solver collected, not from the
    log header: the task runs with `--model none/none` because the solver calls
    `check_text` directly, so Inspect never learns which model actually ran.
    """
    log = read_eval_log(str(path))

    items: dict[str, ItemOutcome] = {}
    cost = Decimal(0)
    n_calls = 0
    model = "unknown"
    tokens_in = tokens_out = cache_read = cache_write = 0

    for sample in log.samples or []:
        metadata = sample.metadata or {}
        for run in metadata.get("runs") or []:
            if run.get("cost_usd") is not None:
                cost += Decimal(str(run["cost_usd"]))
                n_calls += 1
            if model == "unknown" and run.get("model"):
                model = str(run["model"])
            tokens_in += run.get("tokens_in") or 0
            tokens_out += run.get("tokens_out") or 0
            cache_read += run.get("cache_read_tokens") or 0
            cache_write += run.get("cache_write_tokens") or 0

        score = (sample.scores or {}).get("gate_grounding_scorer")
        if score is None:
            continue
        value = score.value if isinstance(score.value, dict) else {}
        meta = score.metadata or {}
        item_id = str(sample.id)
        items[item_id] = ItemOutcome(
            item_id=item_id,
            expected=str(value.get("expected")),
            actual=meta.get("actual"),
            kind=meta.get("kind"),
            outcome=cast(str | None, value.get("outcome")),
        )

    return RunLog(
        path=path,
        model=model,
        items=items,
        cost_usd=cost,
        n_calls=n_calls,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_read=cache_read,
        cache_write=cache_write,
    )


@dataclass(frozen=True)
class Rates:
    n_scored: int
    n_errors: int
    over_claim: tuple[int, int]
    over_flag: tuple[int, int]
    silence_as_contradiction: tuple[int, int]
    framing: tuple[int, int]
    framing_over_claim: int


def _rates(run: RunLog) -> Rates:
    scored = [i for i in run.items.values() if i.actual is not None]
    errors = [i for i in run.items.values() if i.actual is None]

    # Denominators differ, deliberately: over-claim is only defined on items whose
    # ground truth is not "supported", over-flag only on items whose ground truth
    # is. Dividing either by the whole set would understate it.
    not_grounded = [i for i in scored if i.expected != "supported"]
    grounded = [i for i in scored if i.expected == "supported"]
    silent = [i for i in scored if i.expected == "review"]

    over_claims = [i for i in not_grounded if i.outcome == "over_claim"]
    over_flags = [i for i in grounded if i.outcome == "over_flag"]
    # "The corpus is silent" read as "the corpus contradicts this". CLAUDE.md names
    # this as a known limitation of the gate, and it is neither an over-claim nor an
    # over-flag, so neither headline rate sees it. It turned out to be the only
    # dimension on which two models measurably differed -- hence a number of its own.
    silence_as_contradiction = [i for i in silent if i.actual == "unsupported"]
    framing = [i for i in scored if i.kind == "framing"]
    framing_over_claim = [i for i in over_claims if i.kind == "framing"]

    return Rates(
        n_scored=len(scored),
        n_errors=len(errors),
        over_claim=(len(over_claims), len(not_grounded)),
        over_flag=(len(over_flags), len(grounded)),
        silence_as_contradiction=(len(silence_as_contradiction), len(silent)),
        framing=(len(framing), len(scored)),
        framing_over_claim=len(framing_over_claim),
    )


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _rate_line(label: str, pair: tuple[int, int]) -> str:
    n, d = pair
    p, lo, hi = _wilson(n, d)
    return f"  {label:<30} {_pct(p):>6}  ({n}/{d})   95% CI {_pct(lo)}-{_pct(hi)}"


def report_one(run: RunLog) -> None:
    r = _rates(run)
    print(f"\n=== {run.model}  ({run.path.name}) ===")
    print(f"  scored {r.n_scored}, harness errors {r.n_errors}")
    print(_rate_line("over-claim rate", r.over_claim))
    print(_rate_line("over-flag rate", r.over_flag))
    print(_rate_line("silence read as contradiction", r.silence_as_contradiction))
    print(
        f"  {'framing':<30} {r.framing[0]}/{r.framing[1]}   "
        f"of which over-claimed: {r.framing_over_claim}"
    )
    print(
        f"  {'cost':<30} ${run.cost_usd:.4f} over {run.n_calls} calls  "
        f"(in {run.tokens_in:,} / out {run.tokens_out:,} / "
        f"cache r {run.cache_read:,} w {run.cache_write:,})"
    )


def report_paired(a: RunLog, b: RunLog) -> None:
    shared = sorted(set(a.items) & set(b.items))
    usable = [k for k in shared if a.items[k].actual is not None and b.items[k].actual is not None]
    dropped = len(shared) - len(usable)

    print(f"\n=== paired: {a.model} vs {b.model} ===")
    print(
        f"  items in both logs: {len(shared)}; usable: {len(usable)}; dropped for errors: {dropped}"
    )

    agree = sum(1 for k in usable if a.items[k].actual == b.items[k].actual)
    print(
        f"  verdict agreement:  {agree}/{len(usable)} ({_pct(agree / len(usable))})"
        if usable
        else "  verdict agreement: n/a"
    )

    dimensions: tuple[tuple[str, Callable[[ItemOutcome], bool], list[str]], ...] = (
        (
            "over-claim",
            lambda i: i.outcome == "over_claim",
            [k for k in usable if a.items[k].expected != "supported"],
        ),
        (
            "over-flag",
            lambda i: i.outcome == "over_flag",
            [k for k in usable if a.items[k].expected == "supported"],
        ),
        (
            "silence->contra",
            lambda i: i.actual == "unsupported",
            [k for k in usable if a.items[k].expected == "review"],
        ),
    )
    for label, predicate, subset in dimensions:
        only_a = [k for k in subset if predicate(a.items[k]) and not predicate(b.items[k])]
        only_b = [k for k in subset if predicate(b.items[k]) and not predicate(a.items[k])]
        p = _mcnemar_exact(len(only_a), len(only_b))
        print(
            f"  {label:<11} discordant: {a.model} only {len(only_a)}, "
            f"{b.model} only {len(only_b)}  ->  McNemar exact p = {p:.3f}"
            + ("  (no detectable difference)" if p > 0.05 else "  (difference)")
        )
        for k in only_a[:5]:
            print(f"      only {a.model}: {k}  expected={a.items[k].expected}")
        for k in only_b[:5]:
            print(f"      only {b.model}: {k}  expected={b.items[k].expected}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="One or two .eval logs.")
    args = parser.parse_args(argv)

    runs = [load_run(p) for p in args.logs]
    for run in runs:
        report_one(run)
    if len(runs) == 2:
        report_paired(runs[0], runs[1])
    elif len(runs) > 2:
        print("\n(paired comparison only runs for exactly two logs)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
