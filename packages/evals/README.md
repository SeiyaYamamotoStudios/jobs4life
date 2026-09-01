# jfl_evals

The Inspect (AISI) evaluation harness for the claim gate (`jfl_gate.gate.check_text`),
plus the tier-1 golden set (see `DATASET.md`). CLAUDE.md: "The Inspect harness is the
headline deliverable ... given a choice between more features and a working harness,
choose the harness."

No Postgres anywhere in this package -- `jfl_evals.repos` supplies in-memory
`GroundingRepository` / `RunRepository` implementations, and `jfl_evals.spans` builds
a tiny per-item corpus from each golden item's evidence sentences. The solver in
`jfl_evals.tasks` calls `check_text` directly: the real code path, not a
re-implementation, so what this measures is what a real `jfl check` run would do
against that corpus.

## Running it

Every item is one real, billed Anthropic call through the real gate. Read "Measured
cost" below before running more than the default handful.

```bash
# .env already carries JFL_DATABASE_URL and ANTHROPIC_API_KEY (see .env.example).
# Nothing here opens a database connection, but RequestContext.from_env() -- the
# one place this project reads the environment, see jfl_evals/tasks.py's module
# docstring -- requires JFL_DATABASE_URL to be *set*. If you have not copied
# .env.example to .env, export a dummy instead:
#   export JFL_DATABASE_URL=postgresql://unused/unused
set -a; source .env; set +a

uv run inspect eval packages/evals/src/jfl_evals/tasks.py --model none/none
```

`--model none/none` is required, not cosmetic: the solver never calls Inspect's own
`generate()`, so Inspect's model machinery is genuinely unused, and `none/none` says
so instead of silently defaulting to whatever provider happens to be configured.

This runs the task's own default: **5 items**, not the whole 210-item cache. That
default lives in `jfl_evals.tasks.claim_gate_fever`'s `limit` parameter (`DEFAULT_LIMIT
= 5`), not in Inspect's `--limit` CLI flag -- so a forgotten flag still only costs a
smoke test. To run more, pass it explicitly:

```bash
uv run inspect eval packages/evals/src/jfl_evals/tasks.py --model none/none -T limit=210
```

(210 is the whole cached dataset -- see `DATASET.md`.) `-T limit=<N>` for any N in
between works too. Inspect's own `--limit` flag can still be combined on top of this
(e.g. to stop a run partway through) but is not what keeps a bare invocation cheap.

Results land in `logs/*.eval` (Inspect's own log format -- CLAUDE.md: "Inspect eval
logs stay as Inspect's own files on disk, never in Postgres"). Open with
`uv run inspect view` or read the JSON directly out of the zip.

## Measured cost

Ran the default 5-item smoke test for real on 2026-09-01, against
`claude-opus-5`, and read the actual `RunRecord`s each sample's solver collected
back out of the resulting `.eval` log (`state.metadata["runs"]`):

| | |
|---|---|
| Items | 5 |
| Total cost | **$0.1058** |
| Mean cost / item | **$0.02116** |
| tokens in / out (total) | 198 / 1,530 |
| cache read / write tokens (total) | 0 / 10,649 |

**Extrapolated full run (210 items, at the same per-item rate): ~$4.44.** Cheap
enough to run in full when there's a reason to; still not something to run on every
`--limit`-less invocation by accident, which is the whole point of the small default.

One caveat worth being explicit about: **cache_read_tokens was 0 across all 5
samples.** Each golden item gets its own tiny synthetic corpus (its FEVER evidence
sentences), so every item's system prompt differs and every call is a fresh cache
write (billed at ~1.25x the input rate, per `jfl_gate.pricing`) rather than a cache
hit (~0.1x). In real usage, one user's whole corpus is cached once and reused across
every check against it, which is a materially cheaper per-check cost than what this
eval measures. The $0.02/item figure above is a fair estimate of *this eval's* cost,
not of a real `jfl check` call once a user's corpus is warm in cache.

## What the first real run already found

Worth reporting honestly rather than only as an abstract capability: on that same
5-item smoke test, one item -- "Dark matter is theoretical." (FEVER label
`NOT ENOUGH INFO`, expected verdict `review`, empty corpus) -- came back
`kind="framing"`, `verdict="supported"`. Framing is defined to always resolve to
"supported" (see `jfl_gate.prompt`), so this was simultaneously an over-claim (a
`review`-expected item marked `supported`) and the diagnostic `framing_rate` firing
(1/5 = 0.2 on this tiny sample). See `DATASET.md`'s "What this dataset cannot test"
for why a nonzero framing rate on FEVER data is a real, if narrow, mismatch between
FEVER's claim/evidence distinction and this gate's claim/framing distinction, not
necessarily a gate defect -- and why five items is nowhere near enough to say which.

This is not one failure mode among several: framing is the **only path through the
gate with no check anywhere in it**. A sentence classified `kind="framing"` is
forced to `verdict="supported"` by the prompt contract with no corpus comparison,
and the deterministic rule tier is explicitly forbidden from touching framing
sentences (`jfl_gate/rules.py`). So a framing misclassification that happens to
land on a claim that should not have passed is a *guaranteed* silent pass, not a
judgement call that happened to go the wrong way -- which is why
`framing_over_claim_rate`/`_count` (in `jfl_evals.scoring.ScoreSummary`, surfaced
through the `framing_rate` metric in `tasks.py`) is tracked as its own number
rather than something read off `framing_rate` and `over_claim_rate` by hand. On
this run: `framing_over_claim_count = 1`, `framing_over_claim_rate = 0.2` -- the
one framing item observed was also the one item that over-claimed.

The gate's `reason` for this call: *"A general statement about the world that
asserts nothing about the candidate's scope, ownership, or outcomes, so there is
nothing to trace to the corpus."* That phrasing -- reasoning in terms of
"candidate," "scope," "ownership" -- points at a **domain-fit mechanism**, not a
generic classifier weakness: the gate's framing prompt was iterated against CV
text (per CLAUDE.md, "near-default judgement, tailor from observed behaviour"),
where "nothing candidate-shaped here" and "this is ungroundable framing" happen to
coincide. "Dark matter is theoretical." breaks that coincidence -- it is a flat,
third-person factual claim with no candidate in it at all, and FEVER's own
annotators treated it as checkable (just under-evidenced), not as the kind of
motivation/sequence/what-was-being-weighed statement the gate's own definition of
framing describes. **Recommendation, not a change made here:** the prompt could be
clarified that a flat third-person factual assertion is a `claim` even when it says
nothing about the person, since framing is specifically about motivation, sequence,
and what was being weighed -- not "is there a candidate in this sentence." This is
recorded as a recommendation only; `packages/gate/src/jfl_gate/prompt.py` was not
touched by this work, since another agent was editing it concurrently and changing
judgement-shaping prose mid-measurement would have invalidated whatever it was
measuring.

## Layout

- `jfl_evals/dataset.py` -- `GoldenItem`, and the JSONL loader (`DatasetError` on a
  malformed line -- see its docstring for why that raises rather than skipping).
- `jfl_evals/spans.py` -- turns one golden item's evidence into `Span` objects
  (`provenance="document"`; see the module docstring for why not `"adjudicated"`).
- `jfl_evals/repos.py` -- in-memory `GroundingRepository` / `RunRepository`.
- `jfl_evals/costs.py` -- totals tokens/cost across a list of `RunRecord`s.
- `jfl_evals/scoring.py` -- the over-claim / over-flag arithmetic. Pure functions,
  no Inspect types -- read its module docstring before touching either rate.
- `jfl_evals/tasks.py` -- the Inspect `@task`, `@solver`, `@scorer`, and the four
  `@metric`s (`over_claim_rate`, `over_flag_rate`, `confusion_matrix`,
  `framing_rate` -- the last of which also carries the joint
  `framing_over_claim_rate`/`_count`), all thin adapters over `scoring.py`.
- `scripts/build_fever_dataset.py` -- the one-off dataset builder (not a runtime
  dependency of the package -- see its own docstring).
- `data/fever_slice.jsonl` -- the cached golden set itself (210 items).

## Tests

`uv run pytest packages/evals` -- no network, no database, no live model call.
Covers the scorer's arithmetic (including the degenerate "flags everything" /
"passes everything" / perfect-gate cases CLAUDE.md calls out), the dataset loader
(including a malformed line), the in-memory repositories, span construction, and
the task/metric wiring against hand-built data. The solver and scorer's real,
end-to-end behaviour (an actual `check_text` call) is exercised only by running the
eval itself, per "Running it" above -- never by the default test suite.

## Known limitations of this harness, stated plainly

- Tier 1 only. No tier 2 (owner-corpus, defensible-compression) items exist yet --
  optional per CLAUDE.md, not built in this pass.
- Only a 5-item real run has actually been observed end-to-end (see "What the first
  real run already found"). The over-claim / over-flag rates reported above are not
  a validated read on the gate's real-world performance -- five items is a pipeline
  smoke test, not a sample size. Running the fuller 210-item set is a deliberate,
  explicit next step (`-T limit=210`), not something this pass did on the project's
  paid credit balance without being asked. **n=5 cannot distinguish an anomaly from
  a rate** -- the single framing-induced over-claim observed could be a one-off or
  could recur at 20% on this dataset; nothing short of the fuller run tells them apart.
- `check_text`'s cache economics (see "Measured cost") mean this harness's total
  cost does not extrapolate to production per-check cost -- only to the cost of
  running this harness again.
- **The framing miss above may not transfer to production at the same rate it shows
  on FEVER.** The gate's framing definition was iterated against CV text, where
  "nothing candidate-shaped in this sentence" and "this is genuinely ungroundable
  framing" tend to coincide; FEVER is full of flat, topic-less, third-person trivia
  sentences that break that coincidence in a way a real CV rarely does (a CV bullet
  with no candidate in it at all is unusual; a Wikipedia-derived claim with no
  candidate in it is the norm). Tier 1 tests the core claim-vs-framing mechanic
  honestly, and the mechanism identified above (see "What the first real run
  already found") is real. But this dataset's `framing_over_claim_rate` should be
  read as a plausible **upper bound** on the production framing-over-claim rate,
  not as an estimate of it -- domain mismatch runs in the direction of FEVER
  triggering framing misclassifications a CV mostly would not.
