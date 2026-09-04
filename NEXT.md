# Where this got to

State at the end of the session on 2026-09-01. Design decisions live in CLAUDE.md;
this file is only "what is done, what is next, what is known to be wrong".

## Working, verified against real data

- **Corpus ingestion.** `corpus/*.md` to spans with stable ids. `uv run jfl ingest`.
- **The claim gate.** `uv run jfl check "text"` or `--file path.pdf`. Whole corpus in a
  cached system block, one streamed Opus call, per-sentence verdict with cited span ids.
- **Generation slice 2a.** `jfl job add|list|coverage|questions` and `jfl answer`. Paste a
  job ad, get requirements extracted, get a per-requirement corpus coverage verdict
  (`evidenced` / `partial` / `absent` / `contradicted`), get questions for the gaps.
  `jfl answer` stores your words verbatim -- there is no model anywhere in that path.
- **Inspect eval harness.** `packages/evals`, with a 210-item FEVER tier-1 golden set
  (balanced 70/70/70). Reports over-claim and over-flag as separate numbers, never one.
- **Instrumentation.** One `runs` row per model call: tokens, cache hits, cost, latency.
- 291 unit + 26 integration tests. `uv run mypy packages` is clean.

Everything above was exercised against the author's real corpus and real CVs, not only
fixtures. Every defect that mattered was found that way and none by review.

## Setup that is easy to forget

- `docker compose up -d` first; Postgres is on **5433**.
- `set -a; source .env; set +a` -- holds `JFL_DATABASE_URL` and `ANTHROPIC_API_KEY`.
  Never export the key globally: it silently shadows any `ant auth login` profile.
- API credit is separate from a Claude subscription and is **nearly exhausted** -- of the
  $5 added, roughly $3.50 is spent, so ~$1.50 remains. A gate run over a whole CV is
  ~$0.27 at steady state. Check with:
  `select component, stage, count(*), sum(cost_usd) from runs group by 1,2;`
  That query returns ~$1.18, which is a **lower bound**: measurement scripts and the eval
  harness use in-memory repositories and spend invisibly (see "known to be wrong").
- `corpus/` and `analysis/` are gitignored and hold real career detail. Back `corpus/` up
  somewhere private: the database is a rebuildable index over it, so losing the markdown
  loses the source.
- **Never run `git stash`, `git reset` or `git checkout --` while subagents are working.**
  It reverts every tracked file in the repo, not just one agent's. It happened this
  session and nearly destroyed four agents' uncommitted work; it was recovered with
  `git stash apply`. Copy files to a scratch directory instead.

## Next

**Tomorrow starts here: run C1.** `uv run python demo/generate_results.py --limit 1` --
one candidate x job combination through the real pipeline, ~$0.15-0.40, affordable on the
~$0.59 remaining. It replaces the guessed cost of the other eight with a measured one. The
script meters and resumes, so this is safe to run and stop.

Then, in order (costs carry the ~2x output-token spread, so treat them as ranges):

| # | Step | Cost | Blocked on |
|---|---|---|---|
| C1 | One demo combination, measured | ~$0.15-0.40 | nothing |
| C2 | Eval, 50 items on Opus, metered | ~$0.70-1.00 | top-up |
| C3 | Eval, remaining 160 on Opus | ~$2.30-3.00 | C2's per-item mean |
| C4 | Eval, 210 on Sonnet 5 (`JFL_MODEL=claude-sonnet-5`) | ~$0.80-1.20 | C3 |
| C5 | Choose the product model from the numbers; log the decision | -- | C4 |
| D1 | Remaining 8 demo combinations | ~$1.20-2.50 | C5, C1 |
| D2 | Build the static page from the result JSON (Jinja2 only, no `jfl_*` imports) | -- | D1 |
| D3 | Deploy to `job4life.hiltonlabs.org` via Cloudflare Pages | -- | D2 |

After that, `PLAN.md`'s W6 and the local-tool work: interactive gaps-first mode, span
validity periods, then the job queue and localhost web UI.

Deferred deliberately: the corpus-first prompt reorder (it changes prompt text, so it
needs an eval baseline to compare against -- do it after C3, then re-run to confirm the
number did not move).

## Measurements worth not re-deriving

- **The `reason` field name was tripping a safety classifier, and is now `evidence_note`.**
  Every gate call refused (`stop_reason: "refusal"`, category `reasoning_extraction`)
  as apparent reverse-engineering, on prompts that had worked the day before. Bisected
  against the live API 2026-09-02: system prompt alone fine, schema alone fine, together
  refuse; drop or rename the `reason` property and it clears. Independent of `effort` and
  `max_tokens`. Fixed in the gate (f87bf78) and coverage (f0fd4ec). **Do not rename it
  back**, and be wary of adding a `reason`-shaped property to any new schema.
- **The instructions block caches across every gate call, including between workloads.**
  Measured: eval item 1 wrote 2,245 tokens, items 2 and 3 read 2,245 with zero writes --
  and a later `jfl check` on a real CV also read the same 2,245-token entry. Per-eval-item
  cost $0.02116 -> $0.01795 measured over 3 items.
- **Output tokens vary far more than any of the per-call estimates assumed.** A full-CV
  gate run cost **$0.64**, not the ~$0.27 the earlier measurement implied: 21,143 output
  tokens against 9,546 on the earlier run of a comparable 86-sentence document. Eval items
  ranged 190 to 975 output tokens across three samples. Treat every projected total in
  PLAN.md as a lower bound with roughly 2x spread, and meter runs rather than trusting a
  budget computed from one sample.
- **The classifier is not scale-sensitive.** 84 sentences against the 236-span real corpus
  returned real verdicts (29 supported, 41 review, 3 unsupported, 11 framing), no refusal.

From one real CV (86 sentences, 68 claims, 18 framing) and a 32-CV sweep:

- **Output tokens are ~92% of a check's cost.** Corpus caching had already optimised the
  input side. Dropping the echoed sentence text for an index cut output tokens 27.9%
  (13230 -> 9546 on the same document at the same effort), ~21% cheaper at steady state.
  Controlled A/B, same CV, same effort, wire format the only variable.
- **`effort` was measured and left at the default `high`.** `low` saves only 15% of
  output tokens and `medium` 2%, against a risk that could not be ruled out: 2 of 85
  claims were misclassified as `framing` at `low` and 1 at `medium`. **Do not read that
  as proof effort causes it** -- the control run at `high` showed 2 as well, and with one
  document and one call per configuration, effort and sampling noise are not separable.
  The savings were too small to be worth the uncertainty, not the risk proven.
- **Changing `effort` invalidates the prompt cache.** All four runs showed `cache_read=0`
  and a full `cache_write`. So varying effort per call in production would destroy corpus
  caching, which costs more than any effort saving returns.
- **The two heuristic rules in the tier were measured and deleted** (see CLAUDE.md's
  decisions log and `rules.py`'s docstring). They escalated 16/9/14 claims per run from
  `supported` to `review` with `drift_label` still `supported` -- about 22% of claims,
  overriding a model that had traced them cleanly -- matching the owner's own name, a
  word from the boundary section's heading, and ordinary CV vocabulary. What replaced
  them is definitional: a cited span id exists or it does not; a `supported` claim cites
  something or it does not. Those fire zero times on real material, which is correct for
  a guarantee rather than a detector.
- **Do not replay `analysis/gate-runs/*.json` against a freshly parsed corpus.** The span
  ids in those files do not correspond to a fresh `parse_document` (0/65 matched under
  either `source_uri` prefix -- the measurement script used a different `user_id`, which
  is part of the span id). Any "fabricated citation" count derived that way is an
  artefact. Citation integrity was checked correctly in a live run that shared one span
  list between the repository and the check: 0 hallucinated ids, 0 uncited `supported`.
- **Numbers inside framing sentences: 0/14, 0/13, 0/12 across three runs.** A rule
  checking for them would never fire. Do not build it.
- **21 of 68 claims came back `supported`.** Corpus coverage, not gate accuracy, is the
  binding constraint -- and the gap-question write-back is the only thing that moves it.

## Known to be wrong

- **Framing is the only path with no check anywhere.** The prompt forces it to
  `verdict="supported"`, nothing compares it to the corpus, and the rule tier is
  forbidden from touching it, so a claim misfiled as framing is a guaranteed silent pass.
  Two mitigations landed this session -- the CLI now renders it as `NOT CHECKED` rather
  than `SUPPORTED`, and the prompt says a flat assertion about the world is a claim -- but
  neither is measured. The eval has a dedicated `framing x over-claim` counter for this.
- The gate cannot distinguish "the corpus contradicts this" from "the corpus is silent".
  The fix is the gap-question write-back path, not a prompt change.
- The record is a present-tense snapshot, so a historically accurate claim about an
  earlier team composition reads as drift. **Spans need validity periods.** This is the
  one piece of mechanism that came out of the gap-question work.
- A gate looking only for inflation misses half the distance: the CVs under-claim too.
- **`runs` under-reports spend.** Any script using an in-memory `RunRepository` -- as the
  measurement scripts and the eval harness both do -- spends real money invisibly. Treat
  the table as a lower bound. The test suite cannot spend (see the guard in
  `conftest.py`), but a throwaway script run by hand still can, and does.
- **The eval harness is not covered by the test guard.** It is run through
  `inspect eval`, not pytest, so nothing stops a full 210-item run costing ~$4.44. Its
  item limit defaults to 5; a full run has to be asked for explicitly with `-T limit=210`.
- Coverage and extraction match results to requirements positionally (`zip(strict=True)`),
  treating a count mismatch as a parse failure, rather than echoing an id back.
- `packages/evals` needs `datasets` to rebuild the FEVER slice; it is deliberately not a
  permanent dependency, so use `uv run --with datasets` for `scripts/build_fever_dataset.py`.
- Nothing writes to `span_embeddings`; retrieval is unused in v1 by decision.
- **If `uv run jfl ...` fails with "Failed to spawn: jfl", the venv is stale, not the
  config.** `packages/cli/pyproject.toml` declares the console script correctly and the
  installed `jfl_cli-0.1.0.dist-info/entry_points.txt` carries it, but `.venv/bin/jfl`
  can be missing -- a plain `uv sync` does not regenerate it. Fix with
  `uv sync --reinstall-package jfl-cli`. `uv run python -m jfl_cli.main <command>` also
  works and needs no reinstall.
