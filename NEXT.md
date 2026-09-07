# Where this got to

State at the end of the session on 2026-09-05. Design decisions live in CLAUDE.md;
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
- 371 unit + 31 integration tests. `uv run mypy packages` is clean across 63 files.

Everything above was exercised against the author's real corpus and real CVs, not only
fixtures. Every defect that mattered was found that way and none by review.

## Setup that is easy to forget

- `docker compose up -d` first; Postgres is on **5433**.
- `set -a; source .env; set +a` -- holds `JFL_DATABASE_URL` and `ANTHROPIC_API_KEY`.
  Never export the key globally: it silently shadows any `ant auth login` profile.
- API credit is separate from a Claude subscription. $10 was topped up on 2026-09-05;
  **$7.26 of it was spent that day** on C1-C4 and D1, leaving roughly **$3.33**. Nothing
  remaining in the September plan costs anything -- D3 is free, and `build_site.py` makes
  no API calls. A gate run over a whole CV is ~$0.35-0.64; a full 210-item
  eval is ~$2.07. Check with:
  `select component, stage, count(*), sum(cost_usd) from runs group by 1,2;`
  That query is a **lower bound**: the eval harness uses in-memory repositories and
  spends invisibly (see "known to be wrong"). Read the real total from the Anthropic
  console, never from Postgres.
- `corpus/` and `analysis/` are gitignored and hold real career detail. Back `corpus/` up
  somewhere private: the database is a rebuildable index over it, so losing the markdown
  loses the source.
- **Never run `git stash`, `git reset` or `git checkout --` while subagents are working.**
  It reverts every tracked file in the repo, not just one agent's. It happened this
  session and nearly destroyed four agents' uncommitted work; it was recovered with
  `git stash apply`. Copy files to a scratch directory instead.

## Next

**C0-C5, D1 and D2 are done (2026-09-05). Only D3 is left, and it is blocked on one
thing that is not the code's to do.**

The nine demo results exist as real pipeline output, and `demo/site/index.html` is
built, committed, and verified self-contained: 0 external references, 0 fetch/XHR/module
uses, screenshotted at 1280px and 390px. Rebuild it any time with:

```bash
uv run python demo/build_site.py        # reads demo/fixtures/results/*.json, no API calls
```

**D3 -- publish to `job4life.hiltonlabs.org`.** Two routes; pick one, then the rest is
a single command.

| | A: connect GitHub | B: direct upload (Wrangler) |
|---|---|---|
| What the owner grants | Cloudflare read access to the whole private repo | one API token, Pages:Edit scope |
| Deploys when | every push to the branch | when `wrangler pages deploy` is run |
| Build step | none (output dir `demo/site`) | none |

**B is the better fit** and supersedes the GitHub-connection assumption in CLAUDE.md's
2026-09-04 entry. That entry chose a build output directory because it assumed Cloudflare
would clone and build; since `demo/site/` is committed and there is no build command,
the git connection buys only auto-deploy -- and pays for it by granting a third party
read access to a private repo holding a career system. A demo page that changes monthly
does not need to republish on every push. Route A stays available if auto-deploy ever
matters more than the access.

Route B, once `CLOUDFLARE_API_TOKEN` is in `.env`:

```bash
npx wrangler pages deploy demo/site --project-name=job4life
```

Then point `job4life.hiltonlabs.org` at the Pages project in the Cloudflare dashboard.
Leave the apex A record `178.128.137.126` alone -- it is Ghost's shared redirect server.

**Open, and the owner's call:** the gate flags the draft's own *title line* (e.g. "Ingrid
Solberg -- CV bullets (Senior Backend Engineer ...)") as `unsupported` /
`adjacency_substitution`. It is a header naming the role applied for, not a claim to hold
it, so this is document structure being read as assertion -- the same class as the
`George R.R.` re-split. It is currently the most prominent item on the demo page.
Leaving it is honest (the page says the output is unedited, and it is); excluding title
lines from gating is a pipeline change that costs a regeneration. Recommendation: leave
it for September, record it.

After D3: `PLAN.md`'s W6 and the local-tool work -- interactive gaps-first mode, span
validity periods, then the job queue and localhost web UI.

Deferred deliberately: the corpus-first prompt reorder. It changes prompt text, and
there is now a real baseline to compare against (the 2026-09-05 Opus log), so this is
finally cheap to evaluate honestly -- re-run and confirm the number did not move, ~$2.07.

**The headline numbers, from the full 210-item tier-1 run:**

| | Opus 5 | Sonnet 5 |
|---|---|---|
| over-claim rate | **0.7%** (1/140), CI 0.1-3.9% | **0.7%** (1/140), CI 0.1-3.9% |
| over-flag rate | **2.9%** (2/69), CI 0.8-10.0% | **2.9%** (2/69), CI 0.8-10.0% |
| silence read as contradiction | 22.9% (16/70) | **71.4%** (50/70) |
| framing | 0/209 | 0/209 |
| cost | $2.0715 | $1.8451 |

Both models over-claim and over-flag on the *same items* -- zero discordant pairs,
McNemar p = 1.000. They differ 34-0 on silence-as-contradiction, p < 0.001. Opus is
the default; see CLAUDE.md's 2026-09-05 entry for why, and note the reason is cost
and temperament, not accuracy.

**Caveat to state whenever the number is quoted:** framing was 0/209, so tier 1 did
not exercise the gate's one unguarded path at all. FEVER items are factual assertions
by construction, so this is the dataset's shape, not a clean bill of health. The
framing hole remains unmeasured until tier 2 exists.

After D3: `PLAN.md`'s W6 and the local-tool work -- interactive gaps-first mode, span
validity periods, then the job queue and localhost web UI.

Deferred deliberately: the corpus-first prompt reorder. It changes prompt text, and
there is now a real baseline to compare against (the 2026-09-05 Opus log), so this is
finally cheap to evaluate honestly -- re-run and confirm the number did not move,
~$2.07.

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
- **The model re-splits its own input, and it used to abort the whole eval.** On
  `fever-13515`, "Petyr Baelish is created by an American author George R.R. Martin."
  came back as *two* claims -- "...George R.R." and a fragment "Martin.". `split_blocks`
  is not at fault: it returns one block for that text, verified locally. The model split
  at the initials inside its own structured output. Any name with initials, or a
  "Ph.D.", can do this, so it will happen on real CVs too. The eval scorer used to raise
  on it, which killed a 50-item run at sample 45; it now scores as a harness error,
  excluded from both headline rates rather than folded into either. **The underlying gate
  behaviour is unfixed** -- fixing it means a prompt change, which invalidates the
  2026-09-05 baseline, so it waits.
- **A 5-item eval sample overstates the per-item cost by ~2x.** The smoke test measured
  $0.02116/item; the full 210-item run came in at $0.00986. The small sample pays a cache
  write for the shared instructions block that the remaining items then read for free.
  Extrapolate from a full run or not at all.
- **Sonnet 5 is 11% cheaper than Opus 5 here, not 60%.** Measured over the same 210
  items: $1.8451 vs $2.0715, because Sonnet emitted **157,162 output tokens against
  Opus's 68,420** (2.3x) and output is ~92% of the cost. The rate card is not a guide to
  this workload's cost.
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
