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

1. **Decide where an answered gap question lands.** CLAUDE.md says "corpus markdown is the
   source of truth; the database is a rebuildable index over it". `jfl answer` currently
   writes an `adjudicated` span straight to Postgres, so that invariant is false the
   moment anyone answers a question -- and the fact exists nowhere the author can read,
   edit or correct. Recommendation: append to `corpus/answered-questions.md` and
   re-ingest instead. It must be one or the other, never both: the same text ingested as
   a document span and written as an adjudicated span gets two different ids and the gate
   sees the fact twice.
2. **Slice 2b -- drafting behind the claim gate.** CV bullets, cover letters, free-text
   answers, both interaction modes. See the generation section in CLAUDE.md for the
   agreed shape. `packages/generate` already depends on nothing that blocks this, and
   `packages/cli` exists so `jfl_generate` can import `jfl_gate` without a cycle.
3. **Run the full eval** (210 items, ~$4.44) when there is credit. It is now the only way
   to measure the framing prompt change made at the end of this session, which is
   unverified and pushes in the direction that risks over-flagging.
4. **Sent-document store.** The 33 CVs belong in `sent_documents`/`sent_spans` for
   structure and consistency comparison. Never in grounding -- the tables deliberately
   share no FK path with the corpus.

Filling gaps in the author's own corpus is **not** on this list, deliberately. See the
decisions log: it produces no code and does by hand what slice 2a automates.

## Measurements worth not re-deriving

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
- **The two heuristic rules in the tier do not earn their place.** On real claims the
  unsourced-number rule fired 7/68 and escalated nothing; boundary-contact fired 35/68
  and escalated nothing. Across 32 CVs the boundary variants fire on 1.1%-54.7% of
  sentences depending on threshold, and the best-scoring unigram variant's top terms are
  employer names (`visa`, `ziglu`). Citation integrity, by contrast, is definitional and
  costs nothing: 0 hallucinated span ids and 0 uncited `supported` claims on that run.
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
