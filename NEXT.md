# Where this got to

State at the end of the session on 2026-09-01. Design decisions live in CLAUDE.md;
this file is only "what is done, what is next, what is known to be wrong".

## Working, verified against real data

- **Corpus ingestion.** `corpus/*.md` to spans with stable ids. Run `uv run jfl ingest`.
- **The claim gate.** `uv run jfl check "text"` or `--file path.pdf`. Whole corpus in a
  cached system block, one Opus call, per-sentence verdict with cited span ids.
- **Instrumentation.** One `runs` row per invocation: tokens, cache hits, cost, latency.
- 128 unit + 13 integration tests. `uv run pytest` and `uv run pytest -m integration`.

Verified on the author's real corpus and real CVs, not only fixtures. The gate correctly
caught claims contradicting the corpus's stated boundaries, and left narrative framing
alone — the behaviour that decides whether the tool stays switched on.

## Setup that is easy to forget

- `docker compose up -d` first; Postgres is on **5433**.
- `set -a; source .env; set +a` — holds `JFL_DATABASE_URL` and `ANTHROPIC_API_KEY`.
  Never export the key globally: it silently shadows any `ant auth login` profile.
- API credit is separate from a Claude subscription and is **limited**. A gate run over a
  whole CV is roughly $0.13. Check spend with:
  `select count(*), sum(cost_usd) from runs where outcome='ok';`
- `corpus/` is gitignored and holds real career detail. Back it up somewhere private —
  the database is a rebuildable index over it, so losing the markdown loses the source.

## Next

1. **Read the drift analysis** in `analysis/` (gitignored — it quotes real career detail).
   Produced by a subagent, not by the claim gate, so its numbers are not the product's
   numbers. Its headline was independently checked and found overstated; the correction is
   at the top of the file. Its findings are the evidence for the deterministic rule set.
   Two things in it need the author, not code: whether the Visa team included Poland
   before Atlanta, and how much of the commission reporting system he personally wrote.
2. **Add the record gaps** the analysis found — the commission reporting system is the
   biggest. Corpus coverage, not gate accuracy, is currently the binding constraint: the
   gate correctly returns "unsupported" on the CVs' strongest material because the record
   is silent on it.
3. **Deterministic rule tier**, built from what the analysis actually found. Note that
   `invented_quantity` never fired across 2,347 units — every number traced. The load is
   carried by ownership_inflation, scope_inflation and outcome_attribution.
4. **Generation** — see the generation section in CLAUDE.md for the agreed shape.

## Known to be wrong

- The gate cannot distinguish "the corpus contradicts this" from "the corpus is silent on
  this". A true but unrecorded fact reads as an over-claim. The fix is the gap-question
  write-back path, not a prompt change. The analysis makes this concrete: seven CVs' most
  detailed technical claim is unadjudicable because the record does not cover it.
- The record is a present-tense snapshot of a role running since Nov 2024, so any
  historically-accurate claim about an earlier team composition reads as drift. It needs
  time-boxed entries.
- A gate looking only for inflation misses half the distance: the CVs under-claim too.
- PDF extraction rejoins a hyphen-wrapped word with a stray space around the hyphen.
- A wrapped line ending in a bare four-digit number is read as a title boundary. Accepted:
  over-splitting is far cheaper than fusing unrelated claims into one verdict.
- `mypy packages` reports 2 pre-existing errors, both predating the gate work.
- Nothing writes to `span_embeddings`; retrieval is unused in v1 by decision.
