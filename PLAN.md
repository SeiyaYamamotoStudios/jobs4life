# PLAN — jobs4life as a tool the owner actually uses

Written 2026-09-07, replacing the September demo plan (archived at
`docs/PLAN-september-demo.md`, delivered in full). Decisions behind this plan are in
CLAUDE.md's decisions log, 2026-09-07 entries; this file is sequencing, designs, fences
and acceptance criteria only.

**There is no deadline and that is a risk, not a licence.** The September deliverable is
shipped. The bar from here is not "demonstrates the idea" — the owner already uses this
daily across scattered Claude conversations and finds it useful. The bar is **"replaces
that"**, and the thing conversations structurally cannot do is remember. In his words:
*"a clear list of all the applications I have going."*

So the ordering principle throughout: **the tracker comes before the cleverness.** Every
slice below must be usable on its own the day it lands. If a slice would only make sense
once the next one exists, it is sliced wrong.

## The four slices

| | What lands | Usable alone as |
|---|---|---|
| **A** | Google login, enforced tenancy, per-user API key custody, application tracker | The list of live applications, with real timestamps |
| **B** | Queue, quick-actions, paste-an-ad, two scores, the engine behind a UI | Paste an ad and get it scored, drafted and gated without a terminal |
| **C** | Intake: ATS APIs, forwarding address, scored on arrival | Roles arriving without being hunted for |
| **D** | Interview prep, and rejection feedback into a plan | A closed loop from rejection to citable evidence |

---

## Slice A — the shell that remembers

Nothing model-facing. No Anthropic call anywhere in this slice except one optional
validation ping. That is deliberate: it means A can be built, deployed and used while
costing nothing, and it isolates auth bugs from engine bugs.

### A1 — Google login

Authorization Code flow with PKCE via **Authlib**. Scopes `openid email profile`, nothing
else — no Gmail scope, ever (see the forwarding-address decision).

**Identify users by Google's `sub`, never by email.** Email addresses change hands;
`sub` is stable and unique forever. Store email and name for display only, refreshed on
each login.

### A2 — Sessions

Opaque session id in a cookie, session row in Postgres. **Not a JWT** — this app holds
other people's API keys, so instant server-side revocation matters more than statelessness
at one-VPS scale.

Cookie: `__Host-` prefix, `HttpOnly`, `Secure`, `SameSite=Lax`, no `Domain` attribute.
Row carries `user_id`, `created_at`, `expires_at`, `last_seen_at`, and a rolling
expiry. Logging out deletes the row, not just the cookie.

### A3 — Tenancy, enforced structurally

Per CLAUDE.md: a repository is **constructed with** the `user_id` it may act on and has
no per-call override. There must be no code path where forgetting a `WHERE user_id = ...`
is expressible.

**Acceptance is a test, not a review:** a test that reflects over every public repository
method and asserts none of them accepts a `user_id` argument. A `WHERE` clause a reviewer
has to spot is not enforcement, and this is the failure that leaks one user's career
history to another.

### A4 — Credential custody

Envelope encryption, per the non-negotiable standing rule.

- **KEK** — 32 bytes, base64, from `JFL_MASTER_KEY` in the host environment. Never in
  Postgres. A database compromise alone therefore yields nothing usable.
- **DEK** — 32 random bytes per user, AES-GCM-encrypted under the KEK.
- **The API key** — AES-GCM-encrypted under that user's DEK.
- Table `user_credentials`: `user_id`, `provider`, `wrapped_dek`, `dek_nonce`,
  `ciphertext`, `nonce`, `key_hint` (last 4 characters, for display), `created_at`,
  `last_used_at`.

**Write-only from the browser's side.** A key can be set or replaced, never read back —
the UI shows `sk-ant-…4f2a` and a Replace button. Never logged, never in `runs`, never in
a trace, never in an error message.

Validate on entry with one cheap Anthropic call so a typo fails at the form rather than
three screens later.

### A5 — The application tracker

The wedge. Two tables:

- `applications` — `user_id`, `job_id`, `status`, `source`, `notes`, timestamps
- `application_events` — every status transition, with a timestamp and optional note

Statuses: `interested` → `applied` → `screening` → `interviewing` → `offer`, plus
terminal `rejected` and `withdrawn`. Transitions are recorded, never overwritten: the
event log *is* the timeline, and it is what later gets injected into model context.

Job entry already works — `jfl job add` parses a pasted ad today. Slice A puts a textarea
in front of it.

### A6 — Timestamps everywhere

Per the 2026-09-07 decision. Current time plus the user's live application timeline go
into model context on every call from slice B onwards; in slice A it is UI only — every
event shows an absolute date *and* a relative one ("Tue 8 Sep, 3 days ago"), because
"tomorrow" is exactly what the owner said gets lost.

### Slice A acceptance

1. Two Google accounts, side by side: neither can see or reach the other's applications
   by any URL, including guessed ids.
2. The repository-method reflection test passes.
3. An API key can be set and replaced, never read back, and appears nowhere in logs,
   `runs`, or a traceback.
4. Deployed on the VPS behind the Cloudflare Tunnel, reachable over HTTPS, with the
   demo page at `jobs4life.hiltonlabs.org` still serving and untouched.
5. The owner can add a real job ad and move it through states — and prefers doing that
   to keeping the list in a conversation. **If he doesn't, slice A has failed** and no
   amount of slice B fixes it.

---

## Slice B — the engine, behind a UI

Revised 2026-09-08 after the owner used slice A. His feedback drives the ordering
below; everything except B2 needs the queue, which is why the queue is first.

### B1 — the job queue, first

A gate call takes ~2 minutes and an extraction is not much quicker. Nothing that slow
runs inside a request. His words set the requirement exactly: **fast input, slow
processing is acceptable** — so the form returns immediately and the work happens behind
it.

`tasks` table with `SELECT … FOR UPDATE SKIP LOCKED`, an attempt counter, a max-attempts
cap, and a global `JFL_DISABLE_MODEL_CALLS` kill switch. The worker is a separate
container so it can be stopped without taking the site down — which is also the incident
response if a user's key starts burning money. It also gets the jobs nothing else owns:
purging expired sessions, and later the demo regeneration.

### B2 — status quick-actions (no queue, do it early)

A dropdown plus submit is wrong for the 90% case. A primary button for the **likely next
state** — "Mark as applied", "Mark as screening" — with the full picker still present for
jumps and reversals. Cheap, and the most-used control in the app.

### B3 — paste an ad, get an application

Replaces the current long form. A paste box, an optional URL, and nothing else required.
The extraction engine already exists (`jfl_generate.extract`, exercised today by
`jfl job add`) so this is a form, a task, and a results view.

**Paste is primary and the URL is best-effort.** Most ATS pages render client-side and
many refuse non-browser fetches, and LinkedIn and Indeed are excluded by standing
decision — not only legally, but because a scraper in the architecture reads as poor
judgement. So: try the URL, fall back quietly, never pretend it is reliable.

This is also the slice that proves the queue end to end, which is why it comes first
among the model-facing work.

### B4 — two scores on arrival, never one

On add, the application is scored automatically, with a paragraph for each score.

**Two axes, never composited** — the standing decision, and the owner's three requests map
onto it cleanly: *chances* is **could I get this**, *alignment to interest* is **do I want
this**, and *appropriateness* is a blend of the two, which is exactly what must not be
built. A role he would love and will not get, and one he would hate and would walk into,
must never land on the same number; averaging them hides the disagreement precisely when
it is the useful signal.

Ships **labelled unmeasured**. There is no golden set for fit and inventing one would be
the synthetic-data prohibition in a new coat. The over-claim rate remains the measured
number and must not be confused with these.

### B5 — the rest of the engine's screens

Coverage, gap questions, drafting ("Generate a CV" from an application), and the claim
gate with per-sentence verdicts. All built; all needing a UI and the queue.

**Framing renders as `NOT CHECKED`, never as supported** — the same rule the demo page
obeys. Per-run cost is shown, because the user is paying for it with their own key.

### B6 — corpus upload

A user with no corpus has nothing to measure against, so this gates B's usefulness for
anyone but the owner. Markdown upload through the existing ingestion, spans stored
per-user. "Markdown is the source of truth" holds: uploads are stored and re-ingestible,
never only indexed.

## Slice C — intake

ATS APIs open by design — **Greenhouse, Lever, Ashby, Workable** — for employers the user
names, plus RSS. **No LinkedIn or Indeed scraping**, unchanged and not negotiable.

Email intake is a **dedicated forwarding address**, not an inbox integration. Reaffirmed
2026-09-07; Gmail restricted scopes need an annual CASA assessment.

Arriving roles are scored on the same two axes as B4.

## Slice D — the loops that close

### D1 — interview preparation

Its own section, showing every application in `interviewing`. Paste the recruiter's prep
notes; get a plan anchored to the actual interview date — which is what the timestamps in
A6 exist for.

**It must say what is not worth preparing.** The owner raised this and it is not a
footnote: this tool exists to stop someone presenting a distorted picture of themselves,
and a prep plan that rehearses them into a person they are not is that same failure in
different clothes. Naming the two or three things that matter, and explicitly saying the
rest is over-preparation, is the product's thesis applied to interviews.

### D2 — rejection, feedback, and the plan

A rejection captures its reason, and that reason **is stored verbatim**. No model tidies
it. It is the only external ground truth in the system and it is scarce — the same
reasoning that keeps gap answers in the user's own words.

That feeds a gap-analysis and action-plan section, **separate from applications** because
it is about the person over time rather than about one role.

**Every step names the requirement it closes and the evidence it would produce. A step
that creates no citable evidence is not a step.** That is what stops a plan degenerating
into "get better at Kafka" and makes it "build X, which produces Y". And when that
evidence exists it becomes a corpus span — which is the flywheel closing: a rejection
eventually improves the corpus the claim gate measures against.

## Cross-cutting, before the model-facing work

- **Error references.** Every unhandled exception gets a short reference shown to the
  user and logged with the traceback. Not a browser traceback: the failure that motivated
  this was on `/auth/google/callback`, which is not behind authentication, and a stack
  through `envelope.py` names the crypto layer to whoever hit it.
- **Drop `users.email`'s UNIQUE NOT NULL.** Email is presentation-only now that identity
  is Google's `sub`; the constraint makes a reassigned address a hard login failure for
  its new owner.
- **The prompt batch.** Two known defects — document titles read as assertions, the model
  re-splitting its own input at initials — plus the deferred prompt rename, in one change
  and one $2.07 eval re-run. **Before drafting goes in front of anyone but the owner**, or
  the published over-claim rate describes a prompt that is not running.


## Risks worth naming now

**Slice A is boring and that is the point.** The temptation will be to jump to B because
the engine is the interesting part. The engine already works and is measured; the reason
the tool isn't used daily is that there is nowhere to put an application. Resist.

**Holding other people's API keys raises the stakes on every mistake.** A logging bug is
now a credential disclosure. This is why custody is in slice A rather than bolted on when
the first non-owner user appears.

**Two prompt defects are known and unfixed** — document titles read as assertions, and
the model re-splitting its own input at initials. Both need a prompt change, which
invalidates the 2026-09-05 eval baseline, so they are batched with the deferred prompt
rename into one $2.07 re-run. Do that before slice B puts drafting in front of a second
user.

**The demo page must not rot.** It is the honest front door and it is pre-computed. If
the pipeline changes materially, regenerate it — nine combinations, ~$2.82 — or take it
down. A page claiming "this is what the tool outputs" must stay true.
