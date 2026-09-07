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
| **B** | Job queue, then the existing engine behind a UI | Draft and gate an application without touching a terminal |
| **C** | Intake: ATS APIs, forwarding address, two-axis rating | Roles arriving without being hunted for |
| **D** | Interview stages, rejection feedback, skill-up plans | A closed loop from rejection to a plan |

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

### B1 — The job queue, first

A gate call takes ~2 minutes. Nothing that slow runs inside a request.

`tasks` table with `SELECT … FOR UPDATE SKIP LOCKED`, an attempt counter, a max-attempts
cap, and a global `JFL_DISABLE_MODEL_CALLS` kill switch. The worker is a separate
container so it can be stopped without taking the site down — which is also the incident
response if a user's key starts burning money.

### B2 — The engine's screens

All of this exists and is tested; it needs a UI, not a rewrite: requirement extraction,
per-requirement coverage, gap questions, verbatim answer write-back, drafting, and the
claim gate with per-sentence verdicts.

**Framing renders as `NOT CHECKED`, never as supported** — the same rule the demo page
already obeys, for the same reason. Per-run cost is shown to the user, because they are
paying for it with their own key.

### B3 — Corpus upload

A user with no corpus has nothing to measure against, so this gates B's usefulness for
anyone but the owner. Markdown upload, parsed by the existing ingestion, spans stored
per-user. The "markdown is the source of truth" rule holds: uploads are stored and
re-ingestible, never only indexed.

---

## Slice C — intake

ATS APIs open by design — **Greenhouse, Lever, Ashby, Workable** — for employers the user
names, plus RSS. **No LinkedIn or Indeed scraping**, unchanged and not negotiable.

Email intake is a **dedicated forwarding address**, not an inbox integration. Reaffirmed
2026-09-07; Gmail restricted scopes need an annual CASA assessment.

The rating is model-judged, **two axes, never composited**: *do I want this* and *could I
get this*. It ships unmeasured and labelled unmeasured — there is no golden set for fit,
and inventing one would be the synthetic-data prohibition wearing a new coat.

---

## Slice D — the loop that closes

Interview stages tracked as events on the application, so preparation can be anchored to
a real date.

Then the part that makes this more than a tracker: a rejection, with its reason, becomes
a skill-up plan. Feedback is **the only external ground truth in the system and it is
scarce** — capture it carefully and never paraphrase it into something tidier, for the
same reason gap answers are stored verbatim.

**Every step in a plan names the requirement it closes and the evidence it would produce.
A step that creates no citable evidence is not a step.** That evidence, once produced,
becomes a corpus span — which is the flywheel closing: a rejection eventually improves
the corpus that the claim gate measures against.

---

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
