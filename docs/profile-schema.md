# The profile: one row, four sections

Agreed 2026-09-21. Replaces `profile_answers`, `profile_objectives` and
`profile_ruled_out` (PLAN.md B3a's eighteen free-text questions), which production held
zero rows of, so nothing is migrated.

Research behind it is in `~/jobs4life-profile-research/` (four files: commercial
platforms, standards and taxonomies, psychology, feedback loops) — not in the repo,
because it is long and it is reference material, not decisions. The decisions are here.

## Three stores, not one

| Store | Holds | Why it is separate |
|---|---|---|
| **Profile** (`profiles`) | Preferences and claims about yourself | One read serves scoring, drafting and the filter |
| **Corpus** (`documents` → `spans`) | Facts you have confirmed, verbatim | The claim gate cites span ids; spans retire when you change your mind. A JSON blob cannot be cited |
| **Application artefacts** | CVs, drafts, answers, outcomes | Per application: the record of what was sent and what came back |

A rejection with its attributed reason belongs to its application, never to the profile —
one home per fact. The profile reads outcomes; it does not own them.

## The table

```sql
profiles(id, user_id → users, schema_version int, data jsonb, created_at)
```

**Append-only.** Every save writes a new row; the current profile is the latest. Same rule
the tracker's event timeline and the old `profile_answers` used: what you believed about
yourself in March stays readable, and undo is free. Rows are a few KB.

`data` has four sections, and **one Pydantic model is the only write path**:

```jsonc
{
  "constraints":  [{"kind": "comp_floor", "stance": "must",
                    "value": {"guaranteed": 120000, "headline": 145000, "ccy": "GBP"},
                    "note": "base + pension, ignoring equity"}],
  "capabilities": [{"label": "FX pricing platforms", "tier": "production_depth",
                    "interest": "want_more", "last_used": 2024,
                    "evidence": ["<span id>"], "source": "cv_fact"}],
  "disciplines":  {"practises": ["engineering management"], "not": ["frontend"]},
  "objectives":   [{"rank": 1, "text": "...", "evidence_of_delivery": "..."}]
}
```

### constraints
`kind` ∈ location, workplace, level_floor, comp_floor, contract, right_to_work, notice,
categorical_no. `stance` ∈ **must / nice / never** — taken from Hired, the one good idea in
the commercial survey: a negative preference gets equal standing with a positive one.
Locations are an **ordered list**, not a relocate boolean (SEEK's shape). Comp carries
**guaranteed and headline separately**, because a headline number is not an offer.

### capabilities
`tier` ∈ **production_depth / working / oversight_only / absent**, set by two or three
behavioural questions rather than a self-rating — the shape SFIA arrived at (seven levels
across five independent attributes, anchored in observable behaviour). SFIA's own text is
licensed against commercial use and covers only IT, so this is our scale in our words, and
it is never labelled with SFIA's name. `interest` is a **separate axis** from depth, as
O*NET rates level and importance separately: what you are good at and what you want to keep
doing are different questions.

**A tier with no evidence is a claim, not a fact** — the same status a CV line has before
confirmation. `evidence` holds span ids.

### disciplines
What you practise, as distinct from what employers called you. Ranked, plus an explicit
"not this" list. Without it, the same job title at two employers reads as the same job.

### objectives
Up to four, **ranked and scored separately, never blended**. Each carries "what would show a
role delivers this". Theory of Work Adjustment split satisfaction from satisfactoriness in
1969 and this project reached the same rule independently; blending hides the trade-off
that is the whole reason to look.

## What this costs us

**The value-list drift guard does not reach inside JSONB.** Elsewhere a `Literal`, a tuple
and a CHECK constraint must agree, and two tests enforce it. Here the Pydantic model is the
only write path and a test asserts its allowed values match what the screens offer. That is
weaker than a CHECK, it was accepted deliberately (owner, 2026-09-21), and it is the price
of a shape we expect to change while we learn what belongs in it.

## How the profile drives the two scores

- **Could I get this** — the job's requirements against corpus coverage, with capability
  tiers as the bridge. Claims with no evidence appear as **levers** ("confirm X and this
  moves to 7"), never as evidence.
- **Do I want this** — **not a predicted-satisfaction score.** Each constraint and objective
  is reported against the ad as **evidenced / partial / silent / contradicted**, and the
  number falls out of that coverage. Computed person-job fit predicts satisfaction at
  ρ≈.28 (against .61 when people rate fit themselves), and people forecast their own job
  satisfaction badly — focalism, honeymoon-hangover, ~30% heritability. What does predict
  satisfaction is met expectations (.39), so the honest product is a list of what the ad
  evidences and what it is silent on. The silences are the questions to ask at interview.

Both numbers, one or two sentences each, never combined, labelled unmeasured.

## How it constrains generation

**A draft may not claim above the tier you confirmed.** Say `working` and no generated CV
says "deep expertise". This is the thesis applied to our own output, and it is cheaper than
catching the same over-claim at the gate.

## Filling it in

Five constraints, one discipline, two or three objectives — about ten minutes. Capability
rows arrive **pre-proposed from confirmed CV facts**, so tiering them is a click each.
Everything is optional; a skipped field reports "not stated" and is never guessed.

## Deliberately later

Tolerance thresholds; the durability/exposure model; O*NET's six work values (public-domain
card sort); move triggers per role (59% of leavers report a discrete trigger, and the CV
already names the moves); behavioural flags — **user-authored only, never model-inferred**;
and the pushback log, which lands with the first score that can be argued with.

## Refused, on evidence

Inferred personality from CVs, drafts or behaviour; anything clinical-adjacent; integrity or
honesty-humility measures — the largest effect available and the last thing a truthfulness
tool should store about its own user; off-the-job embeddedness, which is validated but
proxies family status, age and national origin; and any single composite person-level score.
The EU AI Act bans workplace emotion inference outright.
