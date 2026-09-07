# Demo fixtures

Three fictional candidates and three fictional job ads, used to produce the public
demo at `hiltonlabs.org/jobs4life` (see PLAN.md, W2-W4). Every name, employer, and
product below is invented; none is based on a real person, including the repo owner.

## These are fixtures, not golden-set items

CLAUDE.md forbids generating synthetic *golden-set* items, because a fabricated eval
item would make the measured over-claim rate a lie about how the tool performs on
real material. That prohibition does not apply here, and the distinction matters
enough to spell out:

- A golden-set item makes a measurement claim: "the tool's over-claim rate, measured
  against known-true and known-false labels, is X%." Fabricating the labels fabricates
  the number.
- A demo fixture makes no measurement claim at all. It claims only "this is what the
  real pipeline outputs for this input" -- and W3 keeps that claim true by running the
  actual extract -> coverage -> draft -> gate pipeline over these fixtures with no
  hand-editing of results. The fiction is in the *input* (an invented candidate and an
  invented job ad); the *output* shown on the demo page is never fiction.

Nothing here is used by `packages/evals` or counted toward the eval numbers in
`packages/evals/README.md`. If that ever changes, this README is wrong and should be
fixed first.

## Layout

- `candidates/*.md` -- three verification records in the corpus markdown format
  described in `packages/core/src/jfl_core/ingest/parser.py`: a single h1 (the
  candidate's name, excluded from `section_path` as the document title), h2 employers,
  h3 roles with dates, bullets underneath, and a closing "Things Stated Explicitly as
  NOT True, or Boundaries to Hold" section whose heading matches
  `jfl_gate.rules._BOUNDARY_SECTION_MARKERS` case-insensitively.
- `job-ads/*.txt` -- three plain-text job ads of the kind someone actually pastes:
  headers, an essential/desirable split, some vague requirements and some specific
  ones, and boilerplate (benefits, EO notices) the extraction prompt is expected to
  skip.

## The candidates

- **Mireille Fontaine** (`mireille-fontaine.md`, 46 spans) -- backend/payments
  infrastructure engineer, ~11 years, four employers, currently Staff Engineer at a
  payments company. Rich, detailed record. Carries the demo's under-claim case: a
  from-scratch rebuild of a reconciliation pipeline after a data-loss incident,
  stated as flatly as everything else around it.
- **Tobias Reyes** (`tobias-reyes.md`, 41 spans) -- engineering manager, ~7 years,
  three employers, promoted from senior IC into EM at his current company eighteen
  months ago. Thinner record than Mireille's -- fewer employers, shorter bullets,
  less texture per role -- deliberately, to show how a thin record behaves
  differently from a rich one under the same checks.
- **Ingrid Solberg** (`ingrid-solberg.md`, 42 spans) -- computer-vision / applied ML
  engineer, ~15 years spanning an early research-lab role and two industry roles,
  currently Senior Applied Scientist on a robotics perception team. Different domain
  and different career shape again: research output (papers, a patent) alongside
  production deployment work.

All three ranges sit inside the 40-60 span target (verified by parsing each file with
`parse_document` and counting the resulting spans -- see "Verification" below).

## The job ads

- **Ledgerbridge Financial** (`ledgerbridge-senior-backend-engineer.txt`) -- Senior
  Backend Engineer, Payments Platform. Hands-on backend/payments IC role: Go/Java,
  Kafka, PCI-DSS familiarity, on-call, mentoring. Desirable: fintech background,
  Rust, budget experience, conference speaking.
- **Nimbus Cloudworks** (`nimbus-engineering-manager.txt`) -- Engineering Manager,
  Platform Team. People management, budget ownership, incident response, cross-org
  representation, with a hands-on-engineer-first requirement. Desirable: regulated
  industry background, Kubernetes, team scaling, public speaking.
- **Solstice Vision Labs** (`solstice-ml-engineer.txt`) -- Machine Learning Engineer,
  Perception. Production computer vision, PyTorch, data annotation pipelines, edge
  hardware constraints. Desirable: publications/patents, mentoring, robotics
  background, research-direction experience.

None of the job-ad employers share a name with any candidate's past employer.

## What the nine pairings are expected to show

Each candidate's stated-boundaries section names specific things ruled out --
payment-processing systems, formal people management, Kafka, ML/CV work, Rust, budget
ownership, and so on, split differently per candidate -- so that different pairings
hit different boundary lines instead of all collapsing onto the same one or two
contradictions.

- **Home-turf pairings** (Mireille x Ledgerbridge, Tobias x Nimbus, Ingrid x
  Solstice) should read mostly `evidenced`, but each carries at least one wrinkle so
  it is not a flat wall of green: Mireille's Rust/budget desirables are
  `contradicted` by her boundaries section; Tobias's "regulated industry" desirable
  is unaddressed for healthcare and directly `contradicted` for payments; Ingrid's
  "sets research direction" desirable lands `partial` (her boundaries section says
  direction is set jointly with her team lead, not solely by her).
- **Adjacent-domain pairings** (Mireille x Nimbus, Tobias x Ledgerbridge, Ingrid x
  Nimbus) should mix `contradicted` (management/budget for Mireille and Ingrid;
  payments-systems and Kafka for Tobias) with `partial` and `absent` on the
  requirements their record doesn't address either way (Kubernetes for Mireille,
  PCI-DSS for Tobias, incident postmortems for Ingrid).
- **Off-domain pairings** (Mireille x Solstice, Tobias x Solstice, Ingrid x
  Ledgerbridge) should be mostly `contradicted`/`absent`, each for different reasons
  drawn from that candidate's own boundaries section, so the three "poor fit" results
  don't read as interchangeable.

Bullets that state what happened without stating the boundary that would let a
generated claim be checked -- "worked on X" with no ownership stated, an outcome with
no metric or causal link, a scope-less "led" -- are spread across all three
candidates. These exist so that when W3 runs real drafts against real coverage, the
model has room to inflate into `scope_inflation`, `ownership_inflation`,
`outcome_attribution`, `strategy_scope`, and `causality` if it does; the fixtures
supply the ambiguity, not the drift itself. Nothing here writes the over-claim -- W3's
generation step does, or doesn't, and either result is a true finding about the real
pipeline.

## Verification

Every candidate file was checked with a throwaway script that ran
`jfl_core.ingest.parser.parse_document` over it and inspected the resulting spans:

- all three parse with no empty-text spans and no bold-label artifacts (the parser's
  known failure mode for a standalone `**Bold**` line becoming a garbage span --
  these fixtures use no bold markdown at all, so the shape can't occur)
- span counts: Mireille 46, Tobias 41, Ingrid 42 -- all inside the 40-60 target
- each boundaries section's `section_path` was checked against
  `jfl_gate.rules._BOUNDARY_SECTION_MARKERS` with `_is_boundary_span`, not eyeballed;
  all ten boundary bullets per candidate matched
- heading breadcrumbs are as expected: employer alone for the h2, `Employer > Role
  (dates)` for the h3s, and the boundaries heading matches on its own text
