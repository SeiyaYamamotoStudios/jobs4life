"""Watched job boards (domain 3, intake): adapters, detection, and the check engine.

The owner pastes an employer's careers-board URL and this package does the rest:
work out which ATS it is (`detect`), fetch every listed job from that ATS's
public API (`adapters`), and decide what the result means for the history of
that board (`engine`).

**Deterministic throughout.** No model call anywhere in this package, and no
`anthropic` dependency to make one with. Intake is fixed control flow by
CLAUDE.md's own taxonomy, and every decision here is a rule that can be read.

**No LinkedIn or Indeed**, in any form -- `detect` rejects their URLs with a
message saying why. The adapters use the ATS APIs that are public by design.

Layering: `engine` is pure (no database, no network) so the rule the whole
feature rests on -- only a complete check may close a presence interval -- is
unit-tested directly. Persistence lives in `jfl_core.storage.boards`; running a
check lives in the worker's `check_board` handler.
"""
