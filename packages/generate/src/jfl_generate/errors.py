"""Shared error type for generation's model calls.

Mirrors `jfl_gate.gate.GateError`: raised only after the failure's `runs` row
has already been written, so the caller (the CLI) just needs to report it.
"""

from __future__ import annotations


class GenerateError(RuntimeError):
    pass
