"""Shared error type for generation's model calls.

Mirrors `jfl_gate.gate.GateError`: raised only after the failure's `runs` row
has already been written, so the caller (the CLI) just needs to report it.
"""

from __future__ import annotations

from jfl_core.model_api import ModelCallError


class GenerateError(ModelCallError):
    """`api_failure` (from `ModelCallError`) is set when the model API refused
    the call -- which is how a worker handler tells "the user's credits ran
    out" from a bug without parsing this message. See `jfl_core.model_api`.
    """
