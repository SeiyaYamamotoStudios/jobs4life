"""Credential custody. See `envelope.py` -- nothing here may log key material."""

from jfl_core.crypto.envelope import (
    MASTER_KEY_ENV,
    MasterKey,
    MasterKeyError,
    SealedSecret,
    SecretUnsealError,
    seal,
    secret_hint,
    unseal,
)

__all__ = [
    "MASTER_KEY_ENV",
    "MasterKey",
    "MasterKeyError",
    "SealedSecret",
    "SecretUnsealError",
    "seal",
    "secret_hint",
    "unseal",
]
