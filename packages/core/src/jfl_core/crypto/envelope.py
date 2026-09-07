"""Envelope encryption for stored credentials. AES-GCM, two layers.

Two layers because they have different lifetimes and different blast radii:

  KEK (master key) -- 32 bytes, base64, from ``JFL_MASTER_KEY`` in the host
                      environment. Never written to Postgres, so a database dump
                      on its own is inert.
  DEK (data key)   -- 32 random bytes minted per stored credential and held only
                      in wrapped form, encrypted under the KEK. Rotating the
                      master key rewraps DEKs and leaves the ciphertext alone.

**A fresh 96-bit nonce is generated for every encryption, at both layers.**
Reusing a nonce under a fixed AES-GCM key is not a weakening, it is a break: two
messages under one (key, nonce) leak their XOR and, worse, leak the GHASH
authentication key, which makes forgery possible. So nonces come from
``os.urandom`` at the point of use and are never derived, cached, or defaulted.

Every encryption binds associated data naming the row it belongs to. AAD is
authenticated but not encrypted -- it hides nothing, it makes a ciphertext refuse
to decrypt anywhere except where it was written. Copying one user's row onto
another user raises ``InvalidTag`` rather than silently succeeding.

Fail loudly on a missing or malformed master key. Never generate one on the fly:
a silently minted key would decrypt nothing that was stored before the restart,
so every user's credential would be orphaned with no error at the point of
failure.

NOTHING IN THIS MODULE MAY LOG, PRINT, REPR OR STRINGIFY KEY MATERIAL OR A
PLAINTEXT SECRET. The dataclasses below deliberately keep key material out of
their ``repr``; error messages name the key *id* (a hash) and never the key.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MASTER_KEY_ENV = "JFL_MASTER_KEY"

KEY_BYTES = 32  # AES-256
NONCE_BYTES = 12  # 96 bits, the only nonce size AES-GCM is specified for

# Bumping this version invalidates every stored ciphertext by design: the AAD no
# longer matches, so an old row fails to decrypt loudly instead of quietly.
_AAD_VERSION = "jfl.v1"


class MasterKeyError(RuntimeError):
    """The master key is absent, malformed, or not the one that sealed a row."""


class SecretUnsealError(RuntimeError):
    """A sealed secret failed to authenticate: wrong key, or tampered/moved row."""


def _key_id(material: bytes) -> str:
    """A stable, non-secret identifier for a key.

    A truncated hash of a 256-bit random key, domain-separated so it cannot
    collide with any other hash the system computes. Publishing it reveals
    nothing usable and lets a row say which KEK sealed it.
    """
    return hashlib.sha256(b"jfl.master-key-id.v1|" + material).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class MasterKey:
    """The key-encryption key. ``material`` is excluded from ``repr`` on purpose."""

    material: bytes = field(repr=False)
    key_id: str

    @classmethod
    def from_base64(cls, value: str) -> MasterKey:
        try:
            material = base64.b64decode(value.strip(), validate=True)
        except (binascii.Error, ValueError):
            # `from None`: never chain an exception whose context could carry the
            # value that failed to decode.
            raise MasterKeyError(
                f"{MASTER_KEY_ENV} is not valid base64. Generate one with: openssl rand -base64 32"
            ) from None
        if len(material) != KEY_BYTES:
            raise MasterKeyError(
                f"{MASTER_KEY_ENV} decodes to {len(material)} bytes; "
                f"{KEY_BYTES} are required. Generate one with: openssl rand -base64 32"
            )
        return cls(material=material, key_id=_key_id(material))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> MasterKey:
        """Read the KEK from the environment. Raises rather than inventing one."""
        source: Mapping[str, str] = os.environ if env is None else env
        raw = source.get(MASTER_KEY_ENV)
        if not raw:
            raise MasterKeyError(
                f"{MASTER_KEY_ENV} is not set. It is never generated automatically: "
                "a fresh key would orphan every credential already stored. "
                "Generate one once with `openssl rand -base64 32` and keep it."
            )
        return cls.from_base64(raw)

    @classmethod
    def generate(cls) -> MasterKey:
        """A new random KEK. For tests and for one-off provisioning only --
        never called on a code path that could run at startup.
        """
        return cls.from_base64(base64.b64encode(os.urandom(KEY_BYTES)).decode())


@dataclass(frozen=True, slots=True)
class SealedSecret:
    """Exactly what goes in a `user_credentials` row. No plaintext, ever."""

    wrapped_dek: bytes
    dek_nonce: bytes
    ciphertext: bytes
    nonce: bytes
    master_key_id: str


def _aad(layer: str, user_id: uuid.UUID, provider: str) -> bytes:
    """Context binding: which layer, whose row, which provider."""
    return f"{_AAD_VERSION}|{layer}|{user_id}|{provider}".encode()


def seal(master: MasterKey, plaintext: str, *, user_id: uuid.UUID, provider: str) -> SealedSecret:
    """Wrap a fresh DEK under the KEK, then encrypt the secret under the DEK."""
    dek = os.urandom(KEY_BYTES)
    dek_nonce = os.urandom(NONCE_BYTES)
    wrapped_dek = AESGCM(master.material).encrypt(dek_nonce, dek, _aad("dek", user_id, provider))

    nonce = os.urandom(NONCE_BYTES)
    ciphertext = AESGCM(dek).encrypt(
        nonce, plaintext.encode("utf-8"), _aad("secret", user_id, provider)
    )
    return SealedSecret(
        wrapped_dek=wrapped_dek,
        dek_nonce=dek_nonce,
        ciphertext=ciphertext,
        nonce=nonce,
        master_key_id=master.key_id,
    )


def unseal(master: MasterKey, sealed: SealedSecret, *, user_id: uuid.UUID, provider: str) -> str:
    """Recover the plaintext. Raises rather than returning anything approximate."""
    if sealed.master_key_id != master.key_id:
        raise MasterKeyError(
            f"this credential was sealed under master key {sealed.master_key_id!r} "
            f"but {MASTER_KEY_ENV} is {master.key_id!r}. Restore the original key; "
            "the stored secret cannot be recovered without it."
        )
    try:
        dek = AESGCM(master.material).decrypt(
            sealed.dek_nonce, sealed.wrapped_dek, _aad("dek", user_id, provider)
        )
        plaintext = AESGCM(dek).decrypt(
            sealed.nonce, sealed.ciphertext, _aad("secret", user_id, provider)
        )
    except InvalidTag:
        # InvalidTag carries no detail and neither does this: a decrypt failure
        # must not become an oracle.
        raise SecretUnsealError(
            "stored credential failed to authenticate -- wrong master key, "
            "altered ciphertext, or a row moved between users"
        ) from None
    return plaintext.decode("utf-8")


def secret_hint(plaintext: str, *, chars: int = 4) -> str:
    """The last few characters, for display beside a write-only field.

    Four characters of an API key identify which key is stored to the person who
    typed it and are useless to anyone else.
    """
    tail = plaintext.strip()[-chars:]
    return tail if len(tail) == chars else ""
