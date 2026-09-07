"""Envelope encryption. The properties here are the ones whose absence is a
credential disclosure, not a bug.
"""

from __future__ import annotations

import base64
import uuid

import pytest
from jfl_core.crypto.envelope import (
    KEY_BYTES,
    MASTER_KEY_ENV,
    MasterKey,
    MasterKeyError,
    SealedSecret,
    SecretUnsealError,
    seal,
    secret_hint,
    unseal,
)

PROVIDER = "anthropic_api_key"
SECRET = "sk-ant-api03-not-a-real-key-0000000000000000000000004f2a"


@pytest.fixture
def master() -> MasterKey:
    return MasterKey.generate()


@pytest.fixture
def user() -> uuid.UUID:
    return uuid.uuid4()


def test_round_trips(master: MasterKey, user: uuid.UUID) -> None:
    sealed = seal(master, SECRET, user_id=user, provider=PROVIDER)
    assert unseal(master, sealed, user_id=user, provider=PROVIDER) == SECRET


def test_plaintext_appears_nowhere_in_the_sealed_row(master: MasterKey, user: uuid.UUID) -> None:
    sealed = seal(master, SECRET, user_id=user, provider=PROVIDER)
    raw = sealed.ciphertext + sealed.nonce + sealed.wrapped_dek + sealed.dek_nonce
    assert SECRET.encode() not in raw
    assert master.material not in raw


def test_two_encryptions_of_one_plaintext_differ(master: MasterKey, user: uuid.UUID) -> None:
    """Nonce freshness, observed from the outside.

    A repeated nonce under one key is not a weakening of GCM, it is a break: the
    keystream and the GHASH authentication key both fall out. Identical
    ciphertext for identical plaintext is the symptom.
    """
    first = seal(master, SECRET, user_id=user, provider=PROVIDER)
    second = seal(master, SECRET, user_id=user, provider=PROVIDER)

    assert first.nonce != second.nonce
    assert first.dek_nonce != second.dek_nonce
    assert first.ciphertext != second.ciphertext
    assert first.wrapped_dek != second.wrapped_dek
    # Both still decrypt: different ciphertext, same secret.
    assert unseal(master, first, user_id=user, provider=PROVIDER) == SECRET
    assert unseal(master, second, user_id=user, provider=PROVIDER) == SECRET


def test_nonces_are_ninety_six_bits(master: MasterKey, user: uuid.UUID) -> None:
    sealed = seal(master, SECRET, user_id=user, provider=PROVIDER)
    assert len(sealed.nonce) == 12
    assert len(sealed.dek_nonce) == 12


def test_many_seals_never_repeat_a_nonce(master: MasterKey, user: uuid.UUID) -> None:
    nonces = {seal(master, SECRET, user_id=user, provider=PROVIDER).nonce for _ in range(200)}
    assert len(nonces) == 200


def test_a_row_cannot_be_transplanted_to_another_user(master: MasterKey, user: uuid.UUID) -> None:
    """AAD binds the ciphertext to its row. Copying it onto another user fails."""
    sealed = seal(master, SECRET, user_id=user, provider=PROVIDER)
    with pytest.raises(SecretUnsealError):
        unseal(master, sealed, user_id=uuid.uuid4(), provider=PROVIDER)


def test_a_row_cannot_be_transplanted_to_another_provider(
    master: MasterKey, user: uuid.UUID
) -> None:
    sealed = seal(master, SECRET, user_id=user, provider=PROVIDER)
    with pytest.raises(SecretUnsealError):
        unseal(master, sealed, user_id=user, provider="openai_api_key")


def test_tampered_ciphertext_is_rejected(master: MasterKey, user: uuid.UUID) -> None:
    sealed = seal(master, SECRET, user_id=user, provider=PROVIDER)
    flipped = bytearray(sealed.ciphertext)
    flipped[0] ^= 0x01
    with pytest.raises(SecretUnsealError):
        unseal(
            master,
            SealedSecret(
                wrapped_dek=sealed.wrapped_dek,
                dek_nonce=sealed.dek_nonce,
                ciphertext=bytes(flipped),
                nonce=sealed.nonce,
                master_key_id=sealed.master_key_id,
            ),
            user_id=user,
            provider=PROVIDER,
        )


def test_a_different_master_key_is_named_not_guessed(user: uuid.UUID) -> None:
    """The wrong KEK gets a diagnosis, not an InvalidTag -- the difference
    between "restore your key" and an hour of debugging.
    """
    sealed = seal(MasterKey.generate(), SECRET, user_id=user, provider=PROVIDER)
    with pytest.raises(MasterKeyError) as exc:
        unseal(MasterKey.generate(), sealed, user_id=user, provider=PROVIDER)
    assert MASTER_KEY_ENV in str(exc.value)


def test_master_key_is_never_in_a_repr(master: MasterKey) -> None:
    text = repr(master)
    assert master.key_id in text
    assert base64.b64encode(master.material).decode() not in text
    assert str(master.material) not in text


def test_missing_master_key_fails_loudly() -> None:
    with pytest.raises(MasterKeyError) as exc:
        MasterKey.from_env({})
    assert "never generated automatically" in str(exc.value)


def test_malformed_master_key_fails_loudly() -> None:
    with pytest.raises(MasterKeyError):
        MasterKey.from_env({MASTER_KEY_ENV: "not base64!!!"})


def test_short_master_key_is_rejected() -> None:
    short = base64.b64encode(b"\x00" * 16).decode()
    with pytest.raises(MasterKeyError) as exc:
        MasterKey.from_env({MASTER_KEY_ENV: short})
    assert str(KEY_BYTES) in str(exc.value)


def test_key_id_is_stable_and_not_the_key() -> None:
    material = base64.b64encode(b"\x11" * KEY_BYTES).decode()
    a = MasterKey.from_base64(material)
    b = MasterKey.from_base64(material)
    assert a.key_id == b.key_id
    assert a.key_id != material
    assert len(a.key_id) == 16


def test_hint_is_the_last_four_characters() -> None:
    assert secret_hint(SECRET) == "4f2a"
    assert secret_hint("abc") == ""  # too short to hint without leaking the lot
