"""Stored credentials, tenancy-scoped. Ciphertext in, ciphertext out.

This repository never sees a plaintext secret and has no way to produce one: it
moves `SealedSecret` values, and unsealing needs the master key, which lives at
the application boundary and is never handed to storage. That is not tidiness --
it means no logging or debugging change *here* can ever print a key, because
there is no key here to print.

`provider` maps onto the schema's `user_credentials.kind` column, which predates
this slice and already carries the CHECK constraint listing valid values. One
credential per (user, provider): `label` stays empty, reserved for the day a user
wants two keys for one provider.
"""

from __future__ import annotations

import datetime as dt
import uuid

from pydantic import BaseModel
from sqlalchemy import delete, func, insert, select, update

from jfl_core.crypto.envelope import SealedSecret
from jfl_core.db.tables import user_credentials as user_credentials_table
from jfl_core.storage.tenancy import TenantScopedRepository

ANTHROPIC_API_KEY = "anthropic_api_key"

# One credential per provider for now; the column exists so a second can be added
# without a migration.
_DEFAULT_LABEL = ""


class CredentialSummary(BaseModel):
    """Everything the UI is allowed to know about a stored key.

    Note what is absent: the secret, and any part of it beyond the four
    characters in `key_hint`. A key is write-only from the browser's side, so
    there is no shape of response that could carry one back.
    """

    provider: str
    key_hint: str
    master_key_id: str
    created_at: dt.datetime
    rotated_at: dt.datetime | None
    last_used_at: dt.datetime | None


class PostgresCredentialRepository(TenantScopedRepository):
    """Credential rows for exactly one user, fixed at construction."""

    def store(self, *, provider: str, sealed: SealedSecret, key_hint: str) -> None:
        """Insert or replace this user's credential for `provider`.

        Replacing overwrites every crypto field together -- ciphertext, nonce,
        wrapped DEK and its nonce are one unit and a partial update would
        produce a row that cannot be decrypted.
        """
        existing = self._conn.execute(
            select(user_credentials_table.c.id).where(
                user_credentials_table.c.user_id == self._user_id,
                user_credentials_table.c.kind == provider,
                user_credentials_table.c.label == _DEFAULT_LABEL,
            )
        ).first()

        values = {
            "ciphertext": sealed.ciphertext,
            "nonce": sealed.nonce,
            "wrapped_dek": sealed.wrapped_dek,
            "dek_nonce": sealed.dek_nonce,
            "master_key_id": sealed.master_key_id,
            "key_hint": key_hint,
        }
        if existing is None:
            self._conn.execute(
                insert(user_credentials_table).values(
                    id=uuid.uuid4(),
                    user_id=self._user_id,
                    kind=provider,
                    label=_DEFAULT_LABEL,
                    **values,
                )
            )
            return
        self._conn.execute(
            update(user_credentials_table)
            .where(user_credentials_table.c.id == existing.id)
            .values(rotated_at=func.now(), last_used_at=None, **values)
        )

    def summary(self, provider: str) -> CredentialSummary | None:
        row = self._conn.execute(
            select(
                user_credentials_table.c.kind,
                user_credentials_table.c.key_hint,
                user_credentials_table.c.master_key_id,
                user_credentials_table.c.created_at,
                user_credentials_table.c.rotated_at,
                user_credentials_table.c.last_used_at,
            ).where(
                user_credentials_table.c.user_id == self._user_id,
                user_credentials_table.c.kind == provider,
                user_credentials_table.c.label == _DEFAULT_LABEL,
            )
        ).first()
        if row is None:
            return None
        return CredentialSummary(
            provider=row.kind,
            key_hint=row.key_hint,
            master_key_id=row.master_key_id,
            created_at=row.created_at,
            rotated_at=row.rotated_at,
            last_used_at=row.last_used_at,
        )

    def load_sealed(self, provider: str) -> SealedSecret | None:
        """The stored ciphertext. Unsealing it needs the master key, which this
        layer does not have -- see `jfl_core.crypto.envelope.unseal`.
        """
        row = self._conn.execute(
            select(
                user_credentials_table.c.ciphertext,
                user_credentials_table.c.nonce,
                user_credentials_table.c.wrapped_dek,
                user_credentials_table.c.dek_nonce,
                user_credentials_table.c.master_key_id,
            ).where(
                user_credentials_table.c.user_id == self._user_id,
                user_credentials_table.c.kind == provider,
                user_credentials_table.c.label == _DEFAULT_LABEL,
            )
        ).first()
        if row is None:
            return None
        return SealedSecret(
            ciphertext=row.ciphertext,
            nonce=row.nonce,
            wrapped_dek=row.wrapped_dek,
            dek_nonce=row.dek_nonce,
            master_key_id=row.master_key_id,
        )

    def mark_used(self, provider: str) -> None:
        self._conn.execute(
            update(user_credentials_table)
            .where(
                user_credentials_table.c.user_id == self._user_id,
                user_credentials_table.c.kind == provider,
                user_credentials_table.c.label == _DEFAULT_LABEL,
            )
            .values(last_used_at=func.now())
        )

    def delete(self, provider: str) -> None:
        self._conn.execute(
            delete(user_credentials_table).where(
                user_credentials_table.c.user_id == self._user_id,
                user_credentials_table.c.kind == provider,
                user_credentials_table.c.label == _DEFAULT_LABEL,
            )
        )
