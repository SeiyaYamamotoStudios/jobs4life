"""Identity and sessions. The only two repositories that legitimately run before
a user is known -- see `jfl_core.storage.tenancy.PreAuthRepository`.

Two rules encoded here rather than left to the caller:

  * a user is found by Google's `sub`, never by email. Email addresses are
    reassigned -- inside a workspace they are reassigned to *different people* --
    so matching on one would eventually hand somebody another person's account.
    Email and display name are written on every login for presentation and are
    never read back as a key. `users.email` carries no uniqueness constraint
    (migration 6b3ce06d7b4e) for the same reason: two different
    `sub`s legitimately displaying the same address, mid-reassignment, must
    both be able to sign in.
  * a session row stores sha256 of the cookie value, never the value. Reading
    this table yields no usable session.
"""

from __future__ import annotations

import datetime as dt
import uuid

from pydantic import BaseModel
from sqlalchemy import delete, insert, select, update

from jfl_core.db.tables import sessions as sessions_table
from jfl_core.db.tables import users as users_table
from jfl_core.storage.tenancy import PreAuthRepository


class AccountUser(BaseModel):
    """A signed-in identity. `google_sub` is the key; the rest is presentation."""

    id: uuid.UUID
    google_sub: str | None
    email: str
    display_name: str | None
    is_active: bool


class AuthenticatedSession(BaseModel):
    """What a valid cookie resolves to: the session and who it belongs to."""

    id: uuid.UUID
    user: AccountUser
    csrf_token: str
    created_at: dt.datetime
    last_seen_at: dt.datetime
    expires_at: dt.datetime


class PostgresUserRepository(PreAuthRepository):
    """Resolves a Google identity to a local user row.

    Not tenant-scoped because it is what *produces* the tenant: at the point it
    runs there is no `user_id` yet, only an id token.
    """

    def upsert_google_user(
        self, *, google_sub: str, email: str, display_name: str | None
    ) -> AccountUser:
        """Find the account for this `sub`, or create one; refresh the display
        fields either way. Returns the account as it now stands.
        """
        row = self._conn.execute(
            select(
                users_table.c.id,
                users_table.c.google_sub,
                users_table.c.email,
                users_table.c.display_name,
                users_table.c.is_active,
            ).where(users_table.c.google_sub == google_sub)
        ).first()

        if row is not None:
            self._conn.execute(
                update(users_table)
                .where(users_table.c.id == row.id)
                .values(email=email, display_name=display_name)
            )
            return AccountUser(
                id=row.id,
                google_sub=row.google_sub,
                email=email,
                display_name=display_name,
                is_active=row.is_active,
            )

        # No email-clash check: `users.email` is not unique, so a second `sub`
        # arriving with an address another account already displays (Google
        # reassigned it) simply gets its own row. Refusing that here would be
        # the hard login failure this table's lack of a unique constraint
        # exists to avoid -- see the module docstring.
        user_id = uuid.uuid4()
        self._conn.execute(
            insert(users_table).values(
                id=user_id,
                google_sub=google_sub,
                email=email,
                display_name=display_name,
            )
        )
        return AccountUser(
            id=user_id,
            google_sub=google_sub,
            email=email,
            display_name=display_name,
            is_active=True,
        )


class PostgresSessionRepository(PreAuthRepository):
    """Session rows keyed by the hash of the cookie value.

    Not tenant-scoped because `lookup` is the step that discovers which tenant is
    asking; the methods that do take a `user_id` (`create`, `delete_for_user`)
    are called with one the caller has just established, never with one from a
    request parameter.
    """

    def create(
        self,
        *,
        user_id: uuid.UUID,
        token_hash: str,
        csrf_token: str,
        expires_at: dt.datetime,
        user_agent: str | None,
    ) -> uuid.UUID:
        session_id = uuid.uuid4()
        self._conn.execute(
            insert(sessions_table).values(
                id=session_id,
                user_id=user_id,
                token_hash=token_hash,
                csrf_token=csrf_token,
                expires_at=expires_at,
                user_agent=user_agent,
            )
        )
        return session_id

    def lookup(self, token_hash: str, *, now: dt.datetime) -> AuthenticatedSession | None:
        """The live session for this token hash, or None.

        Expired and deactivated-user rows resolve to None rather than to a
        session the caller then has to remember to check.
        """
        row = self._conn.execute(
            select(
                sessions_table.c.id,
                sessions_table.c.csrf_token,
                sessions_table.c.created_at,
                sessions_table.c.last_seen_at,
                sessions_table.c.expires_at,
                users_table.c.id.label("user_id"),
                users_table.c.google_sub,
                users_table.c.email,
                users_table.c.display_name,
                users_table.c.is_active,
            )
            .select_from(
                sessions_table.join(users_table, sessions_table.c.user_id == users_table.c.id)
            )
            .where(sessions_table.c.token_hash == token_hash)
        ).first()
        if row is None or row.expires_at <= now or not row.is_active:
            return None
        return AuthenticatedSession(
            id=row.id,
            user=AccountUser(
                id=row.user_id,
                google_sub=row.google_sub,
                email=row.email,
                display_name=row.display_name,
                is_active=row.is_active,
            ),
            csrf_token=row.csrf_token,
            created_at=row.created_at,
            last_seen_at=row.last_seen_at,
            expires_at=row.expires_at,
        )

    def touch(self, token_hash: str, *, now: dt.datetime, expires_at: dt.datetime) -> None:
        """Roll the expiry forward. The TTL policy lives in the caller."""
        self._conn.execute(
            update(sessions_table)
            .where(sessions_table.c.token_hash == token_hash)
            .values(last_seen_at=now, expires_at=expires_at)
        )

    def delete(self, token_hash: str) -> None:
        """Log out. The row goes; a cleared cookie alone would leave a live
        session behind for anyone who copied it.
        """
        self._conn.execute(delete(sessions_table).where(sessions_table.c.token_hash == token_hash))

    def delete_for_user(self, user_id: uuid.UUID) -> int:
        """Revoke every session for one account -- the incident-response lever."""
        result = self._conn.execute(
            delete(sessions_table).where(sessions_table.c.user_id == user_id)
        )
        return result.rowcount

    def purge_expired(self, *, now: dt.datetime) -> int:
        result = self._conn.execute(
            delete(sessions_table).where(sessions_table.c.expires_at <= now)
        )
        return result.rowcount
