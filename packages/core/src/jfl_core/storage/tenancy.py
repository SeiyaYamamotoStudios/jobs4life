"""Tenancy, enforced by construction rather than by review.

The failure this exists to prevent is a missing `WHERE user_id = ...`, which
leaks one person's career history to another. A clause a reviewer has to spot is
not enforcement -- so a tenancy-scoped repository is **constructed with** the
user it may act on and offers no way to name a different one.

The rule is mechanical, which is what makes it testable:

  * a `TenantScopedRepository` subclass takes its `user_id` in `__init__` and
    stores it privately;
  * no public method of such a class may take a parameter named `user_id`;
  * every repository class in a module that participates in this scheme must
    subclass either `TenantScopedRepository` or `PreAuthRepository`.

`tests/test_tenancy_enforcement.py` walks the packages and fails the build on any
violation. It discovers classes by subclass and by name, so a repository added
next year is covered without anyone remembering to update the test.

`jfl_core.storage.postgres` predates this scheme -- it is the single-user CLI's
storage layer, takes `user_id` per call, and is listed as legacy in that test.
New repositories do not get to join it.
"""

from __future__ import annotations

import uuid

from sqlalchemy.engine import Connection


class TenantScopedRepository:
    """Bound at construction to the one user it may touch.

    Subclasses hold their queries to `self._user_id` and never accept a caller's
    idea of who is asking. The base class deliberately provides no
    `for_user(...)` or `with_user(...)` escape hatch: if such a method existed,
    forgetting to call it would be expressible again.

    Like `jfl_core.storage.postgres`, this takes a `Connection` and never opens
    or commits a transaction -- the caller owns that boundary.
    """

    def __init__(self, conn: Connection, user_id: uuid.UUID) -> None:
        self._conn = conn
        self._user_id = user_id

    @property
    def user_id(self) -> uuid.UUID:
        """Whose data this instance can see. Readable, never settable."""
        return self._user_id


class PreAuthRepository:
    """Deliberately NOT tenant-scoped, because it runs before a tenant is known.

    A request arrives carrying a cookie and nothing else; something has to turn
    that into a `user_id`, and that something cannot itself be scoped to one.
    Only identity resolution and session lookup may live here, and every subclass
    must say in its own docstring why it cannot be scoped.

    Anything that reads or writes a user's *content* belongs in
    `TenantScopedRepository`, no exceptions.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn
