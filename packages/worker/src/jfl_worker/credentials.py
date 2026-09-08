"""Fetching a user's Anthropic key at the moment a handler needs it.

Four lines of substance, and they are deliberately duplicated from
`jfl_web.credentials.load_api_key` rather than imported. The worker is a daemon
that must be deployable on its own -- a dependency on the web package would make
the queue container carry FastAPI, Authlib and an OAuth client in order to
decrypt a string. Nor can this live in `jfl_core.storage.credentials`, whose
whole promise is that no logging or debugging change *there* can print a key
because there is no key there to print: it moves ciphertext and never meets the
master key.

The shape of the promise here is the same as the web's:

  * the plaintext exists as a local variable across a handful of statements and
    is handed straight to the Anthropic client;
  * it is never written into a task payload, a log line, `tasks.last_error`, a
    `runs` row, or the `applications` row the handler updates;
  * every exception raised on this path is built from a literal.
"""

from __future__ import annotations

from jfl_core.crypto.envelope import MasterKey, unseal
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository


def load_api_key(repo: PostgresCredentialRepository, master_key: MasterKey) -> str | None:
    """This user's Anthropic key, for making one call with. None if they have
    not stored one -- a normal state, not an error, and the caller decides what
    to say about it.

    The repository is already bound to a user, so there is no user argument to
    get wrong: `repo.user_id` is both the row it reads and the AAD the
    ciphertext is checked against, which is what makes a row copied between
    users fail to authenticate rather than decrypt.

    Marks the credential used, which is the only trace a key use leaves.
    """
    sealed = repo.load_sealed(ANTHROPIC_API_KEY)
    if sealed is None:
        return None
    plaintext = unseal(master_key, sealed, user_id=repo.user_id, provider=ANTHROPIC_API_KEY)
    repo.mark_used(ANTHROPIC_API_KEY)
    return plaintext
