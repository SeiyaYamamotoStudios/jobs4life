"""Setting and reading a user's Anthropic API key.

The whole point of this module is the direction of travel: a key comes in from a
form, gets sealed, and is written. Nothing here returns it to a browser. The only
reader is `load_api_key`, which exists for the model calls slice B will make, and
its result goes to the Anthropic client and nowhere else.

**A key must never reach a log line, a `runs` row, a trace or an error message.**
That is not a habit to maintain, it is enforced by shape: the repository only
ever holds ciphertext, the plaintext exists as a local variable across a handful
of statements, and every error raised here is constructed from a literal.
"""

from __future__ import annotations

from jfl_core.crypto.envelope import MasterKey, seal, secret_hint, unseal
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository


class InvalidApiKeyError(ValueError):
    """The submitted key is unusable. The message never quotes the key."""


def normalise_submitted_key(raw: str) -> str:
    """Trim and sanity-check. Deliberately does not enforce an `sk-ant-` prefix:
    a prefix check would turn a change at Anthropic's end into an outage here,
    and the validation call below catches a genuine typo anyway.
    """
    key = raw.strip()
    if not key:
        raise InvalidApiKeyError("Enter an API key.")
    if any(c.isspace() for c in key):
        raise InvalidApiKeyError("That key contains whitespace -- check for a bad paste.")
    if len(key) < 16:
        raise InvalidApiKeyError("That does not look like a complete API key.")
    return key


def store_api_key(repo: PostgresCredentialRepository, master_key: MasterKey, plaintext: str) -> str:
    """Seal and store. Returns the four-character hint the UI may display.

    The repository is already bound to a user, so there is no user argument to
    get wrong: `repo.user_id` is both the row it writes and the AAD the
    ciphertext is bound to.
    """
    sealed = seal(master_key, plaintext, user_id=repo.user_id, provider=ANTHROPIC_API_KEY)
    hint = secret_hint(plaintext)
    repo.store(provider=ANTHROPIC_API_KEY, sealed=sealed, key_hint=hint)
    return hint


def load_api_key(repo: PostgresCredentialRepository, master_key: MasterKey) -> str | None:
    """The stored key, for making a model call with. Never for rendering.

    Marks the credential used, which is the only trace a key use leaves.
    """
    sealed = repo.load_sealed(ANTHROPIC_API_KEY)
    if sealed is None:
        return None
    plaintext = unseal(master_key, sealed, user_id=repo.user_id, provider=ANTHROPIC_API_KEY)
    repo.mark_used(ANTHROPIC_API_KEY)
    return plaintext


def validate_api_key(plaintext: str) -> None:
    """One call to the Models API to prove the key authenticates.

    `GET /v1/models` costs nothing -- no tokens in, no tokens out -- so a typo
    fails at the form instead of three screens later, and checking costs the user
    no money. It proves authentication, not quota.

    Raises `InvalidApiKeyError` only when Anthropic actually rejects the key. A
    network failure or a 5xx is our problem, not the user's: the key is stored
    anyway rather than blocking on our own outage.

    Constructing a client is what the root `conftest.py` guard blocks, so in
    tests this function is unreachable without both the `e2e` marker and
    `JFL_ALLOW_REAL_API=1`. Do not add a bypass.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=plaintext, max_retries=0, timeout=15.0)
    try:
        client.models.list(limit=1)
    except anthropic.AuthenticationError:
        # `from None`: never chain the SDK's exception. Its repr includes the
        # request that carried the key in a header.
        raise InvalidApiKeyError("Anthropic rejected that key.") from None
    except anthropic.PermissionDeniedError:
        raise InvalidApiKeyError("That key authenticates but has no access.") from None
    except anthropic.APIError:
        return  # our outage, not their key
