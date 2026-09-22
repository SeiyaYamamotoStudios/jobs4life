"""The web layer's handle on the one corpus write path.

`jfl_core.corpus_source` is that path -- markdown in `documents.text`, re-parsed
into ordinary `provenance='document'` spans, one document per user. This module
adds nothing to it and implements none of it again. It exists because the
functions there take a `Connection` and a `user_id`, and a route that can name a
user is a route that can name the wrong one: a tenancy-scoped repository is
constructed with the one user it may act on and offers no per-call override
(`jfl_core.storage.tenancy`, and the rule `tests/test_tenancy_enforcement.py`
enforces). So the web layer gets this, and never a raw connection.

The flow behind it is profile questions 15 and 16 -- where your depth is
genuine, and the gaps that keep coming up. Unlike every other answer on that
page those are claims about the person rather than preferences, so they become
corpus text, by exactly the route a confirmed CV fact takes and deliberately not
a second one: two mechanisms for one kind of fact is how one sentence ends up
with two span ids that the claim gate reads as two pieces of evidence.

**No model call anywhere on this path**, and the text is stored byte for byte
aside from collapsing embedded newlines so one statement is one span. A model
tidying a user's sentence into a neater corpus fact is the ratchet in miniature:
the user is then held to wording they did not choose, by a tool whose whole
claim is that it measures distance from what they actually said.

**Spans are retired, never deleted.** A statement the user supersedes or clears
stops grounding anything (retired spans are excluded from `all_spans`) but its
row stays, so a citation recorded against it still resolves.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from jfl_core.corpus_source import append_confirmed_fact as _append_confirmed_fact
from jfl_core.corpus_source import replace_section as _replace_section
from jfl_core.storage.tenancy import TenantScopedRepository


class PostgresUserCorpusRepository(TenantScopedRepository):
    """This user's hand-written corpus statements, and no one else's."""

    def replace_section(self, section: str, texts: Sequence[str]) -> list[uuid.UUID]:
        """Make `texts` exactly what this section of the corpus holds, retiring
        whatever else was live there. Returns the surviving span ids, in the
        order given.

        This is what a re-answered profile question needs: the user's newer
        words replace their older ones and the older ones stop grounding
        anything. Passing an empty sequence, or a single blank string, clears
        the section -- an answer the user deleted must not go on being cited at
        them.
        """
        return _replace_section(self._conn, self._user_id, texts, section=section)

    def add_statement(self, section: str, text: str) -> uuid.UUID:
        """Append one statement in the user's own words and return its span.

        The same call `PostgresCandidateFactRepository.confirm` makes, exposed
        here for the other place a person answers a question about themselves in
        their own words: the evidence question a capability pushback opens.
        Deliberately the existing path and not a second one -- two mechanisms
        for one kind of fact is how one sentence ends up with two span ids that
        the claim gate reads as two pieces of evidence.

        Appending rather than replacing, unlike `replace_section`: these
        answers accumulate, and an earlier answer to an earlier question is not
        superseded by a later answer to a different one. Idempotent on the text,
        so answering twice with the same words yields one span.
        """
        return _append_confirmed_fact(self._conn, self._user_id, text, section=section)


__all__ = ["PostgresUserCorpusRepository"]
