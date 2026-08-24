"""Embedding interface. GPU and CPU implementations sit behind it unchanged.

Both implementations use the same model, so the pgvector column keeps a single
fixed width; only the device differs. bge-* wants an instruction prefix on the
query side but not the document side, which is why these are two methods.
"""

from __future__ import annotations

from typing import Protocol

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    model: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
