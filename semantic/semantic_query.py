"""Ground-to-satellite semantic query orchestration.

The ground turns query text into a Gemini vector.  The satellite boundary is
the ``SemanticSearchEngine.search(vector)`` call, which performs only local
FAISS retrieval.  This module runs both sides in one process solely for the
local demonstration CLI; it does not implement networking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from semantic.gemini_embeddings import create_client, embed_query
from semantic.search import SemanticSearchEngine, SemanticSearchResult


DEFAULT_INDEX_PATH = Path("data/semantic.faiss")
DEFAULT_METADATA_PATH = Path("data/semantic_metadata.jsonl")


class SemanticQueryService:
    """Ground semantic service that delegates all vector search to satellite."""

    def __init__(self, semantic_engine: SemanticSearchEngine, gemini_client: Any):
        self._semantic_engine = semantic_engine
        self._gemini_client = gemini_client

    def search(self, query: str) -> list[SemanticSearchResult]:
        """Embed text on the ground, then retrieve local satellite Top 5."""

        query_embedding = embed_query(self._gemini_client, query)
        return self._semantic_engine.search(query_embedding)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local ground-to-satellite semantic search demo."
    )
    parser.add_argument("query", help="Text to embed on the ground.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    # Ground responsibility: text -> Gemini 768D vector.
    client = create_client()
    # Satellite responsibility: vector -> local FAISS Top 5.
    engine = SemanticSearchEngine.from_files(arguments.index, arguments.metadata)
    results = SemanticQueryService(engine, client).search(arguments.query)

    for rank, result in enumerate(results, start=1):
        print("{0}. {1}".format(rank, result.sentence))
        print("   source: {0}".format(result.source_text))
        print("   offset: {0}".format(result.offset))
        print("   semantic similarity: {0:.6f}".format(result.semantic_score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
