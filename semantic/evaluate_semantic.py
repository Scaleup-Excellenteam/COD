"""Small manual evaluator for evidence of semantic retrieval quality.

This is intentionally separate from automated tests: it sends query text to
Gemini and therefore must only be run deliberately against real deployment
artifacts.  It never prints query vectors or credentials.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from semantic.gemini_embeddings import create_client
from semantic.search import SemanticSearchEngine, SemanticSearchResult
from semantic.semantic_query import (
    DEFAULT_INDEX_PATH,
    DEFAULT_METADATA_PATH,
    SemanticQueryService,
)


@dataclass(frozen=True)
class EvaluationResult:
    query: str
    expected_idea: str
    results: list[SemanticSearchResult]
    relevant_rank: int | None
    semantic_score: float | None


def evaluate(
    service: SemanticQueryService,
    query: str,
    expected_sentence: str,
    expected_idea: str,
) -> EvaluationResult:
    """Run one real semantic query and report its matching rank, if present."""

    results = service.search(query)
    relevant_rank = next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if result.sentence == expected_sentence
        ),
        None,
    )
    semantic_score = (
        results[relevant_rank - 1].semantic_score
        if relevant_rank is not None
        else None
    )
    return EvaluationResult(
        query=query,
        expected_idea=expected_idea,
        results=results,
        relevant_rank=relevant_rank,
        semantic_score=semantic_score,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one paraphrased semantic retrieval query."
    )
    parser.add_argument("query")
    parser.add_argument("--expected-sentence", required=True)
    parser.add_argument("--expected-idea", required=True)
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    service = SemanticQueryService(
        SemanticSearchEngine.from_files(arguments.index, arguments.metadata),
        create_client(),
    )
    evaluation = evaluate(
        service,
        arguments.query,
        arguments.expected_sentence,
        arguments.expected_idea,
    )
    print("Query: {0}".format(evaluation.query))
    print("Expected relevant corpus idea: {0}".format(evaluation.expected_idea))
    for rank, result in enumerate(evaluation.results, start=1):
        print("{0}. {1}".format(rank, result.sentence))
        print("   source: {0}".format(result.source_text))
        print("   offset: {0}".format(result.offset))
        print("   semantic score: {0:.6f}".format(result.semantic_score))
    print(
        "Rank of relevant result: {0}".format(
            evaluation.relevant_rank if evaluation.relevant_rank is not None else "not in Top 5"
        )
    )
    if evaluation.semantic_score is not None:
        print("Relevant semantic score: {0:.6f}".format(evaluation.semantic_score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
