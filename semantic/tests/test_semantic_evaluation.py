"""Quota-free checks for reporting semantic retrieval evidence."""

import unittest

from semantic.evaluate_semantic import evaluate
from semantic.search import SemanticSearchResult


class FakeSemanticService:
    def __init__(self, results: list[SemanticSearchResult]) -> None:
        self.results = results
        self.queries = []

    def search(self, query: str) -> list[SemanticSearchResult]:
        self.queries.append(query)
        return self.results


class SemanticEvaluationTests(unittest.TestCase):
    def test_reports_the_rank_and_score_of_a_real_result_shape(self) -> None:
        relevant = SemanticSearchResult(
            sentence="Networking and",
            source_text="0130903086.txt",
            offset=16,
            semantic_score=0.83,
        )
        service = FakeSemanticService([
            SemanticSearchResult("Other", "source.txt", 2, 0.9), relevant
        ])

        evaluation = evaluate(
            service,
            "computer communication connections",
            "Networking and",
            "computer network communication",
        )

        self.assertEqual(service.queries, ["computer communication connections"])
        self.assertEqual(evaluation.relevant_rank, 2)
        self.assertEqual(evaluation.semantic_score, 0.83)
        self.assertEqual(evaluation.expected_idea, "computer network communication")

    def test_reports_an_absent_expected_sentence_without_a_score(self) -> None:
        service = FakeSemanticService([])

        evaluation = evaluate(service, "different wording", "Missing", "idea")

        self.assertIsNone(evaluation.relevant_rank)
        self.assertIsNone(evaluation.semantic_score)


if __name__ == "__main__":
    unittest.main()
