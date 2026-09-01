"""Focused tests for local ground-to-satellite semantic orchestration."""

import unittest
from unittest.mock import patch

from semantic.search import SemanticSearchResult
from semantic.semantic_query import SemanticQueryService, main


class FakeSemanticEngine:
    def __init__(self) -> None:
        self.vectors = []
        self.results = [
            SemanticSearchResult(
                sentence="A relevant original sentence.",
                source_text="corpus/source.txt",
                offset=42,
                semantic_score=0.91,
            )
        ]

    def search(self, vector):
        self.vectors.append(vector)
        return self.results


class SemanticQueryServiceTests(unittest.TestCase):
    def test_ground_embeds_text_then_satellite_searches_that_vector(self) -> None:
        engine = FakeSemanticEngine()
        client = object()
        vector = [0.5] * 768

        with patch("semantic.semantic_query.embed_query", return_value=vector) as embed:
            results = SemanticQueryService(engine, client).search("different words")

        embed.assert_called_once_with(client, "different words")
        self.assertEqual(engine.vectors, [vector])
        self.assertEqual(results, engine.results)
        self.assertFalse(hasattr(results[0], "score"))


class SemanticQueryCliTests(unittest.TestCase):
    def test_cli_loads_artifacts_and_prints_semantic_result_fields(self) -> None:
        engine = FakeSemanticEngine()
        client = object()
        with patch("semantic.semantic_query.create_client", return_value=client), patch(
            "semantic.semantic_query.SemanticSearchEngine.from_files",
            return_value=engine,
        ) as load, patch(
            "semantic.semantic_query.embed_query", return_value=[1.0] * 768
        ), patch("builtins.print") as output:
            exit_code = main([
                "how do computers learn?",
                "--index", "demo.faiss",
                "--metadata", "demo_metadata.jsonl",
            ])

        self.assertEqual(exit_code, 0)
        load.assert_called_once()
        rendered = "\n".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("1. A relevant original sentence.", rendered)
        self.assertIn("source: corpus/source.txt", rendered)
        self.assertIn("offset: 42", rendered)
        self.assertIn("semantic similarity: 0.910000", rendered)
        self.assertNotIn("[1.0", rendered)


if __name__ == "__main__":
    unittest.main()
