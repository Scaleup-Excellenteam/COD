"""Focused tests for satellite-local FAISS semantic retrieval."""

from pathlib import Path
import json
import tempfile
import unittest

import faiss
import numpy as np

from semantic.search import (
    EMBEDDING_DIMENSIONS,
    SemanticQueryError,
    SemanticSearchEngine,
    SemanticSearchError,
)


class SemanticSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.index_path = self.root / "semantic.faiss"
        self.metadata_path = self.root / "semantic_metadata.jsonl"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def vector(first: float = 0.0, second: float = 0.0) -> list[float]:
        return [first, second] + [0.0] * (EMBEDDING_DIMENSIONS - 2)

    def write_artifacts(self, vectors: list[list[float]]) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
        index.add(matrix)
        faiss.write_index(index, str(self.index_path))
        self.metadata_path.write_text(
            "".join(
                json.dumps(
                    {
                        "id": position + 10,
                        "sentence": "Original sentence {0}.".format(position),
                        "source_text": "source/{0}.txt".format(position),
                        "offset": position + 2,
                    }
                )
                + "\n"
                for position in range(len(vectors))
            ),
            encoding="utf-8",
        )

    def test_cosine_ranking_preserves_metadata_and_semantic_score(self) -> None:
        self.write_artifacts(
            [self.vector(1, 0), self.vector(0.8, 0.6), self.vector(0, 1)]
        )
        engine = SemanticSearchEngine.from_files(
            self.index_path, self.metadata_path
        )

        results = engine.search(self.vector(9, 0))

        self.assertEqual([result.sentence for result in results], [
            "Original sentence 0.",
            "Original sentence 1.",
            "Original sentence 2.",
        ])
        self.assertEqual(results[0].source_text, "source/0.txt")
        self.assertEqual(results[0].offset, 2)
        self.assertAlmostEqual(results[0].semantic_score, 1.0, places=6)
        self.assertAlmostEqual(results[1].semantic_score, 0.8, places=6)

    def test_returns_at_most_five_and_handles_a_smaller_corpus(self) -> None:
        self.write_artifacts(
            [self.vector(1, position / 10) for position in range(6)]
        )
        engine = SemanticSearchEngine.from_files(
            self.index_path, self.metadata_path
        )
        self.assertEqual(len(engine.search(self.vector(1, 0))), 5)

        self.write_artifacts([self.vector(1, 0), self.vector(0, 1)])
        smaller_engine = SemanticSearchEngine.from_files(
            self.index_path, self.metadata_path
        )
        self.assertEqual(len(smaller_engine.search(self.vector(1, 0))), 2)

    def test_rejects_malformed_and_zero_query_vectors(self) -> None:
        self.write_artifacts([self.vector(1, 0)])
        engine = SemanticSearchEngine.from_files(
            self.index_path, self.metadata_path
        )
        invalid_queries = [
            [1.0] * (EMBEDDING_DIMENSIONS - 1),
            [1.0] * (EMBEDDING_DIMENSIONS - 1) + [True],
            [0.0] * EMBEDDING_DIMENSIONS,
            [1.0] * (EMBEDDING_DIMENSIONS - 1) + [float("nan")],
        ]
        for query in invalid_queries:
            with self.subTest(query_tail=query[-1]), self.assertRaises(
                SemanticQueryError
            ):
                engine.search(query)

    def test_rejects_count_mismatch_and_corrupted_metadata(self) -> None:
        self.write_artifacts([self.vector(1, 0), self.vector(0, 1)])
        self.metadata_path.write_text(
            self.metadata_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SemanticSearchError, "2 vectors.*1 records"):
            SemanticSearchEngine.from_files(self.index_path, self.metadata_path)

        self.metadata_path.write_text("not json\n", encoding="utf-8")
        with self.assertRaisesRegex(SemanticSearchError, "Metadata line 1"):
            SemanticSearchEngine.from_files(self.index_path, self.metadata_path)

    def test_rejects_wrong_index_dimension(self) -> None:
        index = faiss.IndexFlatIP(3)
        faiss.write_index(index, str(self.index_path))
        self.metadata_path.write_text("", encoding="utf-8")

        with self.assertRaisesRegex(SemanticSearchError, "dimension is 3"):
            SemanticSearchEngine.from_files(self.index_path, self.metadata_path)


if __name__ == "__main__":
    unittest.main()
