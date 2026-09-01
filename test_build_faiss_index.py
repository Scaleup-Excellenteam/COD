"""Focused tests for the Earth-side FAISS deployment-artifact builder."""

from pathlib import Path
import json
import tempfile
import unittest

import faiss
import numpy as np

from semantic.build_faiss_index import (
    EMBEDDING_DIMENSIONS,
    FaissIndexBuildError,
    build_faiss_index,
)


class FaissIndexBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "embeddings.jsonl"
        self.index_path = self.root / "out" / "semantic.faiss"
        self.metadata_path = self.root / "out" / "semantic_metadata.jsonl"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def record(identifier: int, vector: list[float]) -> dict:
        return {
            "id": identifier,
            "sentence": "Sentence {0}.".format(identifier),
            "source_text": "source-{0}.txt".format(identifier),
            "offset": identifier,
            "embedding": vector,
        }

    def write_records(self, records) -> None:
        self.input_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_builds_normalized_exact_index_with_aligned_metadata(self) -> None:
        records = [
            self.record(2, [2.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)),
            self.record(1, [0.0, 3.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)),
        ]
        self.write_records(records)

        summary = build_faiss_index(
            self.input_path, self.index_path, self.metadata_path
        )

        index = faiss.read_index(str(self.index_path))
        metadata = [
            json.loads(line)
            for line in self.metadata_path.read_text(encoding="utf-8").splitlines()
        ]
        reconstructed = np.vstack([index.reconstruct(row) for row in range(2)])
        self.assertEqual(summary.vectors_indexed, 2)
        self.assertEqual(summary.dimensions, EMBEDDING_DIMENSIONS)
        self.assertEqual(summary.index_type, "IndexFlatIP")
        self.assertEqual(index.ntotal, len(metadata))
        self.assertEqual([record["id"] for record in metadata], [2, 1])
        self.assertTrue(all("embedding" not in record for record in metadata))
        self.assertEqual(reconstructed.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(reconstructed, axis=1), [1.0, 1.0])

    def test_rejects_invalid_vectors_and_preserves_existing_outputs(self) -> None:
        invalid_vectors = [
            [0.0] * (EMBEDDING_DIMENSIONS - 1),
            [0.0] * (EMBEDDING_DIMENSIONS - 1) + [True],
            [0.0] * (EMBEDDING_DIMENSIONS - 1) + ["not numeric"],
            [0.0] * EMBEDDING_DIMENSIONS,
        ]
        for vector in invalid_vectors:
            with self.subTest(vector_tail=vector[-1]):
                self.write_records([self.record(1, vector)])
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                self.index_path.write_bytes(b"existing index")
                self.metadata_path.write_text("existing metadata\n", encoding="utf-8")

                with self.assertRaises(FaissIndexBuildError):
                    build_faiss_index(
                        self.input_path, self.index_path, self.metadata_path
                    )

                self.assertEqual(self.index_path.read_bytes(), b"existing index")
                self.assertEqual(
                    self.metadata_path.read_text(encoding="utf-8"),
                    "existing metadata\n",
                )

    def test_rejects_malformed_records(self) -> None:
        malformed = self.record(1, [1.0] * EMBEDDING_DIMENSIONS)
        del malformed["source_text"]
        self.write_records([malformed])

        with self.assertRaisesRegex(FaissIndexBuildError, "source_text"):
            build_faiss_index(self.input_path, self.index_path, self.metadata_path)


if __name__ == "__main__":
    unittest.main()
