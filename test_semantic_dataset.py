"""Focused tests for the semantic dataset and Gemini configuration."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from semantic import config
from semantic.build_dataset import DatasetBuildError, build_dataset, create_parser


class SemanticDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.index_path = self.root / "index.sqlite3"
        connection = sqlite3.connect(self.index_path)
        connection.execute(
            """
            CREATE TABLE sentences (
                id INTEGER PRIMARY KEY,
                normalized TEXT NOT NULL,
                completed_sentence TEXT NOT NULL,
                source_text TEXT NOT NULL,
                offset INTEGER NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO sentences(
                id, normalized, completed_sentence, source_text, offset
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (3, "third sentence", "Third sentence.", "third.txt", 9),
                (1, "café déjà vu", "Café,  déjà vu!", "quotes/example.txt", 2),
                (2, "second sentence", "Second sentence.", "source.txt", 7),
            ],
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def read_records(path: Path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def test_export_preserves_original_fields_and_orders_by_id(self) -> None:
        output_path = self.root / "dataset.jsonl"

        count = build_dataset(self.index_path, output_path)

        self.assertEqual(count, 3)
        self.assertEqual(
            self.read_records(output_path),
            [
                {
                    "id": 1,
                    "sentence": "Café,  déjà vu!",
                    "source_text": "quotes/example.txt",
                    "offset": 2,
                },
                {
                    "id": 2,
                    "sentence": "Second sentence.",
                    "source_text": "source.txt",
                    "offset": 7,
                },
                {
                    "id": 3,
                    "sentence": "Third sentence.",
                    "source_text": "third.txt",
                    "offset": 9,
                },
            ],
        )

    def test_limit_selects_first_ids_and_creates_output_directory(self) -> None:
        output_path = self.root / "nested" / "dataset.jsonl"

        count = build_dataset(self.index_path, output_path, limit=2)

        self.assertEqual(count, 2)
        self.assertEqual(
            [record["id"] for record in self.read_records(output_path)], [1, 2]
        )

    def test_invalid_index_does_not_replace_existing_output(self) -> None:
        invalid_index = self.root / "invalid.sqlite3"
        connection = sqlite3.connect(invalid_index)
        connection.execute("CREATE TABLE sentences (id INTEGER PRIMARY KEY)")
        connection.close()
        output_path = self.root / "dataset.jsonl"
        output_path.write_text("existing output\n", encoding="utf-8")

        with self.assertRaises(DatasetBuildError):
            build_dataset(invalid_index, output_path)

        self.assertEqual(
            output_path.read_text(encoding="utf-8"), "existing output\n"
        )

    def test_cli_rejects_a_non_positive_limit(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            create_parser().parse_args(["--limit", "0"])

        self.assertEqual(raised.exception.code, 2)


class GeminiConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.env_file = Path(self.temporary_directory.name) / ".env"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_process_environment_takes_precedence_over_env_file(self) -> None:
        self.env_file.write_text("GEMINI_API_KEY=file-key\n", encoding="utf-8")

        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "environment-key"}, clear=True
        ), patch.object(config, "ENV_FILE", self.env_file):
            self.assertEqual(config.get_gemini_api_key(), "environment-key")

    def test_env_file_is_used_when_process_environment_is_missing(self) -> None:
        self.env_file.write_text(
            'GEMINI_API_KEY="file-key"\n', encoding="utf-8"
        )

        with patch.dict(os.environ, {}, clear=True), patch.object(
            config, "ENV_FILE", self.env_file
        ):
            self.assertEqual(config.get_gemini_api_key(), "file-key")

    def test_missing_or_blank_key_raises_a_safe_error(self) -> None:
        self.env_file.write_text("GEMINI_API_KEY=\n", encoding="utf-8")

        with patch.dict(os.environ, {}, clear=True), patch.object(
            config, "ENV_FILE", self.env_file
        ), self.assertRaisesRegex(
            RuntimeError, "GEMINI_API_KEY is not configured"
        ):
            config.get_gemini_api_key()


if __name__ == "__main__":
    unittest.main()
