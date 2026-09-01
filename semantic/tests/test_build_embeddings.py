"""Phase 2 tests for the Earth-side embedding dataset builder."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from google.genai import types

from semantic.build_embeddings import (
    EmbeddingBuildError,
    build_embeddings,
    create_parser,
    validate_selected_records,
)


class EmbeddingCliTests(unittest.TestCase):
    def test_requires_explicit_limit_or_all(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            create_parser().parse_args([])

        self.assertEqual(raised.exception.code, 2)

    def test_limit_and_all_are_mutually_exclusive(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            create_parser().parse_args(["--limit", "3", "--all"])

        self.assertEqual(raised.exception.code, 2)

    def test_limit_must_be_positive(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            create_parser().parse_args(["--limit", "0"])

        self.assertEqual(raised.exception.code, 2)

    def test_rpm_must_be_positive(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            create_parser().parse_args(["--limit", "1", "--rpm", "0"])

        self.assertEqual(raised.exception.code, 2)


class InputPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.input_path = Path(self.temporary_directory.name) / "dataset.jsonl"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def valid_record(identifier: int) -> dict:
        return {
            "id": identifier,
            "sentence": "Sentence {0}".format(identifier),
            "source_text": "source.txt",
            "offset": identifier + 10,
        }

    def write_lines(self, lines) -> None:
        self.input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_limit_validates_and_counts_only_the_first_records(self) -> None:
        self.write_lines(
            [json.dumps(self.valid_record(index)) for index in range(3)]
            + ["not json"]
        )

        count = validate_selected_records(self.input_path, limit=3)

        self.assertEqual(count, 3)

    def test_all_rejects_malformed_json(self) -> None:
        self.write_lines([json.dumps(self.valid_record(1)), "not json"])

        with self.assertRaisesRegex(EmbeddingBuildError, "line 2"):
            validate_selected_records(self.input_path, limit=None)

    def test_rejects_non_object_missing_fields_and_blank_sentences(self) -> None:
        invalid_records = [
            [],
            {"id": 1, "sentence": "Sentence", "source_text": "source.txt"},
            {
                "id": 1,
                "sentence": "   ",
                "source_text": "source.txt",
                "offset": 3,
            },
        ]

        for invalid_record in invalid_records:
            with self.subTest(invalid_record=invalid_record):
                self.write_lines([json.dumps(invalid_record)])
                with self.assertRaises(EmbeddingBuildError):
                    validate_selected_records(self.input_path, limit=None)


class SequenceModels:
    def __init__(self) -> None:
        self.calls = []

    def embed_content(self, **arguments):
        self.calls.append(arguments)
        value = float(len(self.calls))
        return types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[value] * 768)]
        )


class SequenceClient:
    def __init__(self) -> None:
        self.models = SequenceModels()


class FailingModels(SequenceModels):
    def embed_content(self, **arguments):
        self.calls.append(arguments)
        if len(self.calls) == 2:
            raise RuntimeError("simulated Gemini failure")
        return types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[1.0] * 768)]
        )


class FailingClient:
    def __init__(self) -> None:
        self.models = FailingModels()


class EmbeddingBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "dataset.jsonl"
        self.output_path = self.root / "nested" / "embeddings.jsonl"
        self.records = [
            {
                "id": 7,
                "sentence": "Café déjà vu.",
                "source_text": "quotes/one.txt",
                "offset": 4,
            },
            {
                "id": 2,
                "sentence": "Second sentence.",
                "source_text": "books/two.txt",
                "offset": 9,
            },
            {
                "id": 11,
                "sentence": "Third sentence.",
                "source_text": "books/three.txt",
                "offset": 12,
            },
        ]
        self.input_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in self.records
            ),
            encoding="utf-8",
        )
        self.sleep = patch("semantic.build_embeddings.time.sleep").start()
        self.addCleanup(patch.stopall)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def read_output(self):
        return [
            json.loads(line)
            for line in self.output_path.read_text(encoding="utf-8").splitlines()
        ]

    def test_limit_preserves_mapping_order_and_uses_one_request_each(self) -> None:
        client = SequenceClient()
        progress = StringIO()

        count = build_embeddings(
            self.input_path,
            self.output_path,
            limit=2,
            client_factory=lambda: client,
            progress_stream=progress,
        )

        self.assertEqual(count, 2)
        self.assertEqual(len(client.models.calls), 2)
        output = self.read_output()
        self.assertEqual(
            [
                {key: value for key, value in record.items() if key != "embedding"}
                for record in output
            ],
            self.records[:2],
        )
        self.assertEqual(output[0]["embedding"], [1.0] * 768)
        self.assertEqual(output[1]["embedding"], [2.0] * 768)
        self.assertNotIn("title: none | text:", output[0]["sentence"])
        self.assertIn("Selected records: 2", progress.getvalue())
        self.assertIn("Rate limit: 90 requests/minute", progress.getvalue())
        self.assertIn("Embedding: 2 / 2", progress.getvalue())

    @patch("semantic.build_embeddings.time.monotonic", side_effect=[0.0, 0.0, 0.0, 0.2, 2 / 3])
    @patch("semantic.build_embeddings.time.sleep")
    def test_paces_requests_using_remaining_interval(self, sleep, _monotonic) -> None:
        client = SequenceClient()

        build_embeddings(
            self.input_path,
            self.output_path,
            limit=2,
            rpm=90,
            client_factory=lambda: client,
            progress_stream=StringIO(),
        )

        sleep.assert_called_once_with(0.4666666666666666)

    def test_all_writes_every_record_with_exact_keys_and_no_api_key(self) -> None:
        arguments = create_parser().parse_args(["--all"])
        client = SequenceClient()
        api_key = "phase-3-secret-sentinel"

        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            count = build_embeddings(
                self.input_path,
                self.output_path,
                limit=arguments.limit,
                client_factory=lambda: client,
                progress_stream=StringIO(),
            )

        output = self.read_output()
        self.assertTrue(arguments.all)
        self.assertEqual(count, len(self.records))
        self.assertEqual(len(client.models.calls), len(self.records))
        self.assertEqual(
            [record["id"] for record in output],
            [record["id"] for record in self.records],
        )
        for record in output:
            self.assertEqual(
                set(record),
                {"id", "sentence", "source_text", "offset", "embedding"},
            )
            self.assertEqual(len(record["embedding"]), 768)
        self.assertNotIn(api_key, self.output_path.read_text(encoding="utf-8"))

    def test_preflight_failure_happens_before_client_creation(self) -> None:
        self.input_path.write_text(
            json.dumps(self.records[0]) + "\nnot json\n",
            encoding="utf-8",
        )
        client_creations = []

        with self.assertRaises(EmbeddingBuildError):
            build_embeddings(
                self.input_path,
                self.output_path,
                limit=None,
                client_factory=lambda: client_creations.append(True),
                progress_stream=StringIO(),
            )

        self.assertEqual(client_creations, [])
        self.assertFalse(self.output_path.exists())

    def test_first_gemini_failure_stops_and_preserves_existing_output(self) -> None:
        self.output_path.parent.mkdir(parents=True)
        self.output_path.write_text("existing output\n", encoding="utf-8")
        client = FailingClient()

        with self.assertRaisesRegex(EmbeddingBuildError, "record 2"):
            build_embeddings(
                self.input_path,
                self.output_path,
                limit=3,
                client_factory=lambda: client,
                progress_stream=StringIO(),
            )

        self.assertEqual(len(client.models.calls), 2)
        self.assertEqual(
            self.output_path.read_text(encoding="utf-8"),
            "existing output\n",
        )
        self.assertEqual(
            list(self.output_path.parent.glob(".embeddings.jsonl.*.tmp")),
            [],
        )

if __name__ == "__main__":
    unittest.main()
