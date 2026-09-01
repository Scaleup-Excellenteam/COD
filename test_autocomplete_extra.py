"""Additional coverage for autocomplete.py.

This file is purely additive: it introduces new test cases and does not
change any existing test file or assertion. It targets code paths that
test_autocomplete.py does not exercise: text-normalization edge cases, the
penalty helper functions and OneEditMatcher in isolation, engine lifecycle
and error handling, ZIP/corpus edge cases (empty archive, non-text entries,
blank lines, directory entries), the interactive CLI loop, and a couple of
tests that document the engine's algorithmic/performance characteristics
(the FTS5 trigram index it relies on, and how query latency scales with
corpus size).

Run with: py -m unittest -v
"""

from pathlib import Path
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from io import StringIO
from unittest.mock import patch
import os
import random
import tempfile
import time
import unittest
import zipfile

import autocomplete
from autocomplete import (
    AutoCompleteData,
    AutocompleteEngine,
    OneEditMatcher,
    _insertion_or_deletion_penalty,
    _substitution_penalty,
    normalize_text,
    remove_invalid_control_characters,
    run_cli,
)


def write_archive(path: Path, files: dict) -> None:
    """Write a ZIP archive at ``path`` from a ``{name: text}`` mapping."""

    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


class NormalizationEdgeCaseTests(unittest.TestCase):
    """Cases for normalize_text/remove_invalid_control_characters not covered
    by test_autocomplete.py's single combined normalization test."""

    def test_normalize_text_collapses_mixed_whitespace_runs(self) -> None:
        self.assertEqual(normalize_text("a\t\tb\n\nc   d"), "a b c d")

    def test_normalize_text_keeps_unicode_letters(self) -> None:
        # isalnum() is true for accented letters, so they must survive
        # normalization rather than being treated as punctuation.
        self.assertEqual(normalize_text("Café Münchën!"), "café münchën")

    def test_normalize_text_keeps_digits(self) -> None:
        self.assertEqual(normalize_text("Room 101, now!"), "room 101 now")

    def test_normalize_text_all_punctuation_returns_empty_string(self) -> None:
        self.assertEqual(normalize_text("!!! ... ,,, ---"), "")

    def test_remove_invalid_control_characters_preserves_tabs(self) -> None:
        self.assertEqual(remove_invalid_control_characters("a\tb"), "a\tb")

    def test_remove_invalid_control_characters_strips_multiple_control_chars(self) -> None:
        self.assertEqual(remove_invalid_control_characters("a\x00b\x1fc\x7fd"), "abcd")


class PenaltyHelperFunctionTests(unittest.TestCase):
    """Direct unit tests of the private scoring tables, independent of the
    higher-level matching tests that only observe them indirectly."""

    def test_substitution_penalty_values_directly(self) -> None:
        self.assertEqual(
            [_substitution_penalty(position) for position in range(1, 8)],
            [5, 4, 3, 2, 1, 1, 1],
        )

    def test_insertion_or_deletion_penalty_values_directly(self) -> None:
        self.assertEqual(
            [_insertion_or_deletion_penalty(position) for position in range(1, 8)],
            [10, 8, 6, 4, 2, 2, 2],
        )


class OneEditMatcherDirectTests(unittest.TestCase):
    """Exercises OneEditMatcher directly, rather than only through the
    engine's public query API."""

    def test_exact_substring_scores_double_the_query_length(self) -> None:
        matcher = OneEditMatcher("banana")
        self.assertEqual(matcher.best_score("we ate a banana today"), 12)

    def test_returns_none_when_no_legal_one_edit_match_exists(self) -> None:
        matcher = OneEditMatcher("banana")
        self.assertIsNone(matcher.best_score("completely different text"))

    def test_single_character_query_matches_almost_any_text_via_deletion(self) -> None:
        # A one-character query's "deleted" variant is the empty string,
        # which is a substring of everything. This documents that very
        # short queries are inherently low-information/near-universal, and
        # that the (negative) score is still computed deterministically.
        matcher = OneEditMatcher("x")
        self.assertEqual(matcher.best_score("any sentence at all"), -5)


class DataclassContractTests(unittest.TestCase):
    def test_format_for_cli_contains_all_fields(self) -> None:
        result = AutoCompleteData(
            completed_sentence="Hello world.",
            source_text="a.txt",
            offset=3,
            score=10,
        )
        self.assertEqual(result.format_for_cli(1), "1. Hello world. (a.txt:3, score=10)")

    def test_autocomplete_data_is_immutable(self) -> None:
        result = AutoCompleteData("s", "f.txt", 1, 10)
        with self.assertRaises(FrozenInstanceError):
            result.score = 99  # type: ignore[misc]


class EngineErrorHandlingTests(unittest.TestCase):
    def test_raises_file_not_found_for_missing_archive(self) -> None:
        engine = AutocompleteEngine(Path("/nonexistent-directory/no-such-archive.zip"))
        try:
            with self.assertRaises(FileNotFoundError):
                engine.build()
        finally:
            engine.close()

    def test_db_property_raises_before_build_is_called(self) -> None:
        engine = AutocompleteEngine(Path("/nonexistent-directory/no-such-archive.zip"))
        try:
            with self.assertRaises(RuntimeError):
                engine._db  # noqa: B018 - intentionally accessing the guarded property
        finally:
            engine.close()

    def test_get_best_k_completions_raises_before_initialize(self) -> None:
        autocomplete._default_engine = None
        with self.assertRaises(RuntimeError):
            autocomplete.get_best_k_completions("anything")


class EngineLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.archive_path = Path(self.temporary_directory.name) / "corpus.zip"
        write_archive(self.archive_path, {"a.txt": "Hello world.\n"})

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_build_is_idempotent_once_already_built(self) -> None:
        engine = AutocompleteEngine.from_archive(self.archive_path)
        try:
            connection_before = engine._connection
            engine.build()
            self.assertIs(engine._connection, connection_before)
        finally:
            engine.close()

    def test_close_is_safe_to_call_more_than_once(self) -> None:
        engine = AutocompleteEngine.from_archive(self.archive_path)
        engine.close()
        engine.close()  # must not raise

    def test_engine_as_context_manager_cleans_up_temporary_directory(self) -> None:
        with AutocompleteEngine(self.archive_path) as engine:
            self.assertEqual(
                engine.get_best_k_completions("hello world")[0].completed_sentence,
                "Hello world.",
            )
            temporary_directory = engine._temporary_directory.name
        self.assertFalse(os.path.exists(temporary_directory))

    def test_initialize_closes_and_replaces_the_previous_default_engine(self) -> None:
        first = autocomplete.initialize(self.archive_path)
        first_temporary_directory = first._temporary_directory.name
        try:
            second = autocomplete.initialize(self.archive_path)
            self.assertFalse(os.path.exists(first_temporary_directory))
            self.assertIs(autocomplete._default_engine, second)
        finally:
            autocomplete._default_engine.close()
            autocomplete._default_engine = None


class CorpusIndexingEdgeCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_non_txt_and_directory_entries_are_ignored(self) -> None:
        archive_path = Path(self.temporary_directory.name) / "mixed.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("folder/", "")
            archive.writestr("notes.md", "Hello world in markdown.\n")
            archive.writestr("a.txt", "Hello world.\n")
        engine = AutocompleteEngine.from_archive(archive_path)
        try:
            self.assertEqual(engine.file_count, 1)
            results = engine.get_best_k_completions("hello world")
            self.assertEqual([r.source_text for r in results], ["a.txt"])
        finally:
            engine.close()

    def test_blank_lines_are_not_indexed_but_offsets_stay_physical(self) -> None:
        archive_path = Path(self.temporary_directory.name) / "blank.zip"
        write_archive(archive_path, {"a.txt": "   \n\nHello world.\n"})
        engine = AutocompleteEngine.from_archive(archive_path)
        try:
            self.assertEqual(engine.indexed_sentence_count, 1)
            result = engine.get_best_k_completions("hello world")[0]
            self.assertEqual(result.offset, 3)
        finally:
            engine.close()

    def test_file_count_counts_only_txt_files(self) -> None:
        archive_path = Path(self.temporary_directory.name) / "counts.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("a.txt", "One.\n")
            archive.writestr("b.TXT", "Two.\n")
            archive.writestr("c.csv", "not,counted\n")
        engine = AutocompleteEngine.from_archive(archive_path)
        try:
            self.assertEqual(engine.file_count, 2)
        finally:
            engine.close()

    def test_empty_archive_produces_no_results(self) -> None:
        archive_path = Path(self.temporary_directory.name) / "empty.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("readme.md", "There are no text files here.\n")
        engine = AutocompleteEngine.from_archive(archive_path)
        try:
            self.assertEqual(engine.file_count, 0)
            self.assertEqual(engine.indexed_sentence_count, 0)
            self.assertEqual(engine.get_best_k_completions("anything"), [])
        finally:
            engine.close()

    def test_source_text_breaks_ties_alphabetically_across_files(self) -> None:
        archive_path = Path(self.temporary_directory.name) / "tie.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("b.txt", "Hello world.\n")
            archive.writestr("a.txt", "Hello world.\n")
        engine = AutocompleteEngine.from_archive(archive_path)
        try:
            results = engine.get_best_k_completions("hello world")
            self.assertEqual([r.source_text for r in results], ["a.txt", "b.txt"])
        finally:
            engine.close()

    def test_single_character_query_still_returns_at_most_five_results(self) -> None:
        archive_path = Path(self.temporary_directory.name) / "many.zip"
        lines = "\n".join("Line number {0}.".format(i) for i in range(20))
        write_archive(archive_path, {"a.txt": lines})
        engine = AutocompleteEngine.from_archive(archive_path)
        try:
            results = engine.get_best_k_completions("z")
            self.assertLessEqual(len(results), 5)
        finally:
            engine.close()


class RunCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.archive_path = Path(self.temporary_directory.name) / "corpus.zip"
        write_archive(
            self.archive_path,
            {"a.txt": "Hello world.\nHello there.\nGoodbye world.\n"},
        )
        self.engine = AutocompleteEngine.from_archive(self.archive_path)

    def tearDown(self) -> None:
        self.engine.close()
        self.temporary_directory.cleanup()

    def test_run_cli_prints_formatted_suggestion_lines(self) -> None:
        output = StringIO()
        with patch("builtins.input", side_effect=["Hello", EOFError]), redirect_stdout(output):
            run_cli(self.engine)
        printed = output.getvalue()
        self.assertIn("1. Hello there. (a.txt:2, score=10)", printed)
        self.assertIn("2. Hello world. (a.txt:1, score=10)", printed)

    def test_run_cli_handles_keyboard_interrupt_gracefully(self) -> None:
        output = StringIO()
        with patch("builtins.input", side_effect=KeyboardInterrupt), redirect_stdout(output):
            run_cli(self.engine)  # must not raise
        self.assertIn("Goodbye.", output.getvalue())


class ComplexityAndPerformanceTests(unittest.TestCase):
    """Tests that check *how* the engine achieves its speed, not just that
    it returns correct results, per the project's own documented design
    (README: "Why queries are fast")."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_build_creates_an_fts5_trigram_index_for_fast_lookup(self) -> None:
        archive_path = Path(self.temporary_directory.name) / "corpus.zip"
        write_archive(archive_path, {"a.txt": "Hello world.\n"})
        engine = AutocompleteEngine.from_archive(archive_path)
        try:
            row = engine._connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'sentence_search'"
            ).fetchone()
            self.assertIsNotNone(row, "Expected a sentence_search virtual table")
            definition = row[0].lower()
            self.assertIn("fts5", definition)
            self.assertIn("trigram", definition)
        finally:
            engine.close()

    def test_query_latency_scales_sublinearly_with_corpus_size(self) -> None:
        """A 20x larger corpus should not make a matching query ~20x slower.

        The engine is documented to answer queries via a trigram/anchor
        index rather than a full-corpus scan (see README, "Why queries are
        fast"). This test builds a small and a 20x larger corpus, each with
        exactly one line that can match the query, and checks that per-query
        latency grows much more slowly than corpus size -- a coarse, non-
        flaky proxy for "not a linear scan". Absolute timings are not
        asserted, only their ratio, and a generous multiplier is used to
        keep the check robust on slow/shared machines.
        """

        def build_corpus(path: Path, filler_line_count: int) -> None:
            random.seed(1234)  # deterministic filler, independent per call
            vocabulary = [
                "apple", "river", "stone", "cloud", "forest", "engine",
                "market", "signal", "planet", "garden", "window", "bridge",
            ]
            lines = [
                " ".join(random.sample(vocabulary, 5)) + " item {0}.".format(i)
                for i in range(filler_line_count)
            ]
            lines.append("The quick brown fox jumps over the lazy dog today.")
            write_archive(path, {"a.txt": "\n".join(lines)})

        small_path = Path(self.temporary_directory.name) / "small.zip"
        large_path = Path(self.temporary_directory.name) / "large.zip"
        build_corpus(small_path, 1500)
        build_corpus(large_path, 30000)

        query = "quick brown fox jumps"

        def average_query_seconds(archive_path: Path) -> float:
            engine = AutocompleteEngine.from_archive(archive_path)
            try:
                engine.get_best_k_completions(query)  # warm-up
                started_at = time.perf_counter()
                repetitions = 10
                for _ in range(repetitions):
                    results = engine.get_best_k_completions(query)
                elapsed = time.perf_counter() - started_at
                self.assertEqual(len(results), 1)
                return elapsed / repetitions
            finally:
                engine.close()

        small_seconds = average_query_seconds(small_path)
        large_seconds = average_query_seconds(large_path)

        # Corpus grew 20x; a linear scan would grow query time ~20x too.
        # Allow generous slack (10x) so the assertion only fails on
        # genuinely linear-or-worse scaling, not machine noise.
        self.assertLess(
            large_seconds,
            max(small_seconds, 1e-6) * 10,
            "Query latency grew roughly as fast as corpus size ({0:.6f}s -> "
            "{1:.6f}s for a 20x larger corpus); expected sublinear scaling "
            "from the FTS index.".format(small_seconds, large_seconds),
        )


if __name__ == "__main__":
    unittest.main()
