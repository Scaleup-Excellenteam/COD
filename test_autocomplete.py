"""Focused specification tests for autocomplete.py.

Run with: py -m unittest -v
"""

from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from autocomplete import (
    AutocompleteEngine,
    OneEditMatcher,
    normalize_text,
    remove_invalid_control_characters,
    run_cli,
)


class AutocompleteSpecificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.archive_path = Path(self.temporary_directory.name) / "corpus.zip"
        with zipfile.ZipFile(self.archive_path, "w") as archive:
            archive.writestr(
                "quotes/example.txt",
                "\n"
                "To be or not to be, that is the question.\n"
                "Abcdefghij.\n"
                "A, b    c!\n"
                "Alpha beta something.\n"
                "Alpha beta other.\n"
                "Something useful happens here.\n",
            )
            archive.writestr("second.txt", b"A different sentence.\nNoisy\x0c text.\n")
        self.engine = AutocompleteEngine.from_archive(self.archive_path)

    def tearDown(self) -> None:
        self.engine.close()
        self.temporary_directory.cleanup()

    def result_for(self, query: str):
        results = self.engine.get_best_k_completions(query)
        self.assertTrue(results, "Expected a result for query {0!r}".format(query))
        return results[0]

    def test_normalization_ignores_case_punctuation_and_extra_spaces(self) -> None:
        self.assertEqual(normalize_text(" BE,    THAT! "), "be that")
        self.assertEqual(remove_invalid_control_characters("A\fB"), "AB")
        result = self.result_for("be,       that")
        self.assertEqual(result.completed_sentence, "To be or not to be, that is the question.")
        self.assertEqual(result.score, 14)
        self.assertEqual(result.source_text, "quotes/example.txt")
        self.assertEqual(result.offset, 2)

    def test_original_sentence_and_physical_line_number_are_preserved(self) -> None:
        result = self.result_for("a b c")
        self.assertEqual(result.completed_sentence, "A, b    c!")
        self.assertEqual(result.source_text, "quotes/example.txt")
        self.assertEqual(result.offset, 4)

        result = self.result_for("noisy text")
        self.assertEqual(result.completed_sentence, "Noisy text.")
        self.assertEqual(result.offset, 2)

    def test_official_substitution_examples(self) -> None:
        self.assertEqual(self.result_for("2o be").score, 3)
        self.assertEqual(self.result_for("to pe").score, 6)

    def test_official_deletion_and_insertion_examples(self) -> None:
        self.assertEqual(self.result_for("or knot").score, 8)
        self.assertEqual(self.result_for("or nt").score, 8)

    def test_index_keeps_a_match_when_the_first_anchor_has_the_error(self) -> None:
        result = self.result_for("xomething useful")
        self.assertEqual(result.completed_sentence, "Something useful happens here.")
        self.assertEqual(result.score, 25)

    def test_index_keeps_a_match_when_the_last_anchor_has_the_error(self) -> None:
        result = self.result_for("something usefyl")
        self.assertEqual(result.completed_sentence, "Something useful happens here.")
        self.assertEqual(result.score, 29)

    def test_diagnostics_are_collected_from_the_same_search(self) -> None:
        results, diagnostics = self.engine.search_with_diagnostics("xomething useful")

        self.assertTrue(results)
        self.assertEqual(diagnostics.normalized_query, "xomething useful")
        self.assertEqual(diagnostics.search_path, "trigram-anchors")
        self.assertGreater(diagnostics.candidate_row_count, 0)
        self.assertEqual(diagnostics.result_count, len(results))
        self.assertGreaterEqual(diagnostics.total_ms, diagnostics.direct_lookup_ms)
        self.assertEqual(diagnostics.correction_operations["mode"], "candidate-checks")
        self.assertEqual(
            diagnostics.log_story[1],
            'Query split into trigrams for the direct lookup: "xom" -> "ome" -> "met" -> "eth" -> "thi" -> "hin" -> "ing" -> "ng " -> "g u" -> " us" -> "use" -> "sef" -> "efu" -> "ful".',
        )
        self.assertEqual(
            diagnostics.log_story[3],
            'Candidates containing "xomethin" or "g useful" were checked in this order: remove an extra character, replace one character, then add a missing character.',
        )
        self.assertTrue(diagnostics.selected_corrections)
        self.assertEqual(
            diagnostics.correction_trace["replace"][0]["pattern"], "?omething useful"
        )
        self.assertEqual(
            diagnostics.correction_trace["remove_extra"][0]["pattern"], "omething useful"
        )
        correction = diagnostics.selected_corrections[0]
        self.assertEqual(correction["operation"], "replace")
        self.assertEqual(correction["from_character"], "x")
        self.assertEqual(correction["to_character"], "s")
        self.assertEqual(correction["matched_text"], "something useful")

    def test_substitution_penalty_at_every_position(self) -> None:
        target = "abcdefghij"
        penalties = [5, 4, 3, 2, 1, 1, 1, 1, 1, 1]
        for index, penalty in enumerate(penalties):
            query = target[:index] + "z" + target[index + 1 :]
            with self.subTest(position=index + 1):
                self.assertEqual(self.result_for(query).score, 18 - penalty)

    def test_extra_character_penalty_at_every_position(self) -> None:
        target = "abcdefghij"
        penalties = [10, 8, 6, 4, 2, 2, 2, 2, 2, 2, 2]
        for index, penalty in enumerate(penalties):
            query = target[:index] + "z" + target[index:]
            with self.subTest(position=index + 1):
                self.assertEqual(self.result_for(query).score, 20 - penalty)

    def test_missing_character_penalty_at_internal_positions(self) -> None:
        target = "abcdefghij"
        penalties = [6, 4, 2, 2, 2, 2, 2]
        for index, penalty in zip(range(2, 9), penalties):
            query = target[:index] + target[index + 1 :]
            with self.subTest(position=index + 1):
                self.assertEqual(self.result_for(query).score, 18 - penalty)

    def test_ambiguous_match_uses_the_highest_legal_score(self) -> None:
        # Deleting 'b' may be repaired by inserting b at position 2 (score 10),
        # but it can also be repaired by replacing a with b against the suffix
        # "bcdefghij" (score 11). The engine must retain the better match.
        self.assertEqual(self.result_for("acdefghij").score, 11)

    def test_short_query_path_finds_every_generated_one_edit_match(self) -> None:
        """Exercise all edit types across short query lengths 2-5.

        This calls the public completion method, not an internal candidate
        filter, and proves that the direct short-query path preserves matches.
        """

        target = "abcdefghij"
        isolated_archive = Path(self.temporary_directory.name) / "short-cases.zip"
        with zipfile.ZipFile(isolated_archive, "w") as archive:
            archive.writestr("only.txt", "Abcdefghij.\n")
        isolated_engine = AutocompleteEngine.from_archive(isolated_archive)

        queries = set()
        for length in range(2, 6):
            same_length = target[:length]
            for index in range(length):
                queries.add(same_length[:index] + "z" + same_length[index + 1 :])

            shorter = target[: length - 1]
            for index in range(length):
                queries.add(shorter[:index] + "z" + shorter[index:])

            longer = target[: length + 1]
            for index in range(length + 1):
                queries.add(longer[:index] + longer[index + 1 :])

        try:
            for query in queries:
                with self.subTest(query=query):
                    results = isolated_engine.get_best_k_completions(query)
                    expected_score = OneEditMatcher(query).best_score(target)
                    is_returned = any(
                        result.completed_sentence == "Abcdefghij." for result in results
                    )
                    if expected_score is not None and expected_score >= 0:
                        self.assertTrue(is_returned)
                    else:
                        self.assertFalse(is_returned)
        finally:
            isolated_engine.close()

    def test_more_than_one_edit_is_not_a_match(self) -> None:
        self.assertEqual(self.engine.get_best_k_completions("abzdefghyj"), [])

    def test_equal_scores_are_sorted_alphabetically(self) -> None:
        results = self.engine.get_best_k_completions("alpha beta")
        sentences = [result.completed_sentence for result in results]
        self.assertEqual(
            sentences[:2], ["Alpha beta other.", "Alpha beta something."]
        )

    def test_empty_normalized_query_returns_no_suggestions(self) -> None:
        self.assertEqual(self.engine.get_best_k_completions("!!!"), [])

    def test_at_most_five_suggestions_are_returned(self) -> None:
        self.assertLessEqual(len(self.engine.get_best_k_completions("a")), 5)

    def test_negative_score_matches_are_not_returned(self) -> None:
        isolated_archive = Path(self.temporary_directory.name) / "negative-score.zip"
        with zipfile.ZipFile(isolated_archive, "w") as archive:
            archive.writestr("only.txt", "A.\n")
        isolated_engine = AutocompleteEngine.from_archive(isolated_archive)
        try:
            # Both inputs can be repaired only through a low-value one-edit
            # match whose score is negative. Neither may become a suggestion.
            self.assertEqual(isolated_engine.get_best_k_completions("z"), [])
            self.assertEqual(isolated_engine.get_best_k_completions("za"), [])
        finally:
            isolated_engine.close()

    def test_hash_resets_the_interactive_query(self) -> None:
        output = StringIO()
        with patch("builtins.input", side_effect=["alpha", "#", EOFError]), redirect_stdout(output):
            run_cli(self.engine)
        self.assertIn("Here are", output.getvalue())
        self.assertIn("The current query was reset.", output.getvalue())

    def test_persistent_index_is_loaded_when_archive_is_unchanged(self) -> None:
        index_path = Path(self.temporary_directory.name) / "saved-index.sqlite3"
        first_engine = AutocompleteEngine.from_archive(self.archive_path, index_path=index_path)
        first_engine.close()

        loaded_engine = AutocompleteEngine.from_archive(self.archive_path, index_path=index_path)
        try:
            self.assertTrue(index_path.is_file())
            self.assertEqual(loaded_engine.index_status, "loaded")
            self.assertTrue(loaded_engine.get_best_k_completions("to be"))
        finally:
            loaded_engine.close()

    def test_persistent_index_rebuilds_when_archive_changes(self) -> None:
        index_path = Path(self.temporary_directory.name) / "saved-index.sqlite3"
        first_engine = AutocompleteEngine.from_archive(self.archive_path, index_path=index_path)
        first_engine.close()

        with zipfile.ZipFile(self.archive_path, "w") as archive:
            archive.writestr("updated.txt", "A newly indexed sentence.\n")

        rebuilt_engine = AutocompleteEngine.from_archive(self.archive_path, index_path=index_path)
        try:
            self.assertEqual(rebuilt_engine.index_status, "rebuilt")
            self.assertTrue(rebuilt_engine.get_best_k_completions("newly indexed"))
        finally:
            rebuilt_engine.close()


if __name__ == "__main__":
    unittest.main()
