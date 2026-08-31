"""Fast, specification-compliant autocomplete over a ZIP text corpus.

The offline phase creates a temporary SQLite database.  SQLite's built-in FTS5
trigram index is used only to find a small, safe set of *candidates*.  Every
candidate is then checked by :class:`OneEditMatcher`, which is the source of
truth for the assignment's matching and scoring rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple
import argparse
import re
import sqlite3
import string
import tempfile
import time
import zipfile


K = 5
NORMALIZED_CORPUS_ALPHABET = string.ascii_lowercase + string.digits + " "


@dataclass(frozen=True)
class AutoCompleteData:
    """One result returned by ``get_best_k_completions``.

    ``source_text`` is the original relative path of the text file inside the
    archive.  ``offset`` is its one-based line number, as required by the
    assignment.
    """

    completed_sentence: str
    source_text: str
    offset: int
    score: int

    def format_for_cli(self, number: int) -> str:
        return (
            "{0}. {1} ({2}:{3}, score={4})".format(
                number,
                self.completed_sentence,
                self.source_text,
                self.offset,
                self.score,
            )
        )


@dataclass(frozen=True)
class SentenceData:
    """The data retained for one physical line in a source text."""

    normalized_sentence: str
    completed_sentence: str
    source_text: str
    offset: int


def normalize_text(text: str) -> str:
    """Normalize text exactly for matching, never for display.

    Case is ignored, punctuation and non-valid characters are ignored, and all
    whitespace runs become one ordinary space.  Punctuation is *removed*, not
    replaced by a space: this follows the wording "remove punctuation" in the
    supplied English appendix.
    """

    kept = []
    for character in text.casefold():
        if character.isalnum():
            kept.append(character)
        elif character.isspace():
            kept.append(" ")
        # Punctuation and other non-valid characters are deliberately ignored.
    return " ".join("".join(kept).split())


def remove_invalid_control_characters(text: str) -> str:
    """Drop control characters while retaining printable source punctuation.

    This keeps the required original punctuation in the displayed sentence but
    removes document-formatting controls such as form feed (``\\f``), which
    appeared in the supplied corpus and are not meaningful text input.
    """

    return "".join(character for character in text if character.isprintable() or character == "\t")


def _substitution_penalty(position: int) -> int:
    return (5, 4, 3, 2)[position - 1] if position <= 4 else 1


def _insertion_or_deletion_penalty(position: int) -> int:
    return (10, 8, 6, 4)[position - 1] if position <= 4 else 2


class OneEditMatcher:
    """Checks one normalized query against candidate normalized sentences.

    The regular expressions represent exactly the permitted edit operations.
    They are compiled only once per user query, rather than once for every
    candidate sentence.  The method returns the highest legal score or
    ``None`` when no legal substring match exists.
    """

    def __init__(self, normalized_query: str) -> None:
        self.query = normalized_query
        self.length = len(normalized_query)
        self.substitution_patterns: List[Tuple[int, re.Pattern[str]]] = []
        self.insertion_patterns: List[Tuple[int, re.Pattern[str]]] = []
        self.deleted_character_candidates: List[Tuple[int, str]] = []
        self._compile_one_edit_variants()

    def _compile_one_edit_variants(self) -> None:
        query = self.query
        length = self.length

        for index in range(length):
            position = index + 1
            substitution_score = 2 * (length - 1) - _substitution_penalty(position)
            substitution = re.escape(query[:index]) + "." + re.escape(query[index + 1 :])
            self.substitution_patterns.append((substitution_score, re.compile(substitution)))

            deletion_score = 2 * (length - 1) - _insertion_or_deletion_penalty(position)
            self.deleted_character_candidates.append(
                (deletion_score, query[:index] + query[index + 1 :])
            )

        # Inserting a missing character can occur before the first character,
        # between two characters, or after the final character.  Its position
        # is the insertion location in the one-based normalized query.
        for index in range(length + 1):
            position = index + 1
            insertion_score = 2 * length - _insertion_or_deletion_penalty(position)
            insertion = re.escape(query[:index]) + "." + re.escape(query[index:])
            self.insertion_patterns.append((insertion_score, re.compile(insertion)))

    def best_score(self, normalized_sentence: str) -> Optional[int]:
        if self.query in normalized_sentence:
            return 2 * self.length

        best: Optional[int] = None

        for score, candidate in self.deleted_character_candidates:
            if candidate in normalized_sentence and (best is None or score > best):
                best = score

        for score, pattern in self.substitution_patterns:
            if pattern.search(normalized_sentence) and (best is None or score > best):
                best = score

        for score, pattern in self.insertion_patterns:
            if pattern.search(normalized_sentence) and (best is None or score > best):
                best = score

        return best


class AutocompleteEngine:
    """Build once, then answer autocomplete requests with low latency."""

    def __init__(self, archive_path: Path) -> None:
        self.archive_path = Path(archive_path)
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="autocomplete-")
        self._database_path = Path(self._temporary_directory.name) / "index.sqlite3"
        self._connection: Optional[sqlite3.Connection] = None
        self.file_count = 0
        self.indexed_sentence_count = 0

    def __enter__(self) -> "AutocompleteEngine":
        self.build()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    @classmethod
    def from_archive(cls, archive_path: Path) -> "AutocompleteEngine":
        engine = cls(archive_path)
        engine.build()
        return engine

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._temporary_directory.cleanup()

    def build(self) -> None:
        """Run the offline phase over every .txt file in the ZIP archive."""

        if self._connection is not None:
            return
        if not self.archive_path.is_file():
            raise FileNotFoundError("Archive was not found: {0}".format(self.archive_path))

        connection = sqlite3.connect(str(self._database_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("PRAGMA cache_size = -131072")  # Up to 128 MiB cache.
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
        batch: List[Tuple[str, str, str, int]] = []
        connection.execute("BEGIN")
        with zipfile.ZipFile(self.archive_path) as archive:
            for entry in archive.infolist():
                if entry.is_dir() or not entry.filename.lower().endswith(".txt"):
                    continue
                self.file_count += 1
                with archive.open(entry, "r") as source:
                    # Invalid bytes are intentionally discarded, as requested.
                    for offset, raw_line in enumerate(source, start=1):
                        original = remove_invalid_control_characters(
                            raw_line.decode("utf-8", errors="ignore").rstrip("\r\n")
                        )
                        normalized = normalize_text(original)
                        if not normalized:
                            continue
                        batch.append((normalized, original, entry.filename, offset))
                        if len(batch) == 5000:
                            self._insert_batch(connection, batch)
                            batch.clear()
        if batch:
            self._insert_batch(connection, batch)

        connection.commit()

        # Creating this after bulk insertion is much faster than updating it
        # for every one of the millions of corpus lines.
        connection.execute(
            """
            CREATE INDEX sentence_alphabetical_order
            ON sentences(
                completed_sentence COLLATE NOCASE,
                completed_sentence,
                source_text COLLATE NOCASE,
                source_text,
                offset
            )
            """
        )

        self.indexed_sentence_count = connection.execute(
            "SELECT COUNT(*) FROM sentences"
        ).fetchone()[0]

        # The trigram index supports fast exact-substring and anchor queries.
        # It is an external-content FTS table, so source text is stored only once.
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE sentence_search USING fts5(
                    normalized,
                    content='sentences',
                    content_rowid='id',
                    tokenize='trigram'
                )
                """
            )
            connection.execute("INSERT INTO sentence_search(sentence_search) VALUES ('rebuild')")
        except sqlite3.OperationalError as error:
            connection.close()
            self._connection = None
            raise RuntimeError(
                "This Python SQLite build does not include the FTS5 trigram tokenizer. "
                "Use Python 3.11+ with a standard SQLite build."
            ) from error

        connection.commit()
        self._connection = connection

    @staticmethod
    def _insert_batch(
        connection: sqlite3.Connection, batch: Sequence[Tuple[str, str, str, int]]
    ) -> None:
        connection.executemany(
            """
            INSERT INTO sentences(normalized, completed_sentence, source_text, offset)
            VALUES (?, ?, ?, ?)
            """,
            batch,
        )

    @property
    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("The engine has not been initialized. Call build() first.")
        return self._connection

    @staticmethod
    def _fts_phrase(text: str) -> str:
        # normalize_text leaves only letters, digits and spaces, so the phrase
        # contains no FTS query syntax that needs special escaping.
        return '"{0}"'.format(text)

    @staticmethod
    def _result_from_row(row: sqlite3.Row, score: int) -> AutoCompleteData:
        return AutoCompleteData(
            completed_sentence=row["completed_sentence"],
            source_text=row["source_text"],
            offset=row["offset"],
            score=score,
        )

    def _exact_matches(self, normalized_query: str) -> List[AutoCompleteData]:
        """Return at most five exact matches in the required tie-break order."""

        if len(normalized_query) >= 3:
            rows = self._db.execute(
                """
                SELECT s.completed_sentence, s.source_text, s.offset
                FROM sentence_search AS f
                JOIN sentences AS s ON s.id = f.rowid
                WHERE f.normalized MATCH ?
                ORDER BY s.completed_sentence COLLATE NOCASE,
                         s.completed_sentence,
                         s.source_text COLLATE NOCASE,
                         s.source_text,
                         s.offset
                LIMIT ?
                """,
                (self._fts_phrase(normalized_query), K),
            ).fetchall()
        else:
            # The trigram tokenizer cannot index one- and two-character input.
            # This query is less selective, but preserves correctness for short
            # user input and still stops once five alphabetically first matches
            # are found.
            rows = self._db.execute(
                """
                SELECT completed_sentence, source_text, offset
                FROM sentences
                WHERE normalized LIKE ?
                ORDER BY completed_sentence COLLATE NOCASE,
                         completed_sentence,
                         source_text COLLATE NOCASE,
                         source_text,
                         offset
                LIMIT ?
                """,
                ("%{0}%".format(normalized_query), K),
            ).fetchall()

        score = 2 * len(normalized_query)
        return [self._result_from_row(row, score) for row in rows]

    def _variant_rows(self, normalized_variant: str) -> Iterator[sqlite3.Row]:
        """Return the first five alphabetic rows containing one exact variant."""

        if len(normalized_variant) >= 3:
            return self._db.execute(
                """
                SELECT s.id, s.completed_sentence, s.source_text, s.offset
                FROM sentence_search
                JOIN sentences AS s ON s.id = sentence_search.rowid
                WHERE sentence_search MATCH ?
                ORDER BY s.completed_sentence COLLATE NOCASE,
                         s.completed_sentence,
                         s.source_text COLLATE NOCASE,
                         s.source_text,
                         s.offset
                LIMIT ?
                """,
                (self._fts_phrase(normalized_variant), K),
            )
        # FTS trigram cannot index one- or two-character terms. This path is
        # used only for the low-information 2-3 character input cases; normal
        # query lengths use the indexed branch above.
        return self._db.execute(
            """
            SELECT id, completed_sentence, source_text, offset
            FROM sentences
            WHERE normalized LIKE ?
            ORDER BY completed_sentence COLLATE NOCASE,
                     completed_sentence,
                     source_text COLLATE NOCASE,
                     source_text,
                     offset
            LIMIT ?
            """,
            ("%{0}%".format(normalized_variant), K),
        )

    def _short_query_results(
        self, normalized_query: str, exact_results: List[AutoCompleteData]
    ) -> List[AutoCompleteData]:
        """Score 2-5 character queries by directly searching every one-edit form.

        The corpus is English, so every normalized sentence character belongs
        to ``NORMALIZED_CORPUS_ALPHABET``. For each allowed operation we create
        the corrected target substring and look it up directly. Variants of
        length 3+ use FTS, avoiding a broad scan for common anchors such as
        ``be``. Five alphabetically first rows per variant are sufficient: a
        sixth row with the same variant score cannot enter the global top five.
        """

        length = len(normalized_query)
        variants = {}

        def add_variant(variant: str, score: int) -> None:
            current_score = variants.get(variant)
            if current_score is None or score > current_score:
                variants[variant] = score

        for index in range(length):
            position = index + 1
            substitution_score = 2 * (length - 1) - _substitution_penalty(position)
            for character in NORMALIZED_CORPUS_ALPHABET:
                if character != normalized_query[index]:
                    add_variant(
                        normalized_query[:index] + character + normalized_query[index + 1 :],
                        substitution_score,
                    )

            deletion_score = 2 * (length - 1) - _insertion_or_deletion_penalty(position)
            add_variant(normalized_query[:index] + normalized_query[index + 1 :], deletion_score)

        for index in range(length + 1):
            position = index + 1
            insertion_score = 2 * length - _insertion_or_deletion_penalty(position)
            for character in NORMALIZED_CORPUS_ALPHABET:
                add_variant(
                    normalized_query[:index] + character + normalized_query[index:],
                    insertion_score,
                )

        best_by_id = {}
        # Exact results are already known to be the only possible highest-score
        # matches. They may be fewer than five, otherwise the caller exits early.
        for result in exact_results:
            best_by_id[(result.completed_sentence, result.source_text, result.offset)] = result

        for score in sorted(set(variants.values()), reverse=True):
            for variant, variant_score in variants.items():
                if variant_score != score:
                    continue
                for row in self._variant_rows(variant):
                    result = self._result_from_row(row, score)
                    key = (result.completed_sentence, result.source_text, result.offset)
                    current = best_by_id.get(key)
                    if current is None or self._result_sort_key(result) < self._result_sort_key(current):
                        best_by_id[key] = result

            ranked = sorted(best_by_id.values(), key=self._result_sort_key)
            # All remaining variants have lower scores, so they cannot improve
            # the top five once this score group already filled it.
            if len(ranked) >= K:
                return ranked[:K]

        return sorted(best_by_id.values(), key=self._result_sort_key)[:K]

    def _trigram_rows(self, expression: str) -> Iterator[sqlite3.Row]:
        """Return rows selected by the long-substring FTS index."""

        return self._db.execute(
            """
            SELECT s.id, s.normalized, s.completed_sentence, s.source_text, s.offset
            FROM sentence_search
            JOIN sentences AS s ON s.id = sentence_search.rowid
            WHERE sentence_search MATCH ?
            """,
            (expression,),
        )

    def _candidate_rows(self, normalized_query: str) -> Iterator[sqlite3.Row]:
        """Yield all candidates for a query of at least six characters.

        Two non-overlapping anchors are searched. With at most one edit, at
        least one anchor cannot contain that edit and must occur exactly in a
        legal matching sentence.
        """

        anchor_size = min(8, len(normalized_query) // 2)
        anchors = (normalized_query[:anchor_size], normalized_query[-anchor_size:])
        expression = " OR ".join(sorted({self._fts_phrase(anchor) for anchor in anchors}))
        return self._trigram_rows(expression)

    def _all_rows(self) -> Iterator[sqlite3.Row]:
        """Correct fallback for a one-character fuzzy query."""

        return self._db.execute(
            """
            SELECT id, normalized, completed_sentence, source_text, offset
            FROM sentences
            """
        )

    @staticmethod
    def _result_sort_key(result: AutoCompleteData) -> Tuple[int, str, str, str, str, int]:
        return (
            -result.score,
            result.completed_sentence.casefold(),
            result.completed_sentence,
            result.source_text.casefold(),
            result.source_text,
            result.offset,
        )

    def _top_matches_from_rows(
        self, rows: Iterator[sqlite3.Row], matcher: OneEditMatcher
    ) -> List[AutoCompleteData]:
        """Keep only five best results while consuming an SQLite cursor."""

        best: List[AutoCompleteData] = []
        seen_ids = set()
        for row in rows:
            if row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            score = matcher.best_score(row["normalized"])
            if score is None:
                continue
            result = self._result_from_row(row, score)
            if len(best) < K:
                best.append(result)
                best.sort(key=self._result_sort_key)
            elif self._result_sort_key(result) < self._result_sort_key(best[-1]):
                best[-1] = result
                best.sort(key=self._result_sort_key)
        return best

    def get_best_k_completions(self, prefix: str) -> List[AutoCompleteData]:
        """Return the five highest-scoring legal autocomplete results.

        An empty normalized string deliberately returns no results.  A blank
        query would otherwise match every line as an empty substring and is not
        useful in the interactive program.
        """

        normalized_query = normalize_text(prefix)
        if not normalized_query:
            return []

        exact = self._exact_matches(normalized_query)
        # Exact matches have score 2 * len(query), strictly higher than every
        # one-edit match.  Five of them settle the result without further work.
        if len(exact) == K:
            return exact

        if 2 <= len(normalized_query) <= 5:
            return self._short_query_results(normalized_query, exact)

        matcher = OneEditMatcher(normalized_query)
        rows = (
            self._candidate_rows(normalized_query)
            if len(normalized_query) >= 6
            else self._all_rows()
        )
        return self._top_matches_from_rows(rows, matcher)


_default_engine: Optional[AutocompleteEngine] = None


def initialize(archive_path: Path) -> AutocompleteEngine:
    """Initialize the default engine used by the required module-level API."""

    global _default_engine
    if _default_engine is not None:
        _default_engine.close()
    _default_engine = AutocompleteEngine.from_archive(archive_path)
    return _default_engine


def get_best_k_completions(prefix: str) -> List[AutoCompleteData]:
    """The assignment-required completion function."""

    if _default_engine is None:
        raise RuntimeError("Call initialize(archive_path) before requesting completions.")
    return _default_engine.get_best_k_completions(prefix)


def run_cli(engine: AutocompleteEngine) -> None:
    """Run the required online phase, appending input until '#' resets it."""

    query = ""
    print("The system is ready. Type additional text and press Enter.")
    print("Type # by itself to clear the current query. Press Ctrl+C to exit.")
    try:
        while True:
            addition = input("Current text [{0}]: ".format(query))
            if addition == "#":
                query = ""
                print("The current query was reset.")
                continue

            query += addition
            suggestions = engine.get_best_k_completions(query)
            print("Here are {0} suggestions:".format(len(suggestions)))
            for number, suggestion in enumerate(suggestions, start=1):
                print(suggestion.format_for_cli(number))
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-edit text autocomplete over a ZIP archive.")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("Archive (2).zip"),
        help="Path to the ZIP archive containing .txt files.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build and validate the offline index, then exit.",
    )
    arguments = parser.parse_args()

    print("Building the offline search index. This is done once per run...", flush=True)
    started_at = time.perf_counter()
    engine = AutocompleteEngine.from_archive(arguments.archive)
    try:
        print(
            "Indexed {0:,} non-empty lines from {1:,} text files in {2:.1f} seconds.".format(
                engine.indexed_sentence_count,
                engine.file_count,
                time.perf_counter() - started_at,
            )
        )
        if not arguments.build_only:
            run_cli(engine)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
