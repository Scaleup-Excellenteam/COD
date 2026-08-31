"""Corpus-backed performance and behavior check for the autocomplete engine.

Run with:
    py benchmark.py
"""

from pathlib import Path
from time import perf_counter
import argparse

from autocomplete import AutocompleteEngine


DEFAULT_QUERIES = [
    "python",
    "PYTHON,,,    ",
    "pythom",
    "data analysiz",
    "to",
    "to pe",
    "2o be",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark autocomplete on a ZIP corpus.")
    parser.add_argument("--archive", type=Path, default=Path("Archive (2).zip"))
    parser.add_argument("queries", nargs="*", help="Optional queries replacing the default set.")
    arguments = parser.parse_args()
    queries = arguments.queries or DEFAULT_QUERIES

    started_at = perf_counter()
    engine = AutocompleteEngine.from_archive(arguments.archive)
    build_seconds = perf_counter() - started_at
    try:
        print(
            "BUILD: {0:,} lines from {1:,} files in {2:.3f}s".format(
                engine.indexed_sentence_count,
                engine.file_count,
                build_seconds,
            )
        )
        for query in queries:
            started_at = perf_counter()
            results = engine.get_best_k_completions(query)
            query_seconds = perf_counter() - started_at
            first = results[0] if results else None
            print(
                "QUERY: {0!r} | {1:.6f}s | {2} result(s) | first={3!r} | score={4}".format(
                    query,
                    query_seconds,
                    len(results),
                    first.completed_sentence if first else None,
                    first.score if first else None,
                )
            )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
