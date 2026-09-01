"""Export the Part A sentence index as a semantic-embedding JSONL dataset."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Iterator, Optional
import argparse
import json
import os
import sqlite3
import sys
import tempfile


DEFAULT_INDEX_PATH = Path("index.sqlite3")
DEFAULT_OUTPUT_PATH = Path("data/semantic_dataset.jsonl")
REQUIRED_COLUMNS = {"id", "completed_sentence", "source_text", "offset"}


class DatasetBuildError(RuntimeError):
    """A user-facing failure while building the semantic dataset."""


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _open_index(index_path: Path) -> sqlite3.Connection:
    if not index_path.is_file():
        raise DatasetBuildError("Index was not found: {0}".format(index_path))

    try:
        connection = sqlite3.connect(
            "{0}?mode=ro".format(index_path.resolve().as_uri()),
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        return connection
    except (OSError, sqlite3.Error) as error:
        raise DatasetBuildError(
            "Could not open SQLite index '{0}': {1}".format(index_path, error)
        ) from error


def _validate_schema(connection: sqlite3.Connection) -> None:
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(sentences)")
    }
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise DatasetBuildError(
            "Index is missing required sentences columns: {0}".format(
                ", ".join(missing)
            )
        )


def _sentence_rows(
    connection: sqlite3.Connection, limit: Optional[int]
) -> Iterator[sqlite3.Row]:
    query = """
        SELECT id, completed_sentence, source_text, offset
        FROM sentences
        ORDER BY id
    """
    if limit is None:
        return iter(connection.execute(query))
    return iter(connection.execute(query + " LIMIT ?", (limit,)))


def build_dataset(
    index_path: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    limit: Optional[int] = None,
) -> int:
    """Stream indexed sentences to JSONL and return the number written."""

    index_path = Path(index_path)
    output_path = Path(output_path)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if index_path.resolve() == output_path.resolve():
        raise DatasetBuildError("Output path must not be the SQLite index path.")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DatasetBuildError(
            "Could not create output directory '{0}': {1}".format(
                output_path.parent, error
            )
        ) from error

    temporary_path: Optional[Path] = None
    try:
        with closing(_open_index(index_path)) as connection:
            _validate_schema(connection)
            rows = _sentence_rows(connection, limit)
            count = 0
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=output_path.parent,
                prefix=".{0}.".format(output_path.name),
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                for row in rows:
                    record = {
                        "id": row["id"],
                        "sentence": row["completed_sentence"],
                        "source_text": row["source_text"],
                        "offset": row["offset"],
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

        os.replace(temporary_path, output_path)
        temporary_path = None
        return count
    except DatasetBuildError:
        raise
    except sqlite3.Error as error:
        raise DatasetBuildError(
            "Could not read SQLite index '{0}': {1}".format(index_path, error)
        ) from error
    except OSError as error:
        raise DatasetBuildError(
            "Could not write semantic dataset '{0}': {1}".format(
                output_path, error
            )
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a JSONL dataset from the Part A SQLite sentence index."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to the existing Part A SQLite index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_integer,
        help="Export only the first N sentences in ID order.",
    )
    return parser


def main() -> int:
    arguments = create_parser().parse_args()
    try:
        count = build_dataset(arguments.index, arguments.output, arguments.limit)
    except (DatasetBuildError, ValueError) as error:
        print("error: {0}".format(error), file=sys.stderr)
        return 1

    print("Created semantic dataset: {0:,} sentences".format(count))
    print("Output: {0}".format(arguments.output.as_posix()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
