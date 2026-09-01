"""Build Earth-side corpus embeddings as a satellite deployment artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TextIO
import argparse
import json
import os
import sys
import tempfile

from semantic.gemini_embeddings import (
    EMBEDDING_DIMENSIONS,
    MODEL,
    create_client,
    embed_sentence,
)


DEFAULT_INPUT_PATH = Path("data/semantic_dataset.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/semantic_embeddings.jsonl")
REQUIRED_FIELDS = ("id", "sentence", "source_text", "offset")


class EmbeddingBuildError(RuntimeError):
    """A user-facing failure while preparing corpus embeddings."""


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _validate_record(raw_line: str, line_number: int) -> dict:
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as error:
        raise EmbeddingBuildError(
            "Input line {0} is not valid JSON.".format(line_number)
        ) from error

    if not isinstance(record, dict):
        raise EmbeddingBuildError(
            "Input line {0} must be a JSON object.".format(line_number)
        )

    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise EmbeddingBuildError(
            "Input line {0} is missing required fields: {1}.".format(
                line_number, ", ".join(missing)
            )
        )

    sentence = record["sentence"]
    if not isinstance(sentence, str) or not sentence.strip():
        raise EmbeddingBuildError(
            "Input line {0} must contain a nonblank string sentence.".format(
                line_number
            )
        )
    return record


def _selected_records(
    input_path: Path, limit: Optional[int]
) -> Iterator[dict]:
    selected = 0
    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            for line_number, raw_line in enumerate(input_file, start=1):
                if limit is not None and selected >= limit:
                    break
                yield _validate_record(raw_line, line_number)
                selected += 1
    except FileNotFoundError as error:
        raise EmbeddingBuildError(
            "Input dataset was not found: {0}".format(input_path)
        ) from error
    except (OSError, UnicodeError) as error:
        raise EmbeddingBuildError(
            "Could not read input dataset '{0}'.".format(input_path)
        ) from error


def validate_selected_records(
    input_path: Path, limit: Optional[int]
) -> int:
    """Stream-validate selected records and return their count."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    return sum(1 for _record in _selected_records(Path(input_path), limit))


def build_embeddings(
    input_path: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    limit: Optional[int] = None,
    client_factory: Callable[[], Any] = create_client,
    progress_stream: TextIO = sys.stdout,
) -> int:
    """Prepare selected corpus embeddings atomically and return their count."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise EmbeddingBuildError("Input and output paths must be different.")

    selected_count = validate_selected_records(input_path, limit)
    print(
        "Selected records: {0:,}".format(selected_count),
        file=progress_stream,
        flush=True,
    )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise EmbeddingBuildError(
            "Could not create output directory '{0}'.".format(
                output_path.parent
            )
        ) from error

    try:
        client = client_factory() if selected_count else None
    except Exception as error:
        raise EmbeddingBuildError("Could not create the Gemini client.") from error
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=".{0}.".format(output_path.name),
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            for position, record in enumerate(
                _selected_records(input_path, limit), start=1
            ):
                try:
                    embedding = embed_sentence(client, record["sentence"])
                except Exception as error:
                    raise EmbeddingBuildError(
                        "Gemini embedding failed for selected record {0}.".format(
                            position
                        )
                    ) from error

                output_record = {
                    "id": record["id"],
                    "sentence": record["sentence"],
                    "source_text": record["source_text"],
                    "offset": record["offset"],
                    "embedding": embedding,
                }
                output_file.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                print(
                    "\rEmbedding: {0:,} / {1:,}".format(
                        position, selected_count
                    ),
                    end="",
                    file=progress_stream,
                    flush=True,
                )

        if selected_count:
            print(file=progress_stream, flush=True)
        os.replace(temporary_path, output_path)
        temporary_path = None
        return selected_count
    except EmbeddingBuildError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise EmbeddingBuildError(
            "Could not write semantic embeddings '{0}'.".format(output_path)
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Gemini corpus embeddings offline on Earth."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Source semantic dataset JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination semantic embeddings JSONL path.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--limit",
        type=_positive_integer,
        help="Embed only the first N dataset records.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Explicitly embed the entire dataset.",
    )
    return parser


def main() -> int:
    arguments = create_parser().parse_args()
    limit = arguments.limit if not arguments.all else None
    try:
        count = build_embeddings(
            arguments.input,
            arguments.output,
            limit=limit,
        )
    except (EmbeddingBuildError, ValueError) as error:
        print("error: {0}".format(error), file=sys.stderr)
        return 1

    print("Created semantic embeddings: {0:,} sentences".format(count))
    print("Model: {0}".format(MODEL))
    print("Dimensions: {0}".format(EMBEDDING_DIMENSIONS))
    print("Output: {0}".format(arguments.output.as_posix()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
