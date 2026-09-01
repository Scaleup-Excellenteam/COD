"""Build an exact cosine-similarity FAISS deployment artifact on Earth."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Optional
import argparse
import json
import math
import os
import sys
import tempfile

import faiss
import numpy as np


EMBEDDING_DIMENSIONS = 768
DEFAULT_INPUT_PATH = Path("data/semantic_embeddings.jsonl")
DEFAULT_INDEX_OUTPUT_PATH = Path("data/semantic.faiss")
DEFAULT_METADATA_OUTPUT_PATH = Path("data/semantic_metadata.jsonl")
METADATA_FIELDS = ("id", "sentence", "source_text", "offset")
REQUIRED_FIELDS = METADATA_FIELDS + ("embedding",)


class FaissIndexBuildError(RuntimeError):
    """A user-facing failure while building FAISS deployment artifacts."""


@dataclass(frozen=True)
class BuildSummary:
    """Facts about a completed FAISS index build."""

    vectors_indexed: int
    dimensions: int
    index_type: str
    metadata_records: int


def _record_error(line_number: int, message: str) -> FaissIndexBuildError:
    return FaissIndexBuildError("Input line {0} {1}".format(line_number, message))


def _validate_metadata(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise _record_error(
            line_number,
            "is missing required fields: {0}.".format(", ".join(missing)),
        )

    identifier = record["id"]
    if isinstance(identifier, bool) or not isinstance(identifier, int):
        raise _record_error(line_number, "must contain an integer id.")

    for field in ("sentence", "source_text"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise _record_error(
                line_number,
                "must contain a nonblank string {0}.".format(field),
            )

    offset = record["offset"]
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise _record_error(
            line_number, "must contain a nonnegative integer offset."
        )

    return {field: record[field] for field in METADATA_FIELDS}


def _validate_embedding(record: dict[str, Any], line_number: int) -> np.ndarray:
    embedding = record["embedding"]
    if not isinstance(embedding, list):
        raise _record_error(line_number, "must contain an embedding array.")
    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise _record_error(
            line_number,
            "embedding must contain exactly {0} dimensions.".format(
                EMBEDDING_DIMENSIONS
            ),
        )

    if any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in embedding
    ):
        raise _record_error(
            line_number, "embedding must contain only numeric values."
        )

    if any(not math.isfinite(float(value)) for value in embedding):
        raise _record_error(
            line_number, "embedding must contain only finite values."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        vector = np.asarray(embedding, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise _record_error(
            line_number, "embedding values must be representable as float32."
        )
    if float(np.linalg.norm(vector.astype(np.float64))) == 0.0:
        raise _record_error(line_number, "embedding must not be a zero vector.")
    return vector


def _load_input(input_path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            for line_number, raw_line in enumerate(input_file, start=1):
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise _record_error(line_number, "is not valid JSON.") from error
                if not isinstance(record, dict):
                    raise _record_error(line_number, "must be a JSON object.")

                metadata.append(_validate_metadata(record, line_number))
                vectors.append(_validate_embedding(record, line_number))
    except FileNotFoundError as error:
        raise FaissIndexBuildError(
            "Input embeddings were not found: {0}".format(input_path)
        ) from error
    except (OSError, UnicodeError) as error:
        raise FaissIndexBuildError(
            "Could not read input embeddings '{0}'.".format(input_path)
        ) from error

    if vectors:
        matrix = np.stack(vectors).astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
        matrix = (matrix / norms).astype(np.float32, copy=False)
    else:
        matrix = np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
    return matrix, metadata


def _temporary_path(output_path: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=".{0}.".format(output_path.name),
        suffix=".tmp",
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    return path


def build_faiss_index(
    input_path: Path = DEFAULT_INPUT_PATH,
    index_output_path: Path = DEFAULT_INDEX_OUTPUT_PATH,
    metadata_output_path: Path = DEFAULT_METADATA_OUTPUT_PATH,
) -> BuildSummary:
    """Validate embeddings and atomically publish aligned FAISS/metadata files."""

    input_path = Path(input_path)
    index_output_path = Path(index_output_path)
    metadata_output_path = Path(metadata_output_path)
    resolved_paths = {
        input_path.resolve(),
        index_output_path.resolve(),
        metadata_output_path.resolve(),
    }
    if len(resolved_paths) != 3:
        raise FaissIndexBuildError("Input and output paths must all be different.")

    matrix, metadata = _load_input(input_path)
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(matrix)

    index_temporary_path: Optional[Path] = None
    metadata_temporary_path: Optional[Path] = None
    try:
        for output_path in (index_output_path, metadata_output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)

        index_temporary_path = _temporary_path(index_output_path)
        faiss.write_index(index, str(index_temporary_path))

        metadata_temporary_path = _temporary_path(metadata_output_path)
        with metadata_temporary_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as metadata_file:
            for record in metadata:
                metadata_file.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )

        os.replace(index_temporary_path, index_output_path)
        index_temporary_path = None
        os.replace(metadata_temporary_path, metadata_output_path)
        metadata_temporary_path = None
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise FaissIndexBuildError(
            "Could not write FAISS deployment artifacts."
        ) from error
    finally:
        for temporary_path in (index_temporary_path, metadata_temporary_path):
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    return BuildSummary(
        vectors_indexed=index.ntotal,
        dimensions=index.d,
        index_type=type(index).__name__,
        metadata_records=len(metadata),
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an exact normalized FAISS corpus index offline on Earth."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--index-output", type=Path, default=DEFAULT_INDEX_OUTPUT_PATH
    )
    parser.add_argument(
        "--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT_PATH
    )
    return parser


def main() -> int:
    arguments = create_parser().parse_args()
    try:
        summary = build_faiss_index(
            arguments.input,
            arguments.index_output,
            arguments.metadata_output,
        )
    except FaissIndexBuildError as error:
        print("error: {0}".format(error), file=sys.stderr)
        return 1

    print("Vectors indexed: {0}".format(summary.vectors_indexed))
    print("Dimensions: {0}".format(summary.dimensions))
    print("Index type: {0}".format(summary.index_type))
    print("Metadata records: {0}".format(summary.metadata_records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
