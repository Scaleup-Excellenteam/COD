"""Satellite-local semantic retrieval over prebuilt FAISS artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Iterable
import json
import math

import faiss
import numpy as np


EMBEDDING_DIMENSIONS = 768
TOP_K = 5
METADATA_FIELDS = ("id", "sentence", "source_text", "offset")


class SemanticSearchError(RuntimeError):
    """The local semantic deployment artifacts cannot be used safely."""


class SemanticQueryError(ValueError):
    """A query embedding is not valid for the deployed semantic index."""


@dataclass(frozen=True)
class SemanticSearchResult:
    """One satellite-local semantic match, independent of Part A scoring."""

    sentence: str
    source_text: str
    offset: int
    semantic_score: float


def _metadata_error(line_number: int, message: str) -> SemanticSearchError:
    return SemanticSearchError(
        "Metadata line {0} {1}".format(line_number, message)
    )


def _validate_metadata_record(record: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise _metadata_error(line_number, "must be a JSON object.")

    missing = [field for field in METADATA_FIELDS if field not in record]
    if missing:
        raise _metadata_error(
            line_number,
            "is missing required fields: {0}.".format(", ".join(missing)),
        )

    identifier = record["id"]
    if isinstance(identifier, bool) or not isinstance(identifier, int):
        raise _metadata_error(line_number, "must contain an integer id.")
    for field in ("sentence", "source_text"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise _metadata_error(
                line_number,
                "must contain a nonblank string {0}.".format(field),
            )
    offset = record["offset"]
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise _metadata_error(
            line_number, "must contain a nonnegative integer offset."
        )
    return {field: record[field] for field in METADATA_FIELDS}


def _load_metadata(metadata_path: Path) -> tuple[dict[str, Any], ...]:
    records = []
    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            for line_number, raw_line in enumerate(metadata_file, start=1):
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise _metadata_error(
                        line_number, "is not valid JSON."
                    ) from error
                records.append(_validate_metadata_record(record, line_number))
    except FileNotFoundError as error:
        raise SemanticSearchError(
            "Semantic metadata was not found: {0}".format(metadata_path)
        ) from error
    except (OSError, UnicodeError) as error:
        raise SemanticSearchError(
            "Could not read semantic metadata '{0}'.".format(metadata_path)
        ) from error
    return tuple(records)


def _prepare_query(query_embedding: Iterable[Real]) -> np.ndarray:
    if isinstance(query_embedding, (str, bytes)):
        raise SemanticQueryError("Query embedding must be a numeric vector.")
    try:
        values = list(query_embedding)
    except TypeError as error:
        raise SemanticQueryError("Query embedding must be an iterable vector.") from error

    if len(values) != EMBEDDING_DIMENSIONS:
        raise SemanticQueryError(
            "Query embedding must contain exactly {0} dimensions.".format(
                EMBEDDING_DIMENSIONS
            )
        )
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise SemanticQueryError(
            "Query embedding must contain only numeric non-boolean values."
        )
    if any(not math.isfinite(float(value)) for value in values):
        raise SemanticQueryError("Query embedding must contain only finite values.")

    with np.errstate(over="ignore", invalid="ignore"):
        vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise SemanticQueryError(
            "Query embedding values must be representable as float32."
        )

    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if norm == 0.0:
        raise SemanticQueryError("Query embedding must not be a zero vector.")
    return (vector / norm).astype(np.float32, copy=False).reshape(1, -1)


class SemanticSearchEngine:
    """An in-memory satellite engine backed by one exact FAISS index load."""

    def __init__(self, index: faiss.Index, metadata: tuple[dict[str, Any], ...]):
        if index.d != EMBEDDING_DIMENSIONS:
            raise SemanticSearchError(
                "FAISS index dimension is {0}; expected {1}.".format(
                    index.d, EMBEDDING_DIMENSIONS
                )
            )
        if not isinstance(index, faiss.IndexFlatIP):
            raise SemanticSearchError(
                "FAISS index must be an exact IndexFlatIP index."
            )
        if index.ntotal != len(metadata):
            raise SemanticSearchError(
                "FAISS index contains {0} vectors but metadata contains {1} records."
                .format(index.ntotal, len(metadata))
            )
        self._index = index
        self._metadata = metadata

    @classmethod
    def from_files(
        cls, index_path: Path, metadata_path: Path
    ) -> "SemanticSearchEngine":
        """Load and validate both satellite deployment artifacts once."""

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        try:
            index = faiss.read_index(str(index_path))
        except RuntimeError as error:
            raise SemanticSearchError(
                "Could not load FAISS index '{0}'.".format(index_path)
            ) from error
        metadata = _load_metadata(metadata_path)
        return cls(index, metadata)

    def search(self, query_embedding: Iterable[Real]) -> list[SemanticSearchResult]:
        """Return up to five cosine-ranked semantic matches."""

        query = _prepare_query(query_embedding)
        scores, positions = self._index.search(query, TOP_K)
        if scores.shape != (1, TOP_K) or positions.shape != (1, TOP_K):
            raise SemanticSearchError("FAISS returned an unexpected result shape.")

        results = []
        for score, position_value in zip(scores[0], positions[0]):
            position = int(position_value)
            if position == -1:
                continue
            if position < 0 or position >= len(self._metadata):
                raise SemanticSearchError(
                    "FAISS returned invalid metadata position {0}.".format(position)
                )
            if not math.isfinite(float(score)):
                raise SemanticSearchError("FAISS returned a non-finite score.")

            metadata = self._metadata[position]
            results.append(
                SemanticSearchResult(
                    sentence=metadata["sentence"],
                    source_text=metadata["source_text"],
                    offset=metadata["offset"],
                    semantic_score=float(score),
                )
            )
        return results
