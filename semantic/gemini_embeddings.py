"""Earth-side Gemini support for offline corpus preparation.

This module prepares document vectors before deployment. It is not part of the
satellite's per-query runtime.
"""

from __future__ import annotations

from numbers import Real
from typing import Any

from google import genai
from google.genai import types

from semantic import config


MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768


class EmbeddingResponseError(RuntimeError):
    """Gemini returned an unusable embedding response."""


def create_client() -> genai.Client:
    """Create the server-side Gemini client for offline preparation."""

    return genai.Client(
        api_key=config.get_gemini_api_key(),
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )


def prepare_document_text(sentence: str) -> str:
    """Format one corpus sentence for document-side retrieval embedding."""

    return "title: none | text: {0}".format(sentence)


def embed_sentence(client: Any, sentence: str) -> list[float]:
    """Request one document embedding for one corpus sentence."""

    response = client.models.embed_content(
        model=MODEL,
        contents=prepare_document_text(sentence),
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS
        ),
    )
    embeddings = getattr(response, "embeddings", None)
    if not isinstance(embeddings, list) or len(embeddings) != 1:
        raise EmbeddingResponseError(
            "Gemini must return exactly one embedding."
        )

    values = getattr(embeddings[0], "values", None)
    if values is None:
        raise EmbeddingResponseError("Gemini embedding values are missing.")
    if len(values) != EMBEDDING_DIMENSIONS:
        raise EmbeddingResponseError(
            "Gemini embedding must contain exactly {0} dimensions.".format(
                EMBEDDING_DIMENSIONS
            )
        )
    has_non_numeric_value = any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in values
    )
    if has_non_numeric_value:
        raise EmbeddingResponseError(
            "Gemini embedding must contain only numeric values."
        )
    return [float(value) for value in values]
