"""Focused Phase 1 tests for Earth-side Gemini document embeddings."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from google.genai import types

from semantic.gemini_embeddings import (
    EmbeddingResponseError,
    create_client,
    embed_query,
    embed_sentence,
    prepare_document_text,
    prepare_query_text,
)


class FakeModels:
    def __init__(self, response: types.EmbedContentResponse) -> None:
        self.response = response
        self.calls = []

    def embed_content(self, **arguments):
        self.calls.append(arguments)
        return self.response


class FakeClient:
    def __init__(self, response: types.EmbedContentResponse) -> None:
        self.models = FakeModels(response)


class DocumentFormattingTests(unittest.TestCase):
    def test_formats_document_input_without_modifying_original_sentence(
        self,
    ) -> None:
        sentence = "Python is a programming language."

        prepared = prepare_document_text(sentence)

        self.assertEqual(
            prepared,
            "title: none | text: Python is a programming language.",
        )
        self.assertEqual(sentence, "Python is a programming language.")


class ClientCreationTests(unittest.TestCase):
    def test_creates_official_client_with_server_side_configuration(self) -> None:
        sentinel_client = object()

        with patch(
            "semantic.gemini_embeddings.config.get_gemini_api_key",
            return_value="configured-key",
        ), patch(
            "semantic.gemini_embeddings.genai.Client",
            return_value=sentinel_client,
        ) as client_constructor:
            client = create_client()

        self.assertIs(client, sentinel_client)
        client_constructor.assert_called_once_with(
            api_key="configured-key",
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )


class EmbedSentenceTests(unittest.TestCase):
    def test_one_sentence_makes_one_768_dimension_document_request(self) -> None:
        values = [index / 1000 for index in range(768)]
        response = types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=values)]
        )
        client = FakeClient(response)

        embedding = embed_sentence(client, "A harmless corpus sentence.")

        self.assertEqual(embedding, values)
        self.assertEqual(len(client.models.calls), 1)
        call = client.models.calls[0]
        self.assertEqual(call["model"], "gemini-embedding-2")
        self.assertEqual(
            call["contents"],
            "title: none | text: A harmless corpus sentence.",
        )
        self.assertEqual(call["config"].output_dimensionality, 768)

    def test_rejects_an_embedding_with_the_wrong_dimensions(self) -> None:
        response = types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.0] * 767)]
        )
        client = FakeClient(response)

        with self.assertRaisesRegex(EmbeddingResponseError, "768 dimensions"):
            embed_sentence(client, "Sentence")

    def test_rejects_missing_or_multiple_embeddings(self) -> None:
        responses = [
            types.EmbedContentResponse(embeddings=None),
            types.EmbedContentResponse(embeddings=[]),
            types.EmbedContentResponse(
                embeddings=[
                    types.ContentEmbedding(values=[0.0] * 768),
                    types.ContentEmbedding(values=[1.0] * 768),
                ]
            ),
        ]

        for response in responses:
            with self.subTest(response=response), self.assertRaisesRegex(
                EmbeddingResponseError, "exactly one embedding"
            ):
                embed_sentence(FakeClient(response), "Sentence")


class EmbedQueryTests(unittest.TestCase):
    def test_formats_query_input_without_modifying_original_query(self) -> None:
        query = "How do computers learn?"

        prepared = prepare_query_text(query)

        self.assertEqual(
            prepared,
            "task: search result | query: How do computers learn?",
        )
        self.assertEqual(query, "How do computers learn?")

    def test_one_query_makes_one_768_dimension_query_request(self) -> None:
        values = [index / 1000 for index in range(768)]
        client = FakeClient(
            types.EmbedContentResponse(
                embeddings=[types.ContentEmbedding(values=values)]
            )
        )

        embedding = embed_query(client, "How do computers learn?")

        self.assertEqual(embedding, values)
        self.assertEqual(len(client.models.calls), 1)
        call = client.models.calls[0]
        self.assertEqual(call["model"], "gemini-embedding-2")
        self.assertEqual(
            call["contents"],
            "task: search result | query: How do computers learn?",
        )
        self.assertEqual(call["config"].output_dimensionality, 768)

    def test_rejects_blank_query_before_making_a_request(self) -> None:
        client = FakeClient(types.EmbedContentResponse(embeddings=[]))

        for query in ("", "   ", None):
            with self.subTest(query=query), self.assertRaisesRegex(
                ValueError, "nonblank"
            ):
                embed_query(client, query)
        self.assertEqual(client.models.calls, [])

    def test_rejects_invalid_query_responses(self) -> None:
        invalid_responses = [
            types.EmbedContentResponse(embeddings=[]),
            types.EmbedContentResponse(
                embeddings=[types.ContentEmbedding(values=[0.0] * 767)]
            ),
            SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.0] * 767 + [True])]
            ),
        ]

        for response in invalid_responses:
            with self.subTest(response=response), self.assertRaises(
                EmbeddingResponseError
            ):
                embed_query(FakeClient(response), "Useful query")

    def test_rejects_missing_embedding_values(self) -> None:
        response = SimpleNamespace(
            embeddings=[SimpleNamespace(values=None)]
        )

        with self.assertRaisesRegex(EmbeddingResponseError, "values are missing"):
            embed_sentence(FakeClient(response), "Sentence")

    def test_rejects_non_numeric_embedding_values(self) -> None:
        for invalid_value in ["not numeric", None, True]:
            values = [0.0] * 767 + [invalid_value]
            response = SimpleNamespace(
                embeddings=[SimpleNamespace(values=values)]
            )

            with self.subTest(
                invalid_value=invalid_value
            ), self.assertRaisesRegex(EmbeddingResponseError, "numeric values"):
                embed_sentence(FakeClient(response), "Sentence")


if __name__ == "__main__":
    unittest.main()
