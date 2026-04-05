from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config import settings
from src.embedding_client import _EmbeddingClient


def test_openai_compatible_embeddings_use_dedicated_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.LLM, "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.setattr(
        settings.LLM, "EMBEDDING_BASE_URL", "http://embeddings:7997/v1"
    )
    monkeypatch.setattr(settings.LLM, "EMBEDDING_API_KEY", "local-embed-key")

    with patch("src.embedding_client.AsyncOpenAI") as mock_openai:
        client = _EmbeddingClient(provider="openai_compatible")

    mock_openai.assert_called_once_with(
        api_key="local-embed-key",
        base_url="http://embeddings:7997/v1",
    )
    assert client.provider == "openai_compatible"
    assert client.model == "Qwen/Qwen3-Embedding-0.6B"


def test_openrouter_alias_preserves_legacy_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.LLM, "EMBEDDING_MODEL", None)
    monkeypatch.setattr(settings.LLM, "EMBEDDING_BASE_URL", None)
    monkeypatch.setattr(settings.LLM, "OPENAI_COMPATIBLE_BASE_URL", None)

    with patch("src.embedding_client.AsyncOpenAI") as mock_openai:
        client = _EmbeddingClient(api_key="legacy-key", provider="openrouter")

    mock_openai.assert_called_once_with(
        api_key="legacy-key",
        base_url="https://openrouter.ai/api/v1",
    )
    assert client.provider == "openai_compatible"
    assert client.model == "openai/text-embedding-3-small"


@pytest.mark.asyncio
async def test_gemini_embeddings_use_configured_vector_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.LLM, "EMBEDDING_MODEL", None)
    monkeypatch.setattr(settings.VECTOR_STORE, "DIMENSIONS", 1024)

    fake_response = Mock()
    fake_response.embeddings = [Mock(values=[0.1, 0.2, 0.3])]

    class FakeGeminiClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.aio = Mock()
            self.aio.models = Mock()
            self.aio.models.embed_content = AsyncMock(return_value=fake_response)

    with patch("src.embedding_client.genai.Client", FakeGeminiClient):
        client = _EmbeddingClient(api_key="gemini-key", provider="gemini")
        result = await client.embed("hello world")

    client.client.aio.models.embed_content.assert_awaited_once()
    assert (
        client.client.aio.models.embed_content.call_args.kwargs["config"][
            "output_dimensionality"
        ]
        == 1024
    )
    assert result == [0.1, 0.2, 0.3]
