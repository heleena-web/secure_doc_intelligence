import pytest
import httpx
from unittest.mock import AsyncMock, patch

from app.core.errors import (
    OllamaGenerationError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.services.llm.ollama_client import OllamaClient


@pytest.mark.asyncio
async def test_ollama_client_check_health_success():
    client = OllamaClient()
    mock_resp = httpx.Response(200, json={"models": []})
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        assert await client.check_health() is True


@pytest.mark.asyncio
async def test_ollama_client_check_health_failure():
    client = OllamaClient()
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("Connection refused")):
        assert await client.check_health() is False


@pytest.mark.asyncio
async def test_ollama_client_list_models_success():
    client = OllamaClient()
    mock_data = {
        "models": [
            {"name": "qwen3:8b", "model": "qwen3:8b"},
            {"name": "mistral:latest", "model": "mistral:latest"},
        ]
    }
    mock_resp = httpx.Response(200, json=mock_data)
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        models = await client.list_models()
        assert models == ["qwen3:8b", "mistral:latest"]


@pytest.mark.asyncio
async def test_ollama_client_list_models_unavailable():
    client = OllamaClient()
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(OllamaUnavailableError) as exc_info:
            await client.list_models()
        assert exc_info.value.status_code == 503
        assert "Ensure Ollama is running" in exc_info.value.message


@pytest.mark.asyncio
async def test_ollama_client_validate_availability_success():
    client = OllamaClient(model="qwen3:8b")
    mock_data = {"models": [{"name": "qwen3:8b"}, {"name": "phi3:mini"}]}
    mock_resp = httpx.Response(200, json=mock_data)
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        # Should not raise
        await client.validate_availability("qwen3:8b")
        await client.validate_availability("phi3:mini")


@pytest.mark.asyncio
async def test_ollama_client_validate_availability_missing_model():
    client = OllamaClient(model="nonexistent_model:latest")
    mock_data = {"models": [{"name": "qwen3:8b"}, {"name": "phi3:mini"}]}
    mock_resp = httpx.Response(200, json=mock_data)
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(OllamaModelNotFoundError) as exc_info:
            await client.validate_availability("nonexistent_model:latest")
        assert exc_info.value.status_code == 503
        assert "ollama pull nonexistent_model:latest" in exc_info.value.message


@pytest.mark.asyncio
async def test_ollama_client_generate_success():
    client = OllamaClient(model="qwen3:8b")
    mock_data = {
        "model": "qwen3:8b",
        "response": "<think>Thinking through steps...</think>Employees receive 20 days of annual leave.",
        "done": True,
    }
    mock_resp = httpx.Response(200, json=mock_data)
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        result = await client.generate(
            prompt="How many leave days?",
            system="Answer strictly with facts.",
        )
        assert result == "Employees receive 20 days of annual leave."
        assert "<think>" not in result


@pytest.mark.asyncio
async def test_ollama_client_generate_timeout():
    client = OllamaClient(model="qwen3:8b", timeout=5.0)
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("Read timed out")):
        with pytest.raises(OllamaTimeoutError) as exc_info:
            await client.generate(prompt="Question")
        assert exc_info.value.status_code == 504
        assert "timed out after 5.0 seconds" in exc_info.value.message


@pytest.mark.asyncio
async def test_ollama_client_generate_connection_error():
    client = OllamaClient(model="qwen3:8b")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.ConnectError("Failed to connect")):
        with pytest.raises(OllamaUnavailableError) as exc_info:
            await client.generate(prompt="Question")
        assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_ollama_client_generate_server_error():
    client = OllamaClient(model="qwen3:8b")
    mock_resp = httpx.Response(500, text="Internal engine crash")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(OllamaGenerationError) as exc_info:
            await client.generate(prompt="Question")
        assert exc_info.value.status_code == 502
        assert "HTTP 500" in exc_info.value.message
