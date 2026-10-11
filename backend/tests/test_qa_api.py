import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

from app.core.errors import OllamaModelNotFoundError, OllamaUnavailableError


@pytest.mark.asyncio
async def test_qa_endpoint_no_evidence_found(async_client: AsyncClient):
    response = await async_client.post(
        "/api/v1/search/answer",
        json={"question": "What is the vacation policy?", "workspace_id": "default"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "NO_EVIDENCE_FOUND"
    assert data["has_sufficient_evidence"] is False
    assert "not contain sufficient evidence" in data["answer"].lower()
    assert data["citations"] == []
    assert data["retrieved_chunks_count"] == 0


@pytest.mark.asyncio
async def test_qa_endpoint_with_retrieved_evidence(async_client: AsyncClient, sample_pdf_bytes: bytes):
    # Ingest document first
    ingest_res = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")},
        data={"workspace_id": "default"},
    )
    assert ingest_res.status_code == 201

    # Mock Ollama generation to return grounded answer
    with (
        patch("app.services.llm.ollama_client.OllamaClient.validate_availability", new_callable=AsyncMock, return_value=None),
        patch("app.services.llm.ollama_client.OllamaClient.generate", new_callable=AsyncMock) as mock_gen,
    ):
        mock_gen.return_value = "All employees must complete mandatory security training annually."

        response = await async_client.post(
            "/api/v1/search/answer",
            json={"question": "What security training is required?", "workspace_id": "default"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ANSWERED"
        assert data["has_sufficient_evidence"] is True
        assert data["retrieved_chunks_count"] > 0
        assert len(data["citations"]) > 0
        assert data["citations"][0]["filename"] == "report.pdf"
        assert data["citations"][0]["verified"] is True


@pytest.mark.asyncio
async def test_qa_endpoint_qa_router_alias(async_client: AsyncClient, sample_docx_bytes: bytes):
    # Test that /api/v1/qa/answer route alias works identically
    ingest_res = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("policy.docx", sample_docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"workspace_id": "default"},
    )
    assert ingest_res.status_code == 201

    with (
        patch("app.services.llm.ollama_client.OllamaClient.validate_availability", new_callable=AsyncMock, return_value=None),
        patch("app.services.llm.ollama_client.OllamaClient.generate", new_callable=AsyncMock) as mock_gen,
    ):
        mock_gen.return_value = "Audit logs must be retained for at least 7 years in WORM storage."

        response = await async_client.post(
            "/api/v1/qa/answer",
            json={"question": "How long must audit logs be retained?", "workspace_id": "default"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ANSWERED"
        assert len(data["citations"]) > 0
        assert data["citations"][0]["filename"] == "policy.docx"


@pytest.mark.asyncio
async def test_qa_endpoint_ollama_unavailable_returns_503(async_client: AsyncClient, sample_txt_bytes: bytes):
    await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("guide.txt", sample_txt_bytes, "text/plain")},
        data={"workspace_id": "default"},
    )

    with patch(
        "app.services.llm.ollama_client.OllamaClient.validate_availability",
        new_callable=AsyncMock,
        side_effect=OllamaUnavailableError("http://127.0.0.1:11434"),
    ):
        response = await async_client.post(
            "/api/v1/search/answer",
            json={"question": "What SSH settings are required?", "workspace_id": "default"},
        )
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "OllamaUnavailableError"
        assert "Ensure Ollama is running" in data["detail"]


@pytest.mark.asyncio
async def test_qa_endpoint_model_missing_returns_503(async_client: AsyncClient, sample_txt_bytes: bytes):
    await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("guide.txt", sample_txt_bytes, "text/plain")},
        data={"workspace_id": "default"},
    )

    with patch(
        "app.services.llm.ollama_client.OllamaClient.validate_availability",
        new_callable=AsyncMock,
        side_effect=OllamaModelNotFoundError("qwen3:8b", ["llama3.2:3b"]),
    ):
        response = await async_client.post(
            "/api/v1/search/answer",
            json={"question": "What TLS version is needed?", "workspace_id": "default"},
        )
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "OllamaModelNotFoundError"
        assert "ollama pull qwen3:8b" in data["detail"]
