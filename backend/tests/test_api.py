import os
from pathlib import Path
import pytest
from httpx import AsyncClient

from app.core.config import settings


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_ingest_valid_pdf(async_client: AsyncClient, sample_pdf_bytes: bytes):
    response = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")},
        data={"workspace_id": "default"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "report.pdf"
    assert data["file_type"] == "pdf"
    assert data["page_count"] == 2
    assert data["chunk_count"] >= 2
    assert data["char_count"] > 0
    assert data["is_duplicate"] is False
    assert len(data["sample_citations"]) > 0
    assert any("Page:" in c for c in data["sample_citations"])

    # Verify retrieval by document ID
    doc_id = data["document_id"]
    get_res = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == doc_id

    # Verify retrieval of chunks
    chunks_res = await async_client.get(f"/api/v1/documents/{doc_id}/chunks")
    assert chunks_res.status_code == 200
    assert len(chunks_res.json()) == data["chunk_count"]


@pytest.mark.asyncio
async def test_ingest_valid_docx(async_client: AsyncClient, sample_docx_bytes: bytes):
    response = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("policy.docx", sample_docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"workspace_id": "default"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "policy.docx"
    assert data["file_type"] == "docx"
    assert data["chunk_count"] >= 1
    assert any("Section:" in c for c in data["sample_citations"])


@pytest.mark.asyncio
async def test_ingest_valid_txt(async_client: AsyncClient, sample_txt_bytes: bytes):
    response = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("config.txt", sample_txt_bytes, "text/plain")},
        data={"workspace_id": "default"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "config.txt"
    assert data["file_type"] == "txt"
    assert any("Lines:" in c for c in data["sample_citations"])


@pytest.mark.asyncio
async def test_duplicate_ingestion_behavior(async_client: AsyncClient, sample_pdf_bytes: bytes):
    # First upload
    res1 = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("doc_v1.pdf", sample_pdf_bytes, "application/pdf")},
        data={"workspace_id": "default"},
    )
    assert res1.status_code == 201
    data1 = res1.json()
    assert data1["is_duplicate"] is False

    # Second upload of identical content
    res2 = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("doc_v1_reupload.pdf", sample_pdf_bytes, "application/pdf")},
        data={"workspace_id": "default"},
    )
    assert res2.status_code == 201
    data2 = res2.json()
    assert data2["is_duplicate"] is True
    # Document ID references the already existing indexed record
    assert data2["document_id"] == data1["document_id"]
    assert data2["sha256_hash"] == data1["sha256_hash"]


@pytest.mark.asyncio
async def test_empty_file_rejection(async_client: AsyncClient):
    response = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("empty.txt", b"", "text/plain")},
        data={"workspace_id": "default"},
    )
    assert response.status_code == 422
    data = response.json()
    assert "no readable text" in data["detail"]


@pytest.mark.asyncio
async def test_malformed_file_rejection(async_client: AsyncClient):
    response = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("broken.pdf", b"%PDF-1.4 RANDOM_CORRUPT_BYTES", "application/pdf")},
        data={"workspace_id": "default"},
    )
    assert response.status_code == 422
    data = response.json()
    assert "Invalid or corrupt" in data["detail"]


@pytest.mark.asyncio
async def test_unsupported_file_extension(async_client: AsyncClient):
    response = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("malicious.exe", b"MZ\x90\x00BIN", "application/octet-stream")},
        data={"workspace_id": "default"},
    )
    assert response.status_code == 415
    data = response.json()
    assert "Unsupported file type '.exe'" in data["detail"]


@pytest.mark.asyncio
async def test_file_size_limit_enforced(async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    # Set limit to 100 bytes
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 100)
    oversized_data = b"A" * 200

    response = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("large.txt", oversized_data, "text/plain")},
        data={"workspace_id": "default"},
    )
    assert response.status_code == 413
    data = response.json()
    assert "exceeds maximum permitted limit" in data["detail"]


@pytest.mark.asyncio
async def test_temporary_files_cleanup_after_failure(async_client: AsyncClient):
    # Uploading a corrupt file that triggers parser failure
    await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("test_clean.docx", b"NOT_A_VALID_DOCX_FILE", "application/octet-stream")},
        data={"workspace_id": "default"},
    )

    # Assert upload staging directory contains no leftover files
    staging_files = list(settings.UPLOAD_DIR.glob("temp_*"))
    assert len(staging_files) == 0


@pytest.mark.asyncio
async def test_search_endpoint_full_pipeline(async_client: AsyncClient, sample_pdf_bytes: bytes):
    # Ingest PDF
    upload_res = await async_client.post(
        "/api/v1/documents/ingest",
        files={"file": ("cyber_policy.pdf", sample_pdf_bytes, "application/pdf")},
        data={"workspace_id": "default"},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # Search with hybrid rerank
    search_res = await async_client.post(
        "/api/v1/search",
        json={
            "query": "zero-trust least-privilege token access",
            "mode": "hybrid_rerank",
            "top_k": 3,
            "workspace_id": "default",
        },
    )
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["total_candidates"] > 0
    top = data["results"][0]
    assert "Zero-trust" in top["content"] or "least-privilege" in top["content"]
    assert top["document_id"] == doc_id
    assert "Page:" in top["citation_label"]
    assert top["rerank_score"] is not None

    # Delete document
    del_res = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 204

    # Search again: must return 0 results
    search_after_del = await async_client.post(
        "/api/v1/search",
        json={
            "query": "zero-trust least-privilege token access",
            "mode": "hybrid_rerank",
            "top_k": 3,
            "workspace_id": "default",
        },
    )
    assert search_after_del.status_code == 200
    assert search_after_del.json()["total_candidates"] == 0

