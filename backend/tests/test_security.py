from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token, decode_access_token, hash_password
from app.main import app
from app.models.auth import PermissionLevel, UserRole
from app.storage.database import get_db_connection


@pytest.mark.asyncio
async def test_auth_registration_and_argon2id_hashing(unauthenticated_client: AsyncClient):
    # Register new user
    reg_res = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={
            "username": "alice_secure",
            "email": "alice@secure.org",
            "password": "StrongPassword123!",
            "full_name": "Alice Secure",
        },
    )
    assert reg_res.status_code == 201
    reg_data = reg_res.json()
    assert "access_token" in reg_data
    assert reg_data["user"]["username"] == "alice_secure"

    # Verify password in database is hashed with Argon2id and never plaintext
    async with get_db_connection() as db:
        cursor = await db.execute("SELECT hashed_password FROM users WHERE username = 'alice_secure'")
        row = await cursor.fetchone()
        assert row is not None
        stored_hash = row["hashed_password"]
        assert stored_hash.startswith("$argon2id$")
        assert "StrongPassword123!" not in stored_hash

    # Duplicate username rejection
    dup_res = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={
            "username": "alice_secure",
            "email": "different_email@secure.org",
            "password": "StrongPassword123!",
        },
    )
    assert dup_res.status_code == 409
    assert dup_res.json()["error"] == "UserAlreadyExistsError"


@pytest.mark.asyncio
async def test_auth_login_success_and_failure(unauthenticated_client: AsyncClient):
    # Register user
    await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={
            "username": "bob_secure",
            "email": "bob@secure.org",
            "password": "CorrectPassword123!",
        },
    )

    # Login with valid password
    login_res = await unauthenticated_client.post(
        "/api/v1/auth/login",
        json={"username": "bob_secure", "password": "CorrectPassword123!"},
    )
    assert login_res.status_code == 200
    assert "access_token" in login_res.json()

    # Login with invalid password
    bad_res = await unauthenticated_client.post(
        "/api/v1/auth/login",
        json={"username": "bob_secure", "password": "WrongPassword999!"},
    )
    assert bad_res.status_code == 401
    assert bad_res.json()["error"] == "InvalidCredentialsError"

    # Login with non-existent user
    non_existent = await unauthenticated_client.post(
        "/api/v1/auth/login",
        json={"username": "non_existent_user", "password": "AnyPassword!"},
    )
    assert non_existent.status_code == 401
    assert non_existent.json()["error"] == "InvalidCredentialsError"


@pytest.mark.asyncio
async def test_unauthenticated_requests_rejected(unauthenticated_client: AsyncClient):
    # Protected ingestion endpoint
    ingest = await unauthenticated_client.post(
        "/api/v1/documents/ingest",
        data={"workspace_id": "default"},
    )
    assert ingest.status_code == 401

    # Protected search endpoint
    search = await unauthenticated_client.post(
        "/api/v1/search",
        json={"query": "test", "workspace_id": "default"},
    )
    assert search.status_code == 401

    # Protected QA endpoint
    qa = await unauthenticated_client.post(
        "/api/v1/search/answer",
        json={"question": "test", "workspace_id": "default"},
    )
    assert qa.status_code == 401

    # Protected document metadata
    doc = await unauthenticated_client.get("/api/v1/documents/doc-123")
    assert doc.status_code == 401


@pytest.mark.asyncio
async def test_invalid_expired_tampered_token_rejected(unauthenticated_client: AsyncClient):
    # 1. Malformed token
    res1 = await unauthenticated_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer not-a-valid-jwt-token"},
    )
    assert res1.status_code == 401
    assert res1.json()["error"] == "InvalidTokenError"

    # 2. Expired token
    expired_token = create_access_token(
        subject="user-123",
        expires_delta=timedelta(seconds=-30),
    )
    res2 = await unauthenticated_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert res2.status_code == 401
    assert res2.json()["error"] == "TokenExpiredError"

    # 3. Tampered signature
    valid_token = create_access_token(subject="user-123")
    parts = valid_token.split(".")
    tampered_token = f"{parts[0]}.{parts[1]}.tamperedsignature123"
    res3 = await unauthenticated_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {tampered_token}"},
    )
    assert res3.status_code == 401
    assert res3.json()["error"] == "InvalidTokenError"


@pytest.mark.asyncio
async def test_cross_workspace_isolation_rejected(unauthenticated_client: AsyncClient):
    transport = ASGITransport(app=app)

    # Register User A and create private workspace A
    r_a = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={"username": "user_a", "email": "usera@test.com", "password": "Password123!"},
    )
    token_a = r_a.json()["access_token"]
    async with AsyncClient(transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token_a}"}) as client_a:
        ws_res = await client_a.post("/api/v1/workspaces", json={"id": "ws_alpha", "name": "Alpha Workspace"})
        assert ws_res.status_code == 201

    # Register User B (not a member of ws_alpha)
    r_b = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={"username": "user_b", "email": "userb@test.com", "password": "Password123!"},
    )
    token_b = r_b.json()["access_token"]
    async with AsyncClient(transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token_b}"}) as client_b:
        # User B attempts to search in ws_alpha -> 403 Forbidden
        search_res = await client_b.post(
            "/api/v1/search",
            json={"query": "financials", "workspace_id": "ws_alpha"},
        )
        assert search_res.status_code == 403
        assert search_res.json()["error"] == "WorkspaceAccessDeniedError"

        # User B attempts to ingest into ws_alpha -> 403 Forbidden
        ingest_res = await client_b.post(
            "/api/v1/documents/ingest",
            files={"file": ("doc.txt", b"Hello content", "text/plain")},
            data={"workspace_id": "ws_alpha"},
        )
        assert ingest_res.status_code == 403
        assert ingest_res.json()["error"] == "WorkspaceAccessDeniedError"


@pytest.mark.asyncio
async def test_private_document_isolation_between_users(unauthenticated_client: AsyncClient):
    transport = ASGITransport(app=app)

    # 1. Register User A and User B (both enrolled in 'default' workspace)
    r_a = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={"username": "alice_owner", "email": "alice_owner@test.com", "password": "Password123!"},
    )
    token_a = r_a.json()["access_token"]

    r_b = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={"username": "bob_viewer", "email": "bob_viewer@test.com", "password": "Password123!"},
    )
    token_b = r_b.json()["access_token"]

    async with AsyncClient(transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token_a}"}) as client_a, \
               AsyncClient(transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token_b}"}) as client_b:

        # 2. User A uploads private confidential document
        secret_payload = (
            "CONFIDENTIAL EXECUTIVE PAYLOAD\n\n"
            "Canary Identifier: CANARY_SECRET_ALPHA_99182\n"
            "Projected Q4 dividend payout is 15.5 million dollars.\n"
        ).encode("utf-8")

        ingest_res = await client_a.post(
            "/api/v1/documents/ingest",
            files={"file": ("executive_q4.txt", secret_payload, "text/plain")},
            data={"workspace_id": "default", "is_private": "true"},
        )
        assert ingest_res.status_code == 201
        doc_id = ingest_res.json()["document_id"]

        # 3. User A can retrieve metadata and search the document
        a_doc = await client_a.get(f"/api/v1/documents/{doc_id}")
        assert a_doc.status_code == 200
        assert a_doc.json()["id"] == doc_id

        a_search = await client_a.post(
            "/api/v1/search",
            json={"query": "CANARY_SECRET_ALPHA_99182", "workspace_id": "default"},
        )
        assert a_search.status_code == 200
        assert a_search.json()["total_candidates"] > 0

        # 4. NEGATIVE TEST: User B CANNOT read metadata or chunks
        b_doc = await client_b.get(f"/api/v1/documents/{doc_id}")
        assert b_doc.status_code == 403
        assert b_doc.json()["error"] == "DocumentAccessDeniedError"

        b_chunks = await client_b.get(f"/api/v1/documents/{doc_id}/chunks")
        assert b_chunks.status_code == 403

        # 5. NEGATIVE TEST: User B CANNOT search for the private content (0 results returned)
        b_search = await client_b.post(
            "/api/v1/search",
            json={"query": "CANARY_SECRET_ALPHA_99182", "workspace_id": "default"},
        )
        assert b_search.status_code == 200
        assert b_search.json()["total_candidates"] == 0
        assert len(b_search.json()["results"]) == 0

        # 6. NEGATIVE TEST: User B CANNOT delete User A's document
        b_del = await client_b.delete(f"/api/v1/documents/{doc_id}")
        assert b_del.status_code == 403
        assert b_del.json()["error"] == "DocumentAccessDeniedError"


@pytest.mark.asyncio
async def test_unauthorized_content_never_reaches_reranker_or_ollama(unauthenticated_client: AsyncClient):
    transport = ASGITransport(app=app)

    r_a = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={"username": "uploader_a", "email": "uploader_a@test.com", "password": "Password123!"},
    )
    token_a = r_a.json()["access_token"]

    r_b = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={"username": "searcher_b", "email": "searcher_b@test.com", "password": "Password123!"},
    )
    token_b = r_b.json()["access_token"]

    async with AsyncClient(transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token_a}"}) as client_a, \
               AsyncClient(transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token_b}"}) as client_b:

        # Upload private document
        content = "HIGHLY CLASSIFIED: The vault combination is DELTA-8842.".encode("utf-8")
        await client_a.post(
            "/api/v1/documents/ingest",
            files={"file": ("vault.txt", content, "text/plain")},
            data={"workspace_id": "default", "is_private": "true"},
        )

        # Mock Ollama and Reranker to inspect internal candidate/prompt payloads
        with patch("app.services.llm.ollama_client.OllamaClient.generate", new_callable=AsyncMock) as mock_ollama_gen, \
             patch("app.services.retrieval.reranker.FlashRankReranker.rerank") as mock_rerank:

            mock_rerank.return_value = []
            mock_ollama_gen.return_value = "The available documents do not contain sufficient evidence."

            # User B asks question targeting the secret
            qa_res = await client_b.post(
                "/api/v1/search/answer",
                json={"question": "What is the vault combination?", "workspace_id": "default"},
            )
            assert qa_res.status_code == 200
            data = qa_res.json()
            assert data["status"] == "NO_EVIDENCE_FOUND"
            assert data["citations"] == []
            assert "DELTA-8842" not in data["answer"]

            # CRITICAL ASSERTION: The secret document was pruned PRE-RETRIEVAL;
            # Neither reranker nor Ollama received the unauthorized passage
            if mock_rerank.called:
                for candidate in mock_rerank.call_args[1].get("candidates", []):
                    assert "DELTA-8842" not in candidate.content
            # If 0 candidates were retrieved, Ollama was not even invoked
            if mock_ollama_gen.called:
                prompt_arg = mock_ollama_gen.call_args[1].get("prompt", "")
                assert "DELTA-8842" not in prompt_arg


@pytest.mark.asyncio
async def test_permission_grant_and_immediate_revocation(unauthenticated_client: AsyncClient):
    transport = ASGITransport(app=app)

    # Register User A (owner) and User B (viewer)
    r_a = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={"username": "owner_ann", "email": "ann@test.com", "password": "Password123!"},
    )
    token_a = r_a.json()["access_token"]
    user_a_id = r_a.json()["user"]["id"]

    r_b = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={"username": "viewer_ben", "email": "ben@test.com", "password": "Password123!"},
    )
    token_b = r_b.json()["access_token"]
    user_b_id = r_b.json()["user"]["id"]

    async with AsyncClient(transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token_a}"}) as client_a, \
               AsyncClient(transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token_b}"}) as client_b:

        # 1. Ann uploads private document
        doc_res = await client_a.post(
            "/api/v1/documents/ingest",
            files={"file": ("project_mars.txt", b"Project Mars launch date is June 2027.", "text/plain")},
            data={"workspace_id": "default", "is_private": "true"},
        )
        doc_id = doc_res.json()["document_id"]

        # Ben initially cannot access it
        b_before = await client_b.get(f"/api/v1/documents/{doc_id}")
        assert b_before.status_code == 403

        # 2. Ann grants Ben READ permission
        grant_res = await client_a.post(
            f"/api/v1/documents/{doc_id}/permissions",
            json={"username": "viewer_ben", "permission_level": "read"},
        )
        assert grant_res.status_code == 200
        assert grant_res.json()["permission_level"] == "read"

        # Ben can now read and search the document
        b_after_grant = await client_b.get(f"/api/v1/documents/{doc_id}")
        assert b_after_grant.status_code == 200

        b_search = await client_b.post(
            "/api/v1/search",
            json={"query": "Project Mars launch", "workspace_id": "default"},
        )
        assert b_search.status_code == 200
        assert b_search.json()["total_candidates"] > 0

        # 3. Ann REVOKES Ben's permission
        revoke_res = await client_a.delete(f"/api/v1/documents/{doc_id}/permissions/{user_b_id}")
        assert revoke_res.status_code == 200
        assert revoke_res.json()["revoked"] is True

        # 4. IMMEDIATE EFFECT: Ben's very next request is rejected and returns 0 search results
        b_after_revoke = await client_b.get(f"/api/v1/documents/{doc_id}")
        assert b_after_revoke.status_code == 403
        assert b_after_revoke.json()["error"] == "DocumentAccessDeniedError"

        b_search_revoked = await client_b.post(
            "/api/v1/search",
            json={"query": "Project Mars launch", "workspace_id": "default"},
        )
        assert b_search_revoked.status_code == 200
        assert b_search_revoked.json()["total_candidates"] == 0


@pytest.mark.asyncio
async def test_role_elevation_and_unauthorized_admin_prevented(unauthenticated_client: AsyncClient):
    transport = ASGITransport(app=app)

    # Register standard member user
    r_user = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={"username": "regular_emp", "email": "emp@test.com", "password": "Password123!"},
    )
    token_user = r_user.json()["access_token"]

    async with AsyncClient(transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token_user}"}) as client:
        # Regular member tries to add someone or elevate roles in 'default' workspace -> 403 Forbidden
        elevate_res = await client.post(
            "/api/v1/workspaces/default/members",
            json={"username": "regular_emp", "role": "owner"},
        )
        assert elevate_res.status_code == 403
        assert elevate_res.json()["error"] == "PermissionDeniedError"


@pytest.mark.asyncio
async def test_forged_permitted_doc_ids_sanitized_server_side(unauthenticated_client: AsyncClient):
    """Verifies that malicious clients cannot bypass access controls by injecting unauthorized document IDs into request parameters."""
    transport = ASGITransport(app=app)

    # Register User A and upload private document
    r_a = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={"username": "alice_forgery_test", "email": "alice_f@test.com", "password": "Password123!"},
    )
    token_a = r_a.json()["access_token"]

    r_b = await unauthenticated_client.post(
        "/api/v1/auth/register",
        json={"username": "bob_forgery_test", "email": "bob_f@test.com", "password": "Password123!"},
    )
    token_b = r_b.json()["access_token"]

    async with AsyncClient(transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token_a}"}) as client_a, \
               AsyncClient(transport=transport, base_url="http://test", headers={"Authorization": f"Bearer {token_b}"}) as client_b:

        doc_res = await client_a.post(
            "/api/v1/documents/ingest",
            files={"file": ("classified.txt", b"Secret payload: CODEWORD-ZULU-4411", "text/plain")},
            data={"workspace_id": "default", "is_private": "true"},
        )
        secret_doc_id = doc_res.json()["document_id"]

        # Bob maliciously crafts request with Alice's doc_id in permitted_doc_ids parameter
        forged_search = await client_b.post(
            "/api/v1/search",
            json={
                "query": "CODEWORD-ZULU-4411",
                "workspace_id": "default",
                "permitted_doc_ids": [secret_doc_id],
            },
        )
        assert forged_search.status_code == 200
        # Server must sanitize against SQLite ACL: 0 candidates returned!
        assert forged_search.json()["total_candidates"] == 0

        # Also test forged permitted_doc_ids in QA endpoint
        forged_qa = await client_b.post(
            "/api/v1/search/answer",
            json={
                "question": "What is the secret codeword?",
                "workspace_id": "default",
                "permitted_doc_ids": [secret_doc_id],
            },
        )
        assert forged_qa.status_code == 200
        assert forged_qa.json()["status"] == "NO_EVIDENCE_FOUND"
        assert forged_qa.json()["citations"] == []
