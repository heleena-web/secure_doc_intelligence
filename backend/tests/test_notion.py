import uuid
from unittest.mock import AsyncMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.errors import NotionAuthError, NotionPageNotFoundError, NotionRateLimitError
from app.core.security import create_access_token
from app.main import app
from app.models.auth import UserRecord
from app.models.document import DocumentType, IngestionStatus
from app.models.search import SearchMode, SearchQuery
from app.services.notion.client import NotionClient
from app.services.notion.parser import NotionParser
from app.services.notion.service import NotionService
from app.services.retrieval.search_service import SearchService
from app.storage.database import get_db_connection, init_db
from app.storage.repository import DocumentRepository


# =========================================================================
# Mock Fixtures & Helpers
# =========================================================================

MOCK_BOT_RESPONSE = {
    "object": "user",
    "id": "bot-uuid-1234",
    "name": "DocIntelligence Integration",
    "type": "bot",
    "bot": {
        "workspace_name": "Acme Knowledge Base",
    },
}

MOCK_PAGE_1 = {
    "object": "page",
    "id": "page-uuid-1111-2222-3333-444455556666",
    "created_time": "2026-10-01T10:00:00.000Z",
    "last_edited_time": "2026-10-10T12:00:00.000Z",
    "archived": False,
    "url": "https://notion.so/acme/Engineering-Handbook-111122223333444455556666",
    "properties": {
        "title": {
            "id": "title",
            "type": "title",
            "title": [
                {"type": "text", "text": {"content": "Engineering Handbook"}, "plain_text": "Engineering Handbook"}
            ],
        }
    },
}

MOCK_BLOCKS_PAGE_1 = [
    {
        "id": "b-1",
        "type": "heading_1",
        "heading_1": {
            "rich_text": [{"type": "text", "plain_text": "Overview & Architecture"}]
        },
    },
    {
        "id": "b-2",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {"type": "text", "plain_text": "All services must communicate over mutual TLS and respect tenant ACLs."}
            ]
        },
    },
    {
        "id": "b-3",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "plain_text": "Security Checklist"}]
        },
    },
    {
        "id": "b-4",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [{"type": "text", "plain_text": "Rotate API keys every 90 days."}]
        },
    },
    {
        "id": "b-5",
        "type": "to_do",
        "to_do": {
            "checked": True,
            "rich_text": [{"type": "text", "plain_text": "Enable zero-trust network policies."}],
        },
    },
    {
        "id": "b-6",
        "type": "code",
        "code": {
            "language": "python",
            "rich_text": [{"type": "text", "plain_text": "def verify_token(t): return True"}],
        },
    },
    {
        "id": "b-7",
        "type": "callout",
        "callout": {
            "icon": {"emoji": "🔒"},
            "rich_text": [{"type": "text", "plain_text": "Never commit credentials to Git repositories."}],
        },
    },
    {
        "id": "b-8",
        "type": "table",
        "has_children": True,
        "child_blocks": [
            {
                "id": "b-8-r1",
                "type": "table_row",
                "table_row": {
                    "cells": [
                        [{"plain_text": "Service"}],
                        [{"plain_text": "Port"}],
                        [{"plain_text": "Protocol"}],
                    ]
                },
            },
            {
                "id": "b-8-r2",
                "type": "table_row",
                "table_row": {
                    "cells": [
                        [{"plain_text": "API Gateway"}],
                        [{"plain_text": "8000"}],
                        [{"plain_text": "HTTPS"}],
                    ]
                },
            },
        ],
    },
    {
        "id": "b-9",
        "type": "unsupported_custom_widget",
        "unsupported_custom_widget": {"unknown": "data"},
    },
]


# =========================================================================
# 1. NotionClient Tests
# =========================================================================

@pytest.mark.asyncio
async def test_notion_client_get_bot_user_success():
    client = NotionClient(token="secret_test_token_123")
    mock_res = httpx.Response(200, json=MOCK_BOT_RESPONSE)

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_res):
        bot = await client.get_bot_user()
        assert bot["id"] == "bot-uuid-1234"
        assert bot["bot"]["workspace_name"] == "Acme Knowledge Base"


@pytest.mark.asyncio
async def test_notion_client_auth_failure_401():
    client = NotionClient(token="secret_revoked_token")
    mock_res = httpx.Response(401, json={"message": "API token is invalid."})

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_res):
        with pytest.raises(NotionAuthError):
            await client.get_bot_user()


@pytest.mark.asyncio
async def test_notion_client_rate_limit_429():
    client = NotionClient(token="secret_test_token")
    mock_res = httpx.Response(429, headers={"Retry-After": "5"}, json={"message": "Rate limited."})

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_res):
        with pytest.raises(NotionRateLimitError) as exc_info:
            await client.get_bot_user()
        assert exc_info.value.status_code == 429
        assert exc_info.value.details.get("retry_after") == 5.0


@pytest.mark.asyncio
async def test_notion_client_page_not_found_404():
    client = NotionClient(token="secret_test_token")
    mock_res = httpx.Response(404, json={"message": "Could not find page."})

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_res):
        with pytest.raises(NotionPageNotFoundError):
            await client.get_page("non-existent-page-id")


@pytest.mark.asyncio
async def test_notion_client_pagination_fetch_all_pages():
    client = NotionClient(token="secret_test_token")

    page_1_res = {
        "results": [MOCK_PAGE_1],
        "has_more": True,
        "next_cursor": "cursor_page_2",
    }
    mock_page_2 = dict(MOCK_PAGE_1, id="page-2222", url="https://notion.so/page-2")
    page_2_res = {
        "results": [mock_page_2],
        "has_more": False,
        "next_cursor": None,
    }

    mock_post = AsyncMock()
    mock_post.side_effect = [
        httpx.Response(200, json=page_1_res),
        httpx.Response(200, json=page_2_res),
    ]

    with patch("httpx.AsyncClient.post", mock_post):
        pages = await client.fetch_all_pages()
        assert len(pages) == 2
        assert pages[0]["id"] == MOCK_PAGE_1["id"]
        assert pages[1]["id"] == "page-2222"


# =========================================================================
# 2. NotionParser Tests
# =========================================================================

def test_notion_parser_supported_and_unsupported_blocks():
    parser = NotionParser()
    parsed = parser.parse_page(MOCK_PAGE_1, MOCK_BLOCKS_PAGE_1)

    assert parsed.filename == "Engineering Handbook"
    assert parsed.file_type == DocumentType.NOTION.value
    assert parsed.metadata["source_type"] == "notion"
    assert parsed.metadata["notion_page_id"] == MOCK_PAGE_1["id"]
    assert "https://notion.so/acme" in parsed.metadata["url"]

    # Verify blocks extracted
    block_texts = [b.text for b in parsed.blocks]
    assert any("# Overview & Architecture" in t for t in block_texts)
    assert any("All services must communicate over mutual TLS" in t for t in block_texts)
    assert any("## Security Checklist" in t for t in block_texts)
    assert any("• Rotate API keys every 90 days." in t for t in block_texts)
    assert any("[x] Enable zero-trust network policies." in t for t in block_texts)
    assert any("```python" in t for t in block_texts)
    assert any("[🔒] Never commit credentials" in t for t in block_texts)
    assert any("| Service | Port | Protocol |" in t for t in block_texts)


# =========================================================================
# 3. NotionService End-to-End Synchronization & Hybrid Retrieval Tests
# =========================================================================

@pytest.mark.asyncio
async def test_notion_service_sync_and_hybrid_retrieval(tmp_path):
    db_path = tmp_path / "notion_test.db"
    await init_db(db_path)

    async with get_db_connection(db_path) as db:
        repo = DocumentRepository(db)
        await repo.ensure_workspace("ws-notion", "Notion Test Workspace")

        user_id = str(uuid.uuid4())
        await repo.create_user(
            UserRecord(
                id=user_id,
                username="alice_notion",
                email="alice@acme.com",
                hashed_password="hash",
                created_at="2026-10-01",
            )
        )
        await repo.add_workspace_member("ws-notion", user_id, "admin")

        search_svc = SearchService()
        notion_svc = NotionService(search_svc=search_svc)

        # Mock NotionClient responses
        with patch.object(NotionClient, "get_bot_user", new_callable=AsyncMock, return_value=MOCK_BOT_RESPONSE), \
             patch.object(NotionClient, "fetch_all_pages", new_callable=AsyncMock, return_value=[MOCK_PAGE_1]), \
             patch.object(NotionClient, "fetch_all_blocks", new_callable=AsyncMock, return_value=MOCK_BLOCKS_PAGE_1):

            # Step 1: Connect
            conn_res = await notion_svc.connect("ws-notion", user_id, "secret_valid_key", repo)
            assert conn_res.is_connected is True
            assert conn_res.workspace_name == "Acme Knowledge Base"

            # Step 2: First Sync
            sync_res = await notion_svc.sync(
                workspace_id="ws-notion",
                user_id=user_id,
                repo=repo,
                is_private=False,
            )
            assert sync_res.pages_synced == 1
            assert sync_res.pages_updated == 0
            assert sync_res.pages_skipped == 0
            assert len(sync_res.errors) == 0

            # Verify document saved in SQLite
            doc_record = await repo.get_document_by_id(
                (await repo.get_notion_page(MOCK_PAGE_1["id"]))["document_id"]
            )
            assert doc_record is not None
            assert doc_record.file_type == DocumentType.NOTION.value
            assert doc_record.status == IngestionStatus.INDEXED.value

            # Verify chunks in SQLite
            chunks = await repo.get_chunks_by_document(doc_record.id)
            assert len(chunks) > 0
            assert chunks[0].metadata.get("source_type") == "notion"
            assert chunks[0].metadata.get("url") == MOCK_PAGE_1["url"]
            assert "Notion: Engineering Handbook" in chunks[0].citation_label

            # Step 3: Hybrid Retrieval Verification
            query = SearchQuery(
                query="mutual TLS tenant ACLs",
                mode=SearchMode.HYBRID_RERANK,
                workspace_id="ws-notion",
                top_k=3,
            )
            search_out = search_svc.search(query)
            assert len(search_out.results) > 0
            top_hit = search_out.results[0]
            assert "Engineering Handbook" in top_hit.filename
            assert top_hit.metadata.get("source_type") == "notion"
            assert top_hit.metadata.get("url") == MOCK_PAGE_1["url"]


@pytest.mark.asyncio
async def test_notion_duplicate_prevention_on_repeated_sync(tmp_path):
    db_path = tmp_path / "notion_dedup.db"
    await init_db(db_path)

    async with get_db_connection(db_path) as db:
        repo = DocumentRepository(db)
        await repo.ensure_workspace("ws-dedup", "Dedup Workspace")
        user_id = str(uuid.uuid4())
        await repo.create_user(UserRecord(id=user_id, username="bob", email="bob@test.com", hashed_password="h", created_at="2026-10-01"))
        await repo.add_workspace_member("ws-dedup", user_id, "admin")

        notion_svc = NotionService()

        with patch.object(NotionClient, "get_bot_user", new_callable=AsyncMock, return_value=MOCK_BOT_RESPONSE), \
             patch.object(NotionClient, "fetch_all_pages", new_callable=AsyncMock, return_value=[MOCK_PAGE_1]), \
             patch.object(NotionClient, "fetch_all_blocks", new_callable=AsyncMock, return_value=MOCK_BLOCKS_PAGE_1):

            await notion_svc.connect("ws-dedup", user_id, "token_xyz", repo)

            # First sync -> 1 synced
            res1 = await notion_svc.sync("ws-dedup", user_id, repo)
            assert res1.pages_synced == 1
            assert res1.pages_skipped == 0

            # Second sync without changes -> 1 skipped, 0 synced
            res2 = await notion_svc.sync("ws-dedup", user_id, repo)
            assert res2.pages_synced == 0
            assert res2.pages_skipped == 1

            # Documents count in SQLite should remain exactly 1
            docs = await repo.list_documents("ws-dedup")
            assert len(docs) == 1


@pytest.mark.asyncio
async def test_notion_page_update_replaces_stale_chunks(tmp_path):
    db_path = tmp_path / "notion_update.db"
    await init_db(db_path)

    async with get_db_connection(db_path) as db:
        repo = DocumentRepository(db)
        await repo.ensure_workspace("ws-up", "Update Workspace")
        user_id = str(uuid.uuid4())
        await repo.create_user(UserRecord(id=user_id, username="charlie", email="charlie@test.com", hashed_password="h", created_at="2026-10-01"))
        await repo.add_workspace_member("ws-up", user_id, "admin")

        search_svc = SearchService()
        notion_svc = NotionService(search_svc=search_svc)

        # Initial Page Content
        with patch.object(NotionClient, "get_bot_user", new_callable=AsyncMock, return_value=MOCK_BOT_RESPONSE), \
             patch.object(NotionClient, "fetch_all_pages", new_callable=AsyncMock, return_value=[MOCK_PAGE_1]), \
             patch.object(NotionClient, "fetch_all_blocks", new_callable=AsyncMock, return_value=MOCK_BLOCKS_PAGE_1):

            await notion_svc.connect("ws-up", user_id, "token_xyz", repo)
            await notion_svc.sync("ws-up", user_id, repo)

        # Updated Page Content with newer timestamp
        updated_page = dict(MOCK_PAGE_1, last_edited_time="2026-10-11T16:00:00.000Z")
        updated_blocks = [
            {
                "id": "b-updated",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "plain_text": "Updated protocol: All nodes must use WireGuard tunnels."}]
                },
            }
        ]

        with patch.object(NotionClient, "fetch_all_pages", new_callable=AsyncMock, return_value=[updated_page]), \
             patch.object(NotionClient, "fetch_all_blocks", new_callable=AsyncMock, return_value=updated_blocks):

            res_update = await notion_svc.sync("ws-up", user_id, repo)
            assert res_update.pages_updated == 1
            assert res_update.pages_synced == 0

            # Verify updated search returns new content and no stale chunks
            search_res = search_svc.search(
                SearchQuery(query="WireGuard tunnels", mode=SearchMode.BM25, workspace_id="ws-up")
            )
            assert len(search_res.results) > 0
            assert "WireGuard tunnels" in search_res.results[0].content


@pytest.mark.asyncio
async def test_notion_deleted_page_cleanup(tmp_path):
    db_path = tmp_path / "notion_del.db"
    await init_db(db_path)

    async with get_db_connection(db_path) as db:
        repo = DocumentRepository(db)
        await repo.ensure_workspace("ws-del", "Delete Workspace")
        user_id = str(uuid.uuid4())
        await repo.create_user(UserRecord(id=user_id, username="dave", email="dave@test.com", hashed_password="h", created_at="2026-10-01"))
        await repo.add_workspace_member("ws-del", user_id, "admin")

        search_svc = SearchService()
        notion_svc = NotionService(search_svc=search_svc)

        # Sync initial page
        with patch.object(NotionClient, "get_bot_user", new_callable=AsyncMock, return_value=MOCK_BOT_RESPONSE), \
             patch.object(NotionClient, "fetch_all_pages", new_callable=AsyncMock, return_value=[MOCK_PAGE_1]), \
             patch.object(NotionClient, "fetch_all_blocks", new_callable=AsyncMock, return_value=MOCK_BLOCKS_PAGE_1):

            await notion_svc.connect("ws-del", user_id, "token_xyz", repo)
            await notion_svc.sync("ws-del", user_id, repo)
            assert len(await repo.list_notion_pages("ws-del")) == 1

        # Now simulate page deletion on Notion (returns empty list)
        with patch.object(NotionClient, "fetch_all_pages", new_callable=AsyncMock, return_value=[]):
            del_res = await notion_svc.sync("ws-del", user_id, repo)
            assert del_res.pages_removed == 1

            # Assert purged from SQLite
            assert len(await repo.list_notion_pages("ws-del")) == 0
            assert len(await repo.list_documents("ws-del")) == 0

            # Assert purged from search index
            search_res = search_svc.search(
                SearchQuery(query="mutual TLS", mode=SearchMode.BM25, workspace_id="ws-del")
            )
            assert len(search_res.results) == 0


@pytest.mark.asyncio
async def test_notion_cross_user_and_cross_workspace_access_control(tmp_path):
    db_path = tmp_path / "notion_security.db"
    await init_db(db_path)

    async with get_db_connection(db_path) as db:
        repo = DocumentRepository(db)
        await repo.ensure_workspace("ws-alpha", "Workspace Alpha")
        await repo.ensure_workspace("ws-beta", "Workspace Beta")

        user_alice = str(uuid.uuid4())
        user_mallory = str(uuid.uuid4())

        await repo.create_user(UserRecord(id=user_alice, username="alice_sec", email="alice_sec@test.com", hashed_password="h", created_at="2026-10-01"))
        await repo.create_user(UserRecord(id=user_mallory, username="mallory", email="mallory@test.com", hashed_password="h", created_at="2026-10-01"))

        await repo.add_workspace_member("ws-alpha", user_alice, "member")
        await repo.add_workspace_member("ws-beta", user_mallory, "member")

        notion_svc = NotionService()

        # Alice syncs a PRIVATE Notion page in Workspace Alpha
        with patch.object(NotionClient, "get_bot_user", new_callable=AsyncMock, return_value=MOCK_BOT_RESPONSE), \
             patch.object(NotionClient, "fetch_all_pages", new_callable=AsyncMock, return_value=[MOCK_PAGE_1]), \
             patch.object(NotionClient, "fetch_all_blocks", new_callable=AsyncMock, return_value=MOCK_BLOCKS_PAGE_1):

            await notion_svc.connect("ws-alpha", user_alice, "token_alpha", repo)
            await notion_svc.sync("ws-alpha", user_alice, repo, is_private=True)

        tracked_page = await repo.get_notion_page(MOCK_PAGE_1["id"])
        notion_doc_id = tracked_page["document_id"]

        # 1. Mallory in ws-beta cannot access Alice's document
        mallory_docs_in_beta = await repo.get_user_permitted_document_ids(user_mallory, "ws-beta")
        assert notion_doc_id not in mallory_docs_in_beta

        # 2. Mallory has no access to ws-alpha
        mallory_docs_in_alpha = await repo.get_user_permitted_document_ids(user_mallory, "ws-alpha")
        assert notion_doc_id not in mallory_docs_in_alpha

        # 3. Alice owns the document and has access
        alice_docs = await repo.get_user_permitted_document_ids(user_alice, "ws-alpha")
        assert notion_doc_id in alice_docs


# =========================================================================
# 4. FastAPI Notion Router Endpoints Tests
# =========================================================================

@pytest.mark.asyncio
async def test_notion_api_endpoints_workflow(async_client):
    with patch.object(NotionClient, "get_bot_user", new_callable=AsyncMock, return_value=MOCK_BOT_RESPONSE), \
         patch.object(NotionClient, "fetch_all_pages", new_callable=AsyncMock, return_value=[MOCK_PAGE_1]), \
         patch.object(NotionClient, "fetch_all_blocks", new_callable=AsyncMock, return_value=MOCK_BLOCKS_PAGE_1):

        # 1. Initial Status -> Disconnected
        res_status = await async_client.get("/api/v1/notion/status?workspace_id=default")
        assert res_status.status_code == 200
        assert "is_connected" in res_status.json()

        # 2. Connect
        res_conn = await async_client.post(
            "/api/v1/notion/connect",
            json={"workspace_id": "default", "api_key": "secret_mock_token_123"},
        )
        assert res_conn.status_code == 200
        assert res_conn.json()["is_connected"] is True
        assert res_conn.json()["workspace_name"] == "Acme Knowledge Base"
        # Ensure secret is NEVER leaked in response
        assert "secret_mock_token" not in res_conn.text

        # 3. Discover Pages
        res_pages = await async_client.get("/api/v1/notion/pages?workspace_id=default")
        assert res_pages.status_code == 200
        pages = res_pages.json()
        assert len(pages) == 1
        assert pages[0]["title"] == "Engineering Handbook"

        # 4. Sync
        res_sync = await async_client.post(
            "/api/v1/notion/sync",
            json={"workspace_id": "default", "is_private": False},
        )
        assert res_sync.status_code == 200
        sync_data = res_sync.json()
        assert sync_data["pages_synced"] == 1
        assert sync_data["total_pages_found"] == 1

        # 5. Disconnect
        res_disc = await async_client.post("/api/v1/notion/disconnect?workspace_id=default")
        assert res_disc.status_code == 200
