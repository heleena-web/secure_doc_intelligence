import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.core.config import settings
from app.core.errors import NotionAuthError
from app.models.document import DocumentRecord, DocumentType, IngestionStatus
from app.models.notion import (
    NotionConnectionResponse,
    NotionConnectionStatus,
    NotionPageSummary,
    NotionSyncResponse,
)
from app.services.ingestion.chunker import TokenizerAwareChunker
from app.services.notion.client import NotionClient
from app.services.notion.parser import NotionParser, extract_page_title
from app.services.retrieval.search_service import SearchService, search_service
from app.storage.repository import DocumentRepository


class NotionService:
    """Orchestrates Notion workspace connection, page discovery, incremental sync,
    duplicate prevention, stale page cleanup, and hybrid retrieval index synchronization.
    """

    def __init__(
        self,
        parser: Optional[NotionParser] = None,
        chunker: Optional[TokenizerAwareChunker] = None,
        search_svc: Optional[SearchService] = None,
    ):
        self.parser = parser or NotionParser()
        self.chunker = chunker or TokenizerAwareChunker()
        self.search_service = search_svc or search_service

    def _get_token(self, conn_record: Optional[dict] = None) -> Optional[str]:
        """Resolves token from connection record or system environment configuration."""
        if conn_record and conn_record.get("encrypted_token"):
            return conn_record["encrypted_token"]
        return settings.NOTION_API_KEY

    async def get_connection_status(
        self,
        workspace_id: str,
        repo: DocumentRepository,
    ) -> NotionConnectionResponse:
        """Determines active Notion connection status for a workspace without exposing secrets."""
        conn = await repo.get_notion_connection(workspace_id)
        token = self._get_token(conn)
        pages = await repo.list_notion_pages(workspace_id)

        if not token:
            return NotionConnectionResponse(
                workspace_id=workspace_id,
                status=NotionConnectionStatus.DISCONNECTED,
                is_connected=False,
                workspace_name=None,
                bot_id=None,
                bot_name=None,
                last_synced_at=conn.get("last_synced_at") if conn else None,
                page_count=len(pages),
                has_configured_token=False,
            )

        # If connection record already validated
        if conn and conn.get("status") == "connected":
            return NotionConnectionResponse(
                workspace_id=workspace_id,
                status=NotionConnectionStatus.CONNECTED,
                is_connected=True,
                workspace_name=conn.get("workspace_name"),
                bot_id=conn.get("bot_id"),
                bot_name=conn.get("workspace_name"),
                last_synced_at=conn.get("last_synced_at"),
                page_count=len(pages),
                has_configured_token=bool(token),
            )

        # If token exists in settings but not yet tested/saved in DB
        return NotionConnectionResponse(
            workspace_id=workspace_id,
            status=NotionConnectionStatus.DISCONNECTED,
            is_connected=False,
            workspace_name=None,
            bot_id=None,
            bot_name=None,
            last_synced_at=None,
            page_count=len(pages),
            has_configured_token=bool(token),
        )

    async def connect(
        self,
        workspace_id: str,
        user_id: str,
        api_key: Optional[str],
        repo: DocumentRepository,
    ) -> NotionConnectionResponse:
        """Validates token with Notion /v1/users/me and records authorized connection in database."""
        token = (api_key.strip() if api_key and api_key.strip() else None) or settings.NOTION_API_KEY
        if not token:
            raise NotionAuthError("No Notion API key provided and NOTION_API_KEY is not configured in server environment.")

        client = NotionClient(token=token)
        bot_info = await client.get_bot_user()

        bot_id = bot_info.get("id")
        bot_name = bot_info.get("name", "Notion Bot")
        bot_data = bot_info.get("bot", {})
        workspace_name = bot_data.get("workspace_name") or bot_name

        conn_id = str(uuid.uuid4())
        await repo.save_notion_connection(
            connection_id=conn_id,
            workspace_id=workspace_id,
            user_id=user_id,
            bot_id=bot_id,
            workspace_name=workspace_name,
            workspace_icon=None,
            status="connected",
            encrypted_token=api_key.strip() if api_key and api_key.strip() else None,
            last_synced_at=None,
        )

        pages = await repo.list_notion_pages(workspace_id)
        return NotionConnectionResponse(
            workspace_id=workspace_id,
            status=NotionConnectionStatus.CONNECTED,
            is_connected=True,
            workspace_name=workspace_name,
            bot_id=bot_id,
            bot_name=bot_name,
            last_synced_at=None,
            page_count=len(pages),
            has_configured_token=True,
        )

    async def disconnect(
        self,
        workspace_id: str,
        repo: DocumentRepository,
    ) -> bool:
        """Removes workspace connection record from database."""
        return await repo.delete_notion_connection(workspace_id)

    async def list_pages(
        self,
        workspace_id: str,
        repo: DocumentRepository,
        query: str = "",
    ) -> list[NotionPageSummary]:
        """Discovers accessible Notion pages and correlates them with existing synced records."""
        conn = await repo.get_notion_connection(workspace_id)
        token = self._get_token(conn)
        if not token:
            raise NotionAuthError("Notion is not connected. Please connect or configure NOTION_API_KEY.")

        client = NotionClient(token=token)
        raw_pages = await client.fetch_all_pages(query=query)

        tracked = {p["page_id"]: p for p in await repo.list_notion_pages(workspace_id)}
        summaries: list[NotionPageSummary] = []

        for p in raw_pages:
            page_id = p.get("id", "")
            title = extract_page_title(p)
            url = p.get("url", f"https://notion.so/{page_id.replace('-', '')}")
            last_edited_time = p.get("last_edited_time", "")

            tr = tracked.get(page_id)
            is_synced = tr is not None and tr.get("status") == "synced"
            synced_at = tr.get("synced_at") if tr else None
            doc_id = tr.get("document_id") if tr else None

            summaries.append(
                NotionPageSummary(
                    page_id=page_id,
                    title=title,
                    url=url,
                    last_edited_time=last_edited_time,
                    is_synced=is_synced,
                    document_id=doc_id,
                    synced_at=synced_at,
                    status="synced" if is_synced else "available",
                )
            )

        return summaries

    async def sync(
        self,
        workspace_id: str,
        user_id: str,
        repo: DocumentRepository,
        page_ids: Optional[list[str]] = None,
        is_private: bool = True,
        force: bool = False,
    ) -> NotionSyncResponse:
        """Executes full or partial synchronization of Notion knowledge pages."""
        conn = await repo.get_notion_connection(workspace_id)
        token = self._get_token(conn)
        if not token:
            raise NotionAuthError("Notion is not connected. Please connect or configure NOTION_API_KEY.")

        client = NotionClient(token=token)
        raw_pages = await client.fetch_all_pages()

        # Target filtering
        if page_ids is not None:
            target_set = set(page_ids)
            target_pages = [p for p in raw_pages if p.get("id") in target_set]
        else:
            target_pages = raw_pages

        existing_tracked = {p["page_id"]: p for p in await repo.list_notion_pages(workspace_id)}

        pages_synced = 0
        pages_updated = 0
        pages_removed = 0
        pages_skipped = 0
        errors: list[str] = []
        accessible_page_ids = set()

        for page in target_pages:
            page_id = page.get("id")
            if not page_id:
                continue
            accessible_page_ids.add(page_id)

            title = extract_page_title(page)
            url = page.get("url", f"https://notion.so/{page_id.replace('-', '')}")
            last_edited_time = page.get("last_edited_time", "")

            tracked = existing_tracked.get(page_id)

            # Check if page is unmodified (skip only if not forcing resync)
            if not force and tracked and tracked.get("last_edited_time") == last_edited_time and tracked.get("status") == "synced":
                pages_skipped += 1
                continue

            try:
                # 1. Fetch blocks
                blocks = await client.fetch_all_blocks(page_id)

                # 2. Parse blocks into ParsedDocument
                parsed_doc = self.parser.parse_page(page, blocks)

                # Compute content hash
                content_payload = "".join(b.text for b in parsed_doc.blocks)
                content_hash = hashlib.sha256(content_payload.encode("utf-8")).hexdigest()

                # If content hash matches tracked hash despite timestamp change
                if not force and tracked and tracked.get("content_hash") == content_hash:
                    pages_skipped += 1
                    continue

                now_iso = datetime.now(timezone.utc).isoformat()
                is_update = tracked is not None

                if is_update:
                    doc_id = tracked["document_id"]
                    # Evict previous vectors and BM25 chunks
                    self.search_service.remove_document(doc_id)
                    # Purge old chunks in SQLite
                    await repo.db.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
                    await repo.db.commit()
                else:
                    doc_id = str(uuid.uuid4())

                # 3. Create or update DocumentRecord in SQLite
                char_count = parsed_doc.char_count
                doc_record = DocumentRecord(
                    id=doc_id,
                    workspace_id=workspace_id,
                    filename=f"{title}.notion",
                    file_type=DocumentType.NOTION.value,
                    file_size_bytes=char_count,
                    sha256_hash=content_hash,
                    status=IngestionStatus.INDEXED,
                    owner_id=user_id,
                    is_private=is_private,
                    page_count=1,
                    chunk_count=0,
                    char_count=char_count,
                    error_message=None,
                    created_at=now_iso,
                    updated_at=now_iso,
                )

                if is_update:
                    await repo.update_document_status(
                        document_id=doc_id,
                        status=IngestionStatus.INDEXED,
                        char_count=char_count,
                        page_count=1,
                    )
                else:
                    await repo.create_document(doc_record)

                # 4. Tokenizer-Aware Chunking
                chunks = self.chunker.chunk_document(doc_id, workspace_id, parsed_doc)

                # Decorate chunks with verified Notion URL metadata
                for c in chunks:
                    c.metadata["source_type"] = "notion"
                    c.metadata["url"] = url
                    c.metadata["notion_page_id"] = page_id
                    c.metadata["page_title"] = title

                # 5. Persist chunks in SQLite
                await repo.save_chunks(chunks)
                await repo.update_document_status(
                    document_id=doc_id,
                    status=IngestionStatus.INDEXED,
                    chunk_count=len(chunks),
                )

                # 6. Index chunks into BM25 and ChromaDB
                self.search_service.index_document_chunks(chunks, doc_record.filename)

                # 7. Track in notion_pages table
                await repo.save_notion_page(
                    id=str(uuid.uuid4()),
                    page_id=page_id,
                    document_id=doc_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    title=title,
                    url=url,
                    last_edited_time=last_edited_time,
                    content_hash=content_hash,
                    synced_at=now_iso,
                    status="synced",
                )

                if is_update:
                    pages_updated += 1
                else:
                    pages_synced += 1

            except Exception as e:
                errors.append(f"Failed syncing page '{title}' ({page_id}): {str(e)}")

        # 8. Detect deleted or revoked pages (only if full sync)
        if page_ids is None:
            for page_id, tracked in existing_tracked.items():
                if page_id not in accessible_page_ids:
                    try:
                        old_doc_id = tracked["document_id"]
                        self.search_service.remove_document(old_doc_id)
                        await repo.delete_document(old_doc_id)
                        await repo.delete_notion_page(page_id)
                        pages_removed += 1
                    except Exception as e:
                        errors.append(f"Failed removing stale page '{tracked.get('title')}' ({page_id}): {str(e)}")

        # Update last_synced_at on connection record
        now_sync = datetime.now(timezone.utc).isoformat()
        if conn:
            await repo.save_notion_connection(
                connection_id=conn["id"],
                workspace_id=workspace_id,
                user_id=user_id,
                bot_id=conn.get("bot_id"),
                workspace_name=conn.get("workspace_name"),
                workspace_icon=conn.get("workspace_icon"),
                status="connected",
                last_synced_at=now_sync,
            )

        return NotionSyncResponse(
            workspace_id=workspace_id,
            total_pages_found=len(target_pages),
            pages_synced=pages_synced,
            pages_updated=pages_updated,
            pages_removed=pages_removed,
            pages_skipped=pages_skipped,
            synced_at=now_sync,
            errors=errors,
        )


# Global singleton instance
notion_service = NotionService()
