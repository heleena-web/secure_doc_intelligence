import uuid
from datetime import datetime, timezone
import pytest

from app.models.document import (
    ChunkRecord,
    DocumentRecord,
    DocumentType,
    IngestionStatus,
)
from app.storage.database import get_db_connection, init_db
from app.storage.repository import DocumentRepository


@pytest.mark.asyncio
async def test_storage_document_crud_and_duplicate_detection(tmp_path):
    test_db = tmp_path / "storage_test.db"
    await init_db(test_db)

    async with get_db_connection(test_db) as db:
        repo = DocumentRepository(db)

        # 1. Create document
        doc_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        sample_hash = "a" * 64

        doc = DocumentRecord(
            id=doc_id,
            workspace_id="default",
            filename="financial_summary.pdf",
            file_type=DocumentType.PDF.value,
            file_size_bytes=1024,
            sha256_hash=sample_hash,
            status=IngestionStatus.PENDING,
            page_count=3,
            chunk_count=0,
            char_count=500,
            created_at=now,
            updated_at=now,
        )

        created = await repo.create_document(doc)
        assert created.id == doc_id

        # 2. Duplicate detection by hash
        found = await repo.get_document_by_hash("default", sample_hash)
        assert found is not None
        assert found.id == doc_id
        assert found.filename == "financial_summary.pdf"

        # Not found in different workspace
        notFound = await repo.get_document_by_hash("other_workspace", sample_hash)
        assert notFound is None

        # 3. Update status
        await repo.update_document_status(
            document_id=doc_id,
            status=IngestionStatus.INDEXED,
            chunk_count=5,
            char_count=520,
        )
        updated = await repo.get_document_by_id(doc_id)
        assert updated.status == IngestionStatus.INDEXED
        assert updated.chunk_count == 5


@pytest.mark.asyncio
async def test_storage_chunks_and_cascading_deletion(tmp_path):
    test_db = tmp_path / "cascading_test.db"
    await init_db(test_db)

    async with get_db_connection(test_db) as db:
        repo = DocumentRepository(db)

        doc_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        doc = DocumentRecord(
            id=doc_id,
            workspace_id="default",
            filename="contract.docx",
            file_type=DocumentType.DOCX.value,
            file_size_bytes=2048,
            sha256_hash="b" * 64,
            status=IngestionStatus.INDEXED,
            page_count=None,
            chunk_count=2,
            char_count=300,
            created_at=now,
            updated_at=now,
        )
        await repo.create_document(doc)

        # Create two chunks
        chunks = [
            ChunkRecord(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                workspace_id="default",
                chunk_index=0,
                char_start=0,
                char_end=150,
                token_count=30,
                content="Chunk 1 content",
                citation_label="[Doc: contract.docx, Section: Intro]",
                metadata={"section": "Intro"},
                content_hash="c1" + "0" * 62,
            ),
            ChunkRecord(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                workspace_id="default",
                chunk_index=1,
                char_start=151,
                char_end=300,
                token_count=35,
                content="Chunk 2 content",
                citation_label="[Doc: contract.docx, Section: Terms]",
                metadata={"section": "Terms"},
                content_hash="c2" + "0" * 62,
            ),
        ]
        await repo.save_chunks(chunks)

        # Verify chunks retrieved
        saved_chunks = await repo.get_chunks_by_document(doc_id)
        assert len(saved_chunks) == 2
        assert saved_chunks[0].chunk_index == 0
        assert saved_chunks[1].chunk_index == 1

        # Delete document -> cascading foreign key deletes chunks
        deleted = await repo.delete_document(doc_id)
        assert deleted is True

        # Assert document is gone
        assert await repo.get_document_by_id(doc_id) is None
        # Assert chunks are cascaded away
        assert await repo.get_chunks_by_document(doc_id) == []


@pytest.mark.asyncio
async def test_fresh_database_initialization(tmp_path):
    """Verifies that a fresh database initializes all tables, columns, and indexes cleanly."""
    fresh_db = tmp_path / "fresh_init.db"
    await init_db(fresh_db)

    async with get_db_connection(fresh_db) as db:
        # Check tables
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
        required_tables = {"users", "workspaces", "workspace_members", "documents", "document_permissions", "chunks"}
        assert required_tables.issubset(tables)

        # Check documents columns
        col_cursor = await db.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in await col_cursor.fetchall()}
        assert "owner_id" in columns
        assert "is_private" in columns

        # Check index on owner_id exists
        idx_cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in await idx_cursor.fetchall()}
        assert "idx_documents_owner" in indexes

    # Verify idempotency
    await init_db(fresh_db)


@pytest.mark.asyncio
async def test_migration_from_legacy_phase1_schema(tmp_path):
    """Regression test: Verifies that an existing database created with older Phase 1 schema
    (missing owner_id, is_private, and document_permissions) safely migrates without error.
    """
    import aiosqlite
    legacy_db = tmp_path / "legacy_phase1.db"

    # Step 1: Initialize legacy schema missing owner_id, is_private, and document_permissions
    legacy_schema = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE workspaces (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE documents (
        id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        file_type TEXT NOT NULL,
        file_size_bytes INTEGER NOT NULL,
        sha256_hash TEXT NOT NULL,
        status TEXT NOT NULL,
        page_count INTEGER,
        chunk_count INTEGER DEFAULT 0,
        char_count INTEGER DEFAULT 0,
        error_message TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
    );

    CREATE INDEX idx_documents_workspace_hash ON documents(workspace_id, sha256_hash);
    CREATE INDEX idx_documents_workspace_status ON documents(workspace_id, status);

    CREATE TABLE chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        token_count INTEGER NOT NULL,
        content TEXT NOT NULL,
        citation_label TEXT NOT NULL,
        metadata_json TEXT,
        content_hash TEXT NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
    );
    """
    async with aiosqlite.connect(str(legacy_db)) as db:
        await db.executescript(legacy_schema)
        # Seed existing data
        await db.execute(
            "INSERT INTO workspaces (id, name, created_at) VALUES (?, ?, ?)",
            ("ws-legacy", "Legacy Workspace", "2026-10-10T10:00:00Z"),
        )
        await db.execute(
            """
            INSERT INTO documents (
                id, workspace_id, filename, file_type, file_size_bytes, sha256_hash, status,
                page_count, chunk_count, char_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "doc-legacy-1", "ws-legacy", "legacy_doc.pdf", "pdf", 1024, "hash123", "INDEXED",
                1, 1, 100, "2026-10-10T10:00:00Z", "2026-10-10T10:00:00Z"
            ),
        )
        await db.execute(
            """
            INSERT INTO chunks (
                id, document_id, workspace_id, chunk_index, char_start, char_end,
                token_count, content, citation_label, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "chunk-legacy-1", "doc-legacy-1", "ws-legacy", 0, 0, 100,
                20, "Legacy chunk content", "[Doc: legacy_doc.pdf, Page: 1]", "chunkhash123"
            ),
        )
        await db.commit()

    # Step 2: Run init_db (must NOT crash with 'no such column: owner_id')
    await init_db(legacy_db)

    # Step 3: Verify migration outcomes and data preservation
    async with get_db_connection(legacy_db) as db:
        repo = DocumentRepository(db)

        # Check existing document is intact with new columns populated
        migrated_doc = await repo.get_document_by_id("doc-legacy-1")
        assert migrated_doc is not None
        assert migrated_doc.filename == "legacy_doc.pdf"
        assert migrated_doc.owner_id is None
        assert migrated_doc.is_private is True  # Defaulted to True / 1

        # Check existing chunks are intact
        chunks = await repo.get_chunks_by_document("doc-legacy-1")
        assert len(chunks) == 1
        assert chunks[0].content == "Legacy chunk content"

        # Check new tables exist
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in await cursor.fetchall()}
        assert "document_permissions" in tables
        assert "users" in tables
        assert "workspace_members" in tables

        # Check idx_documents_owner index was created
        idx_cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in await idx_cursor.fetchall()}
        assert "idx_documents_owner" in indexes

    # Step 4: Verify idempotency - running init_db again causes no errors
    await init_db(legacy_db)

