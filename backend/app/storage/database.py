import aiosqlite
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from app.core.config import settings


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    full_name TEXT,
    is_active INTEGER DEFAULT 1,
    is_superuser INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspace_members (
    workspace_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (workspace_id, user_id),
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ws_members_user ON workspace_members(user_id);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    sha256_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    owner_id TEXT,
    is_private INTEGER DEFAULT 1,
    page_count INTEGER,
    chunk_count INTEGER DEFAULT 0,
    char_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_workspace_hash 
ON documents(workspace_id, sha256_hash);

CREATE INDEX IF NOT EXISTS idx_documents_workspace_status 
ON documents(workspace_id, status);

CREATE INDEX IF NOT EXISTS idx_documents_owner 
ON documents(owner_id);

CREATE TABLE IF NOT EXISTS document_permissions (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    permission_level TEXT NOT NULL,
    granted_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(document_id, user_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_doc_perms_user ON document_permissions(user_id);
CREATE INDEX IF NOT EXISTS idx_doc_perms_doc ON document_permissions(document_id);

CREATE TABLE IF NOT EXISTS chunks (
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

CREATE INDEX IF NOT EXISTS idx_chunks_document_index 
ON chunks(document_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_chunks_workspace 
ON chunks(workspace_id);

CREATE TABLE IF NOT EXISTS notion_connections (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    bot_id TEXT,
    workspace_name TEXT,
    workspace_icon TEXT,
    status TEXT NOT NULL,
    encrypted_token TEXT,
    last_synced_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_notion_conn_ws 
ON notion_connections(workspace_id);

CREATE INDEX IF NOT EXISTS idx_notion_conn_user 
ON notion_connections(user_id);

CREATE TABLE IF NOT EXISTS notion_pages (
    id TEXT PRIMARY KEY,
    page_id TEXT UNIQUE NOT NULL,
    document_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    last_edited_time TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    synced_at TEXT NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_notion_pages_page_id 
ON notion_pages(page_id);

CREATE INDEX IF NOT EXISTS idx_notion_pages_ws 
ON notion_pages(workspace_id);

CREATE INDEX IF NOT EXISTS idx_notion_pages_doc 
ON notion_pages(document_id);
"""


@asynccontextmanager
async def get_db_connection(db_path: Path | str | None = None) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Provides an async connection to the SQLite database with WAL and foreign key support."""
    path = db_path or settings.SQLITE_DB_PATH
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(path))
    else:
        conn = await aiosqlite.connect(path)

    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON;")
    if str(path) != ":memory:":
        await conn.execute("PRAGMA journal_mode = WAL;")
        await conn.execute("PRAGMA synchronous = NORMAL;")

    try:
        yield conn
    finally:
        await conn.close()


async def _apply_pre_ddl_migrations(db: aiosqlite.Connection) -> None:
    """Safely upgrades older table schemas before executing indexes or dependent DDL."""
    # Check if documents table exists
    cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents';")
    row = await cursor.fetchone()
    if row:
        # Check current columns in documents table
        col_cursor = await db.execute("PRAGMA table_info(documents);")
        columns = [col[1] for col in await col_cursor.fetchall()]

        if "owner_id" not in columns:
            await db.execute("ALTER TABLE documents ADD COLUMN owner_id TEXT;")
        if "is_private" not in columns:
            await db.execute("ALTER TABLE documents ADD COLUMN is_private INTEGER DEFAULT 1;")
            await db.execute("UPDATE documents SET is_private = 1 WHERE is_private IS NULL;")
        await db.commit()


async def init_db(db_path: Path | str | None = None) -> None:
    """Initializes tables, default workspaces, and applies migrations in SQLite."""
    async with get_db_connection(db_path) as db:
        # Step 1: Pre-DDL migrations for existing databases created with older schemas
        await _apply_pre_ddl_migrations(db)

        # Step 2: Apply complete schema DDL (creates missing tables and indexes)
        await db.executescript(SCHEMA_SQL)

        # Step 3: Ensure default workspace exists
        await db.execute(
            """
            INSERT OR IGNORE INTO workspaces (id, name, created_at)
            VALUES (?, ?, datetime('now'))
            """,
            ("default", "Default Workspace"),
        )
        await db.commit()

