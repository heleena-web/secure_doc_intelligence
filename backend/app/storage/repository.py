import json
import uuid
from datetime import datetime, timezone
from typing import Optional
import aiosqlite

from app.models.auth import (
    DocumentPermissionRecord,
    PermissionLevel,
    PermissionResponse,
    UserRecord,
    UserRole,
    WorkspaceMemberRecord,
    WorkspaceResponse,
)
from app.models.document import ChunkRecord, DocumentRecord, IngestionStatus


class DocumentRepository:
    """Authoritative SQLite repository for users, workspaces, permissions, documents, and chunks."""

    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    # ---------------------------------------------------------
    # User Management
    # ---------------------------------------------------------

    async def create_user(self, user: UserRecord) -> UserRecord:
        """Stores a newly registered user with their Argon2id password hash."""
        await self.db.execute(
            """
            INSERT INTO users (id, username, email, hashed_password, full_name, is_active, is_superuser, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user.id,
                user.username,
                user.email,
                user.hashed_password,
                user.full_name,
                1 if user.is_active else 0,
                1 if user.is_superuser else 0,
                user.created_at,
            ),
        )
        await self.db.commit()
        return user

    async def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        """Retrieves a user by their primary key UUID."""
        cursor = await self.db.execute(
            """
            SELECT id, username, email, hashed_password, full_name, is_active, is_superuser, created_at
            FROM users WHERE id = ?
            """,
            (user_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        data = dict(row)
        data["is_active"] = bool(data["is_active"])
        data["is_superuser"] = bool(data["is_superuser"])
        return UserRecord(**data)

    async def get_user_by_username(self, username: str) -> Optional[UserRecord]:
        """Retrieves a user by their unique username."""
        cursor = await self.db.execute(
            """
            SELECT id, username, email, hashed_password, full_name, is_active, is_superuser, created_at
            FROM users WHERE username = ?
            """,
            (username,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        data = dict(row)
        data["is_active"] = bool(data["is_active"])
        data["is_superuser"] = bool(data["is_superuser"])
        return UserRecord(**data)

    async def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        """Retrieves a user by their unique email."""
        cursor = await self.db.execute(
            """
            SELECT id, username, email, hashed_password, full_name, is_active, is_superuser, created_at
            FROM users WHERE email = ?
            """,
            (email,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        data = dict(row)
        data["is_active"] = bool(data["is_active"])
        data["is_superuser"] = bool(data["is_superuser"])
        return UserRecord(**data)

    async def get_user_by_identifier(self, identifier: str) -> Optional[UserRecord]:
        """Retrieves user by either username or email."""
        cursor = await self.db.execute(
            """
            SELECT id, username, email, hashed_password, full_name, is_active, is_superuser, created_at
            FROM users WHERE username = ? OR email = ?
            """,
            (identifier, identifier),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        data = dict(row)
        data["is_active"] = bool(data["is_active"])
        data["is_superuser"] = bool(data["is_superuser"])
        return UserRecord(**data)

    # ---------------------------------------------------------
    # Workspace & Membership Management
    # ---------------------------------------------------------

    async def ensure_workspace(self, workspace_id: str, name: str = "Default Workspace") -> None:
        """Ensures that the target workspace exists."""
        await self.db.execute(
            """
            INSERT OR IGNORE INTO workspaces (id, name, created_at)
            VALUES (?, ?, ?)
            """,
            (workspace_id, name, datetime.now(timezone.utc).isoformat()),
        )
        await self.db.commit()

    async def create_workspace(self, workspace_id: str, name: str, owner_user_id: Optional[str] = None) -> None:
        """Creates a new workspace and assigns the creator as workspace owner."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            INSERT INTO workspaces (id, name, created_at)
            VALUES (?, ?, ?)
            """,
            (workspace_id, name, now),
        )
        if owner_user_id:
            await self.db.execute(
                """
                INSERT INTO workspace_members (workspace_id, user_id, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (workspace_id, owner_user_id, UserRole.OWNER.value, now),
            )
        await self.db.commit()

    async def add_workspace_member(self, workspace_id: str, user_id: str, role: UserRole | str) -> None:
        """Adds or updates a user's role in a workspace."""
        now = datetime.now(timezone.utc).isoformat()
        role_val = role.value if hasattr(role, "value") else str(role)
        await self.db.execute(
            """
            INSERT INTO workspace_members (workspace_id, user_id, role, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(workspace_id, user_id) DO UPDATE SET role = excluded.role
            """,
            (workspace_id, user_id, role_val, now),
        )
        await self.db.commit()

    async def remove_workspace_member(self, workspace_id: str, user_id: str) -> bool:
        """Removes a user's membership from a workspace."""
        cursor = await self.db.execute(
            "DELETE FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
            (workspace_id, user_id),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def get_workspace_member(self, workspace_id: str, user_id: str) -> Optional[WorkspaceMemberRecord]:
        """Retrieves a user's membership and role in a workspace."""
        cursor = await self.db.execute(
            """
            SELECT workspace_id, user_id, role, created_at
            FROM workspace_members
            WHERE workspace_id = ? AND user_id = ?
            """,
            (workspace_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return WorkspaceMemberRecord(**dict(row))

    async def list_user_workspaces(self, user_id: str) -> list[WorkspaceResponse]:
        """Lists all workspaces that the user belongs to, including their role."""
        cursor = await self.db.execute(
            """
            SELECT w.id, w.name, wm.role, w.created_at
            FROM workspaces w
            JOIN workspace_members wm ON w.id = wm.workspace_id
            WHERE wm.user_id = ?
            ORDER BY w.name ASC
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [WorkspaceResponse(**dict(r)) for r in rows]

    # ---------------------------------------------------------
    # Document Ownership & Document-Level Permissions
    # ---------------------------------------------------------

    async def get_document_by_hash(self, workspace_id: str, sha256_hash: str) -> Optional[DocumentRecord]:
        """Finds an existing document with the given content hash in the workspace."""
        cursor = await self.db.execute(
            """
            SELECT id, workspace_id, filename, file_type, file_size_bytes, 
                   sha256_hash, status, owner_id, is_private, page_count, chunk_count, char_count, 
                   error_message, created_at, updated_at
            FROM documents
            WHERE workspace_id = ? AND sha256_hash = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (workspace_id, sha256_hash),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        data = dict(row)
        data["is_private"] = bool(data.get("is_private", 1))
        return DocumentRecord(**data)

    async def get_document_by_id(self, document_id: str) -> Optional[DocumentRecord]:
        """Retrieves a document record by its primary key UUID."""
        cursor = await self.db.execute(
            """
            SELECT id, workspace_id, filename, file_type, file_size_bytes, 
                   sha256_hash, status, owner_id, is_private, page_count, chunk_count, char_count, 
                   error_message, created_at, updated_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        data = dict(row)
        data["is_private"] = bool(data.get("is_private", 1))
        return DocumentRecord(**data)

    async def create_document(self, doc: DocumentRecord) -> DocumentRecord:
        """Inserts a new document record with owner and privacy metadata in SQLite."""
        await self.ensure_workspace(doc.workspace_id)
        await self.db.execute(
            """
            INSERT INTO documents (
                id, workspace_id, filename, file_type, file_size_bytes,
                sha256_hash, status, owner_id, is_private, page_count, chunk_count, char_count,
                error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.id,
                doc.workspace_id,
                doc.filename,
                doc.file_type,
                doc.file_size_bytes,
                doc.sha256_hash,
                doc.status.value,
                doc.owner_id,
                1 if doc.is_private else 0,
                doc.page_count,
                doc.chunk_count,
                doc.char_count,
                doc.error_message,
                doc.created_at,
                doc.updated_at,
            ),
        )
        await self.db.commit()
        return doc

    async def update_document_status(
        self,
        document_id: str,
        status: IngestionStatus,
        chunk_count: int | None = None,
        char_count: int | None = None,
        page_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Updates ingestion status and counts for a document."""
        now = datetime.now(timezone.utc).isoformat()
        updates = ["status = ?", "updated_at = ?"]
        params: list[object] = [status.value, now]

        if chunk_count is not None:
            updates.append("chunk_count = ?")
            params.append(chunk_count)
        if char_count is not None:
            updates.append("char_count = ?")
            params.append(char_count)
        if page_count is not None:
            updates.append("page_count = ?")
            params.append(page_count)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(document_id)
        sql = f"UPDATE documents SET {', '.join(updates)} WHERE id = ?"
        await self.db.execute(sql, params)
        await self.db.commit()

    async def save_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Bulk inserts chunks for a document in a single transaction."""
        if not chunks:
            return

        chunk_rows = [
            (
                c.id,
                c.document_id,
                c.workspace_id,
                c.chunk_index,
                c.char_start,
                c.char_end,
                c.token_count,
                c.content,
                c.citation_label,
                json.dumps(c.metadata),
                c.content_hash,
            )
            for c in chunks
        ]

        await self.db.executemany(
            """
            INSERT INTO chunks (
                id, document_id, workspace_id, chunk_index, char_start, char_end,
                token_count, content, citation_label, metadata_json, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )
        await self.db.commit()

    async def get_chunks_by_document(self, document_id: str) -> list[ChunkRecord]:
        """Fetches all chunks for a document ordered by sequence index."""
        cursor = await self.db.execute(
            """
            SELECT id, document_id, workspace_id, chunk_index, char_start, char_end,
                   token_count, content, citation_label, metadata_json, content_hash
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC
            """,
            (document_id,),
        )
        rows = await cursor.fetchall()
        chunks: list[ChunkRecord] = []
        for r in rows:
            data = dict(r)
            meta_json = data.pop("metadata_json", None)
            metadata = json.loads(meta_json) if meta_json else {}
            chunks.append(ChunkRecord(**data, metadata=metadata))
        return chunks

    async def delete_document(self, document_id: str) -> bool:
        """Deletes a document; cascading foreign keys remove its chunks and permissions."""
        cursor = await self.db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        await self.db.commit()
        return cursor.rowcount > 0

    async def list_documents(
        self,
        workspace_id: str,
        permitted_doc_ids: Optional[list[str]] = None,
    ) -> list[DocumentRecord]:
        """Lists documents in a workspace, optionally filtered by authorized document IDs."""
        if permitted_doc_ids is not None:
            if not permitted_doc_ids:
                return []
            placeholders = ",".join("?" for _ in permitted_doc_ids)
            sql = f"""
                SELECT id, workspace_id, filename, file_type, file_size_bytes, 
                       sha256_hash, status, owner_id, is_private, page_count, chunk_count, char_count, 
                       error_message, created_at, updated_at
                FROM documents
                WHERE workspace_id = ? AND id IN ({placeholders})
                ORDER BY created_at DESC
            """
            params = [workspace_id] + list(permitted_doc_ids)
        else:
            sql = """
                SELECT id, workspace_id, filename, file_type, file_size_bytes, 
                       sha256_hash, status, owner_id, is_private, page_count, chunk_count, char_count, 
                       error_message, created_at, updated_at
                FROM documents
                WHERE workspace_id = ?
                ORDER BY created_at DESC
            """
            params = [workspace_id]

        cursor = await self.db.execute(sql, params)
        rows = await cursor.fetchall()
        docs: list[DocumentRecord] = []
        for r in rows:
            data = dict(r)
            data["is_private"] = bool(data.get("is_private", 1))
            docs.append(DocumentRecord(**data))
        return docs

    # ---------------------------------------------------------
    # Granular Document Permission Grants & Revocations
    # ---------------------------------------------------------

    async def grant_document_permission(
        self,
        document_id: str,
        user_id: str,
        permission_level: PermissionLevel,
        granted_by: str,
    ) -> DocumentPermissionRecord:
        """Explicitly grants or upgrades a user's access level to a document."""
        perm_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            INSERT INTO document_permissions (id, document_id, user_id, permission_level, granted_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id, user_id) DO UPDATE SET
                permission_level = excluded.permission_level,
                granted_by = excluded.granted_by,
                created_at = excluded.created_at
            """,
            (perm_id, document_id, user_id, permission_level.value, granted_by, now),
        )
        await self.db.commit()
        return DocumentPermissionRecord(
            id=perm_id,
            document_id=document_id,
            user_id=user_id,
            permission_level=permission_level,
            granted_by=granted_by,
            created_at=now,
        )

    async def revoke_document_permission(self, document_id: str, user_id: str) -> bool:
        """Revokes a user's explicit permission on a document immediately."""
        cursor = await self.db.execute(
            "DELETE FROM document_permissions WHERE document_id = ? AND user_id = ?",
            (document_id, user_id),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def get_document_permissions(self, document_id: str) -> list[PermissionResponse]:
        """Lists all explicit permissions granted on a document."""
        cursor = await self.db.execute(
            """
            SELECT dp.document_id, dp.user_id, u.username, dp.permission_level, dp.granted_by, dp.created_at
            FROM document_permissions dp
            JOIN users u ON dp.user_id = u.id
            WHERE dp.document_id = ?
            ORDER BY dp.created_at ASC
            """,
            (document_id,),
        )
        rows = await cursor.fetchall()
        return [PermissionResponse(**dict(r)) for r in rows]

    # ---------------------------------------------------------
    # Server-Side Authorization & Pre-Retrieval ACL Resolution
    # ---------------------------------------------------------

    async def check_user_document_access(
        self,
        user_id: str,
        document_id: str,
        action: str = "read",
    ) -> bool:
        """Determines whether a user has permission to perform an action on a document."""
        doc = await self.get_document_by_id(document_id)
        if not doc:
            return False

        # Verify workspace membership
        member = await self.get_workspace_member(doc.workspace_id, user_id)
        if not member:
            return False

        # Workspace owners and admins have full access across all workspace documents
        if member.role in (UserRole.OWNER, UserRole.ADMIN):
            return True

        # Document owner has full access to their own document
        if doc.owner_id == user_id:
            return True

        # Check explicit permissions on this document
        cursor = await self.db.execute(
            """
            SELECT permission_level
            FROM document_permissions
            WHERE document_id = ? AND user_id = ?
            """,
            (document_id, user_id),
        )
        row = await cursor.fetchone()
        if row:
            level = row["permission_level"]
            if level == PermissionLevel.ADMIN.value:
                return True
            if level == PermissionLevel.WRITE.value and action in ("read", "write"):
                return True
            if level == PermissionLevel.READ.value and action == "read":
                return True

        # If document is workspace-shared (not private), members with read role can view
        if not doc.is_private and action == "read":
            return True

        return False

    async def get_user_permitted_document_ids(
        self,
        user_id: str,
        workspace_id: str,
    ) -> list[str]:
        """Resolves the exact set of document IDs in workspace_id accessible to user_id.
        Enforces least-privilege pre-retrieval security before chunks reach BM25, ChromaDB, Reranker, or LLM.
        """
        # Step 1: Verify user is a member of the workspace
        member = await self.get_workspace_member(workspace_id, user_id)
        if not member:
            return []

        # Step 2: Workspace owners and admins can access all documents in the workspace
        if member.role in (UserRole.OWNER, UserRole.ADMIN):
            cursor = await self.db.execute(
                "SELECT id FROM documents WHERE workspace_id = ?",
                (workspace_id,),
            )
            rows = await cursor.fetchall()
            return [r["id"] for r in rows]

        # Step 3: Regular members/viewers can only access:
        #  - Documents they own
        #  - Workspace-shared documents (is_private = 0)
        #  - Documents explicitly granted to them via document_permissions
        cursor = await self.db.execute(
            """
            SELECT DISTINCT d.id
            FROM documents d
            LEFT JOIN document_permissions dp ON d.id = dp.document_id AND dp.user_id = ?
            WHERE d.workspace_id = ?
              AND (
                  d.owner_id = ?
                  OR d.is_private = 0
                  OR dp.permission_level IN ('read', 'write', 'admin')
              )
            """,
            (user_id, workspace_id, user_id),
        )
        rows = await cursor.fetchall()
        return [r["id"] for r in rows]

    # ---------------------------------------------------------
    # Notion Workspace Connection & Synced Page Management
    # ---------------------------------------------------------

    async def get_notion_connection(self, workspace_id: str) -> Optional[dict]:
        """Retrieves active Notion connection configuration for a workspace."""
        cursor = await self.db.execute(
            """
            SELECT id, workspace_id, user_id, bot_id, workspace_name, workspace_icon,
                   status, encrypted_token, last_synced_at, created_at, updated_at
            FROM notion_connections
            WHERE workspace_id = ?
            """,
            (workspace_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def save_notion_connection(
        self,
        connection_id: str,
        workspace_id: str,
        user_id: str,
        bot_id: Optional[str],
        workspace_name: Optional[str],
        workspace_icon: Optional[str],
        status: str,
        encrypted_token: Optional[str] = None,
        last_synced_at: Optional[str] = None,
    ) -> None:
        """Stores or updates Notion integration connection record for a workspace."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            INSERT INTO notion_connections (
                id, workspace_id, user_id, bot_id, workspace_name, workspace_icon,
                status, encrypted_token, last_synced_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                bot_id = excluded.bot_id,
                workspace_name = excluded.workspace_name,
                workspace_icon = excluded.workspace_icon,
                status = excluded.status,
                encrypted_token = COALESCE(excluded.encrypted_token, notion_connections.encrypted_token),
                last_synced_at = COALESCE(excluded.last_synced_at, notion_connections.last_synced_at),
                updated_at = excluded.updated_at
            """,
            (
                connection_id, workspace_id, user_id, bot_id, workspace_name, workspace_icon,
                status, encrypted_token, last_synced_at, now, now,
            ),
        )
        await self.db.commit()

    async def delete_notion_connection(self, workspace_id: str) -> bool:
        """Removes Notion workspace connection record."""
        cursor = await self.db.execute("DELETE FROM notion_connections WHERE workspace_id = ?", (workspace_id,))
        await self.db.commit()
        return cursor.rowcount > 0

    async def get_notion_page(self, page_id: str) -> Optional[dict]:
        """Retrieves synced Notion page record by Notion page ID."""
        cursor = await self.db.execute(
            """
            SELECT id, page_id, document_id, workspace_id, user_id, title, url,
                   last_edited_time, content_hash, synced_at, status
            FROM notion_pages
            WHERE page_id = ?
            """,
            (page_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_notion_pages(self, workspace_id: str) -> list[dict]:
        """Lists all synced Notion pages in a workspace."""
        cursor = await self.db.execute(
            """
            SELECT id, page_id, document_id, workspace_id, user_id, title, url,
                   last_edited_time, content_hash, synced_at, status
            FROM notion_pages
            WHERE workspace_id = ?
            ORDER BY synced_at DESC
            """,
            (workspace_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def save_notion_page(
        self,
        id: str,
        page_id: str,
        document_id: str,
        workspace_id: str,
        user_id: str,
        title: str,
        url: str,
        last_edited_time: str,
        content_hash: str,
        synced_at: str,
        status: str = "synced",
    ) -> None:
        """Inserts or updates tracked Notion page metadata."""
        await self.db.execute(
            """
            INSERT INTO notion_pages (
                id, page_id, document_id, workspace_id, user_id, title, url,
                last_edited_time, content_hash, synced_at, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(page_id) DO UPDATE SET
                document_id = excluded.document_id,
                title = excluded.title,
                url = excluded.url,
                last_edited_time = excluded.last_edited_time,
                content_hash = excluded.content_hash,
                synced_at = excluded.synced_at,
                status = excluded.status
            """,
            (
                id, page_id, document_id, workspace_id, user_id, title, url,
                last_edited_time, content_hash, synced_at, status,
            ),
        )
        await self.db.commit()

    async def delete_notion_page(self, page_id: str) -> bool:
        """Deletes a tracked Notion page record."""
        cursor = await self.db.execute("DELETE FROM notion_pages WHERE page_id = ?", (page_id,))
        await self.db.commit()
        return cursor.rowcount > 0
