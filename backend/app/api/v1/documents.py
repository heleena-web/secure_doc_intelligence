from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, status, UploadFile

from app.api.deps import (
    get_current_user,
    get_repository,
    require_workspace_membership,
)
from app.core.config import settings
from app.core.errors import (
    DocumentAccessDeniedError,
    DocumentNotFoundError,
    PermissionDeniedError,
    UserNotFoundError,
)
from app.models.auth import (
    GrantPermissionRequest,
    PermissionResponse,
    UserRecord,
    UserRole,
)
from app.models.document import (
    ChunkRecord,
    DocumentRecord,
    IngestionSummary,
)
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.retrieval.search_service import search_service
from app.storage.repository import DocumentRepository

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/ingest", response_model=IngestionSummary, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: UploadFile = File(...),
    workspace_id: str = Form("default"),
    is_private: bool = Form(True),
    max_tokens: Optional[int] = Form(None),
    overlap_tokens: Optional[int] = Form(None),
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> IngestionSummary:
    """Ingests, parses, chunks, and stores a document (PDF, DOCX, or TXT).
    Enforces authentication, workspace tenancy, and document ownership.
    """
    # Enforce workspace tenancy
    await require_workspace_membership(workspace_id, current_user, repo)

    filename = file.filename or "unknown"
    temp_path = settings.UPLOAD_DIR / f"temp_{file.filename}"

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        pipeline = IngestionPipeline(repo)
        summary = await pipeline.ingest_document(
            content=content,
            filename=filename,
            workspace_id=workspace_id,
            owner_id=current_user.id,
            is_private=is_private,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            temp_file_path=temp_path,
        )
        return summary

    finally:
        await file.close()
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@router.get("/{document_id}", response_model=DocumentRecord)
async def get_document(
    document_id: str,
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> DocumentRecord:
    """Retrieves document record if the authenticated user has read access."""
    doc = await repo.get_document_by_id(document_id)
    if not doc:
        raise DocumentNotFoundError(document_id)

    has_access = await repo.check_user_document_access(current_user.id, document_id, action="read")
    if not has_access:
        raise DocumentAccessDeniedError(document_id, "read")

    return doc


@router.get("/{document_id}/chunks", response_model=list[ChunkRecord])
async def get_document_chunks(
    document_id: str,
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> list[ChunkRecord]:
    """Retrieves stored chunks and citations for an authorized document."""
    doc = await repo.get_document_by_id(document_id)
    if not doc:
        raise DocumentNotFoundError(document_id)

    has_access = await repo.check_user_document_access(current_user.id, document_id, action="read")
    if not has_access:
        raise DocumentAccessDeniedError(document_id, "read")

    chunks = await repo.get_chunks_by_document(document_id)
    return chunks


@router.get("", response_model=list[DocumentRecord])
async def list_documents(
    workspace_id: str = "default",
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> list[DocumentRecord]:
    """Lists all documents in a workspace that the user has authorization to view."""
    await require_workspace_membership(workspace_id, current_user, repo)
    permitted_doc_ids = await repo.get_user_permitted_document_ids(current_user.id, workspace_id)
    return await repo.list_documents(workspace_id, permitted_doc_ids=permitted_doc_ids)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> None:
    """Deletes a document if caller is owner or workspace admin; purges chunks and search indices."""
    doc = await repo.get_document_by_id(document_id)
    if not doc:
        raise DocumentNotFoundError(document_id)

    has_access = await repo.check_user_document_access(current_user.id, document_id, action="write")
    if not has_access:
        raise DocumentAccessDeniedError(document_id, "delete")

    deleted = await repo.delete_document(document_id)
    if not deleted:
        raise DocumentNotFoundError(document_id)

    search_service.remove_document(document_id)


# -------------------------------------------------------------
# Explicit Document Permissions
# -------------------------------------------------------------

@router.post("/{document_id}/permissions", response_model=PermissionResponse, status_code=status.HTTP_200_OK)
async def grant_document_permission(
    document_id: str,
    req: GrantPermissionRequest,
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> PermissionResponse:
    """Grants explicit document-level permissions. Caller must be owner or workspace admin."""
    doc = await repo.get_document_by_id(document_id)
    if not doc:
        raise DocumentNotFoundError(document_id)

    # Caller must have admin access to the document or workspace
    caller_member = await require_workspace_membership(doc.workspace_id, current_user, repo)
    is_owner_or_admin = (
        doc.owner_id == current_user.id
        or caller_member.role in (UserRole.OWNER, UserRole.ADMIN)
    )
    if not is_owner_or_admin:
        raise PermissionDeniedError("Only the document owner or workspace administrators can grant permissions.")

    # Target user must exist and belong to the workspace
    target_user = await repo.get_user_by_username(req.username)
    if not target_user:
        raise UserNotFoundError(req.username)

    target_member = await repo.get_workspace_member(doc.workspace_id, target_user.id)
    if not target_member:
        raise PermissionDeniedError(f"User '{req.username}' is not a member of workspace '{doc.workspace_id}'.")

    perm = await repo.grant_document_permission(
        document_id=document_id,
        user_id=target_user.id,
        permission_level=req.permission_level,
        granted_by=current_user.id,
    )

    return PermissionResponse(
        document_id=document_id,
        user_id=target_user.id,
        username=target_user.username,
        permission_level=perm.permission_level,
        granted_by=current_user.username,
        created_at=perm.created_at,
    )


@router.delete("/{document_id}/permissions/{user_id}", status_code=status.HTTP_200_OK)
async def revoke_document_permission(
    document_id: str,
    user_id: str,
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> dict:
    """Revokes explicit permission from a user on a document immediately."""
    doc = await repo.get_document_by_id(document_id)
    if not doc:
        raise DocumentNotFoundError(document_id)

    caller_member = await require_workspace_membership(doc.workspace_id, current_user, repo)
    is_owner_or_admin = (
        doc.owner_id == current_user.id
        or caller_member.role in (UserRole.OWNER, UserRole.ADMIN)
    )
    if not is_owner_or_admin:
        raise PermissionDeniedError("Only the document owner or workspace administrators can revoke permissions.")

    revoked = await repo.revoke_document_permission(document_id, user_id)
    return {"status": "success", "revoked": revoked}


@router.get("/{document_id}/permissions", response_model=list[PermissionResponse], status_code=status.HTTP_200_OK)
async def list_document_permissions(
    document_id: str,
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> list[PermissionResponse]:
    """Lists explicit permissions on a document. Caller must have read access."""
    doc = await repo.get_document_by_id(document_id)
    if not doc:
        raise DocumentNotFoundError(document_id)

    has_access = await repo.check_user_document_access(current_user.id, document_id, action="read")
    if not has_access:
        raise DocumentAccessDeniedError(document_id, "read")

    return await repo.get_document_permissions(document_id)
