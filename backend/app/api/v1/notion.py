from typing import Optional
from fastapi import APIRouter, Depends, Query, status

from app.api.deps import (
    get_current_user,
    get_repository,
    require_workspace_membership,
)
from app.models.auth import UserRecord
from app.models.notion import (
    NotionConnectionResponse,
    NotionConnectRequest,
    NotionPageSummary,
    NotionSyncRequest,
    NotionSyncResponse,
)
from app.services.notion.service import notion_service
from app.storage.repository import DocumentRepository

router = APIRouter(prefix="/notion", tags=["Notion Knowledge Integration"])


@router.get("/status", response_model=NotionConnectionResponse, status_code=status.HTTP_200_OK)
async def get_notion_status(
    workspace_id: str = Query("default"),
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> NotionConnectionResponse:
    """Retrieves Notion connection and synchronization status for the requested workspace."""
    await require_workspace_membership(workspace_id, current_user, repo)
    return await notion_service.get_connection_status(workspace_id, repo)


@router.post("/connect", response_model=NotionConnectionResponse, status_code=status.HTTP_200_OK)
async def connect_notion(
    req: NotionConnectRequest,
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> NotionConnectionResponse:
    """Connects Notion workspace using either provided token or server-side configured NOTION_API_KEY.
    Verifies credentials against Notion before persisting connection state.
    """
    await require_workspace_membership(req.workspace_id, current_user, repo)
    return await notion_service.connect(req.workspace_id, current_user.id, req.api_key, repo)


@router.post("/disconnect", status_code=status.HTTP_200_OK)
async def disconnect_notion(
    workspace_id: str = Query("default"),
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> dict[str, str]:
    """Disconnects Notion integration for the specified workspace."""
    await require_workspace_membership(workspace_id, current_user, repo)
    await notion_service.disconnect(workspace_id, repo)
    return {"message": f"Notion disconnected for workspace '{workspace_id}'."}


@router.get("/pages", response_model=list[NotionPageSummary], status_code=status.HTTP_200_OK)
async def list_notion_pages(
    workspace_id: str = Query("default"),
    query: str = Query(""),
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> list[NotionPageSummary]:
    """Discovers accessible Notion pages and shows their current synchronization status."""
    await require_workspace_membership(workspace_id, current_user, repo)
    return await notion_service.list_pages(workspace_id, repo, query=query)


@router.post("/sync", response_model=NotionSyncResponse, status_code=status.HTTP_200_OK)
async def sync_notion_pages(
    req: NotionSyncRequest,
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> NotionSyncResponse:
    """Synchronizes Notion pages into workspace document library and hybrid search indices.
    Detects page updates and purges removed pages automatically.
    """
    await require_workspace_membership(req.workspace_id, current_user, repo)
    return await notion_service.sync(
        workspace_id=req.workspace_id,
        user_id=current_user.id,
        repo=repo,
        page_ids=req.page_ids,
        is_private=req.is_private,
        force=req.force,
    )
