from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class NotionConnectionStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class NotionConnectionResponse(BaseModel):
    """Safe, non-secret-leaking representation of Notion workspace connection."""

    workspace_id: str
    status: NotionConnectionStatus
    is_connected: bool
    workspace_name: Optional[str] = None
    bot_id: Optional[str] = None
    bot_name: Optional[str] = None
    last_synced_at: Optional[str] = None
    page_count: int = 0
    has_configured_token: bool = False


class NotionConnectRequest(BaseModel):
    """User request to connect Notion to a workspace.
    api_key is optional; if omitted, the backend uses the securely configured NOTION_API_KEY environment variable.
    """

    workspace_id: str = "default"
    api_key: Optional[str] = None


class NotionPageSummary(BaseModel):
    """Summary of an accessible Notion page."""

    page_id: str
    title: str
    url: str
    last_edited_time: str
    is_synced: bool = False
    document_id: Optional[str] = None
    synced_at: Optional[str] = None
    status: str = "available"


class NotionSyncRequest(BaseModel):
    """Request to synchronize Notion pages into workspace document library."""

    workspace_id: str = "default"
    page_ids: Optional[list[str]] = None
    is_private: bool = True
    force: bool = False


class NotionSyncResponse(BaseModel):
    """Summary result of a Notion synchronization run."""

    workspace_id: str
    total_pages_found: int
    pages_synced: int
    pages_updated: int
    pages_removed: int
    pages_skipped: int
    synced_at: str
    errors: list[str] = Field(default_factory=list)
