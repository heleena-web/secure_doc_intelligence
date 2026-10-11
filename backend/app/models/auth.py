from enum import Enum
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class PermissionLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class UserRecord(BaseModel):
    """Authoritative user record stored in SQLite."""

    id: str
    username: str
    email: str
    hashed_password: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False
    created_at: str


class UserResponse(BaseModel):
    """Public user profile returned in responses."""

    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: str


class UserRegisterRequest(BaseModel):
    """Registration request payload."""

    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = None


class UserLoginRequest(BaseModel):
    """Login request payload."""

    username: str  # Can be username or email
    password: str


class TokenResponse(BaseModel):
    """JWT bearer token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: UserResponse


class WorkspaceMemberRecord(BaseModel):
    """Workspace membership with role."""

    workspace_id: str
    user_id: str
    role: UserRole
    created_at: str


class WorkspaceCreateRequest(BaseModel):
    """Workspace creation payload."""

    id: str = Field(..., min_length=2, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(..., min_length=2, max_length=100)


class WorkspaceResponse(BaseModel):
    """Workspace detail with caller's role."""

    id: str
    name: str
    role: Optional[UserRole] = None
    created_at: str


class WorkspaceMemberAddRequest(BaseModel):
    """Add user to workspace payload."""

    username: str
    role: UserRole = UserRole.MEMBER


class DocumentPermissionRecord(BaseModel):
    """Explicit document-level access permission."""

    id: str
    document_id: str
    user_id: str
    permission_level: PermissionLevel
    granted_by: str
    created_at: str


class GrantPermissionRequest(BaseModel):
    """Grant or update document permission payload."""

    username: str
    permission_level: PermissionLevel = PermissionLevel.READ


class PermissionResponse(BaseModel):
    """Permission detail returned in responses."""

    document_id: str
    user_id: str
    username: str
    permission_level: PermissionLevel
    granted_by: str
    created_at: str
