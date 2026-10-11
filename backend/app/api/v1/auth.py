import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status

from app.api.deps import get_current_user, get_repository, require_workspace_membership
from app.core.config import settings
from app.core.errors import (
    InvalidCredentialsError,
    PermissionDeniedError,
    UserAlreadyExistsError,
    UserNotFoundError,
    WorkspaceAccessDeniedError,
)
from app.core.security import create_access_token, hash_password, verify_password
from app.models.auth import (
    TokenResponse,
    UserLoginRequest,
    UserRecord,
    UserRegisterRequest,
    UserResponse,
    UserRole,
    WorkspaceCreateRequest,
    WorkspaceMemberAddRequest,
    WorkspaceResponse,
)
from app.storage.repository import DocumentRepository

router = APIRouter(tags=["Authentication & Workspaces"])


@router.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    req: UserRegisterRequest,
    repo: DocumentRepository = Depends(get_repository),
) -> TokenResponse:
    """Registers a new user, hashes password with Argon2id, and provisions default workspace access."""
    # Check for existing username or email
    existing_user = await repo.get_user_by_username(req.username)
    if existing_user:
        raise UserAlreadyExistsError(req.username)

    existing_email = await repo.get_user_by_email(req.email)
    if existing_email:
        raise UserAlreadyExistsError(req.email)

    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    hashed = hash_password(req.password)

    user_record = UserRecord(
        id=user_id,
        username=req.username,
        email=req.email,
        hashed_password=hashed,
        full_name=req.full_name,
        is_active=True,
        is_superuser=False,
        created_at=now,
    )
    await repo.create_user(user_record)

    # Ensure default workspace exists and enroll user as member
    await repo.ensure_workspace("default", "Default Workspace")
    await repo.add_workspace_member("default", user_id, UserRole.MEMBER)

    # Generate JWT token
    token = create_access_token(subject=user_id, claims={"username": req.username})

    user_resp = UserResponse(
        id=user_id,
        username=req.username,
        email=req.email,
        full_name=req.full_name,
        is_active=True,
        created_at=now,
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        user=user_resp,
    )


@router.post("/auth/login", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def login_user(
    req: UserLoginRequest,
    repo: DocumentRepository = Depends(get_repository),
) -> TokenResponse:
    """Authenticates user via Argon2id verification and returns a signed JWT token."""
    user = await repo.get_user_by_identifier(req.username)
    if not user or not user.is_active:
        raise InvalidCredentialsError()

    if not verify_password(req.password, user.hashed_password):
        raise InvalidCredentialsError()

    token = create_access_token(subject=user.id, claims={"username": user.username})

    user_resp = UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        created_at=user.created_at,
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        user=user_resp,
    )


@router.get("/auth/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def get_current_user_profile(
    current_user: UserRecord = Depends(get_current_user),
) -> UserResponse:
    """Returns profile for currently authenticated user."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )


# -------------------------------------------------------------
# Workspaces
# -------------------------------------------------------------

@router.post("/workspaces", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    req: WorkspaceCreateRequest,
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> WorkspaceResponse:
    """Creates a new workspace with the authenticated user as owner."""
    now = datetime.now(timezone.utc).isoformat()
    await repo.create_workspace(req.id, req.name, owner_user_id=current_user.id)
    return WorkspaceResponse(
        id=req.id,
        name=req.name,
        role=UserRole.OWNER,
        created_at=now,
    )


@router.get("/workspaces", response_model=list[WorkspaceResponse], status_code=status.HTTP_200_OK)
async def list_workspaces(
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> list[WorkspaceResponse]:
    """Lists all workspaces the authenticated user is enrolled in."""
    return await repo.list_user_workspaces(current_user.id)


@router.post("/workspaces/{workspace_id}/members", status_code=status.HTTP_200_OK)
async def add_workspace_member(
    workspace_id: str,
    req: WorkspaceMemberAddRequest,
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> dict:
    """Adds a user to a workspace. Enforces that only workspace owners or admins can add members."""
    # Validate caller is owner or admin in this workspace
    caller_member = await require_workspace_membership(workspace_id, current_user, repo)
    if caller_member.role not in (UserRole.OWNER, UserRole.ADMIN):
        raise PermissionDeniedError("Only workspace owners or administrators can manage membership.")

    # Target user must exist
    target_user = await repo.get_user_by_username(req.username)
    if not target_user:
        raise UserNotFoundError(req.username)

    # Regular admins cannot elevate users to OWNER or grant higher roles than themselves
    if caller_member.role == UserRole.ADMIN and req.role == UserRole.OWNER:
        raise PermissionDeniedError("Administrators cannot transfer or grant workspace ownership.")

    await repo.add_workspace_member(workspace_id, target_user.id, req.role)
    return {
        "status": "success",
        "workspace_id": workspace_id,
        "user_id": target_user.id,
        "username": target_user.username,
        "role": req.role.value,
    }


@router.delete("/workspaces/{workspace_id}/members/{user_id}", status_code=status.HTTP_200_OK)
async def remove_workspace_member(
    workspace_id: str,
    user_id: str,
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> dict:
    """Removes a user from a workspace. Only workspace owners or admins can remove members."""
    caller_member = await require_workspace_membership(workspace_id, current_user, repo)
    if caller_member.role not in (UserRole.OWNER, UserRole.ADMIN):
        raise PermissionDeniedError("Only workspace owners or administrators can remove members.")

    # Users cannot demote/remove the workspace owner unless they are the owner themselves
    target_member = await repo.get_workspace_member(workspace_id, user_id)
    if target_member and target_member.role == UserRole.OWNER and caller_member.role != UserRole.OWNER:
        raise PermissionDeniedError("Administrators cannot remove the workspace owner.")

    removed = await repo.remove_workspace_member(workspace_id, user_id)
    return {"status": "success", "removed": removed}
