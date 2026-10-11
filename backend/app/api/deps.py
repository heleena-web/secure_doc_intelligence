from typing import AsyncGenerator, Optional
import aiosqlite
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.errors import (
    AuthenticationError,
    InvalidTokenError,
    TokenExpiredError,
    UserNotFoundError,
    WorkspaceAccessDeniedError,
)
from app.core.security import decode_access_token
from app.models.auth import UserRecord, WorkspaceMemberRecord
from app.storage.database import get_db_connection
from app.storage.repository import DocumentRepository

# Standard HTTP Bearer scheme (auto_error=False allows handling custom errors and optional auth)
security_scheme = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Dependency yielding an active SQLite database connection with WAL mode."""
    async with get_db_connection() as db:
        yield db


async def get_repository(db: aiosqlite.Connection = Depends(get_db)) -> DocumentRepository:
    """Dependency yielding an initialized DocumentRepository."""
    return DocumentRepository(db)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    repo: DocumentRepository = Depends(get_repository),
) -> Optional[UserRecord]:
    """Resolves authenticated user if Bearer token is provided; returns None if omitted."""
    if not credentials:
        return None

    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except (TokenExpiredError, InvalidTokenError):
        raise
    except Exception as e:
        raise InvalidTokenError(f"Could not validate credentials: {str(e)}")

    user_id = payload.get("sub")
    if not user_id:
        raise InvalidTokenError("Token payload missing subject identifier.")

    user = await repo.get_user_by_id(user_id)
    if not user or not user.is_active:
        raise AuthenticationError("User account does not exist or has been disabled.")

    return user


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    repo: DocumentRepository = Depends(get_repository),
) -> UserRecord:
    """Enforces authentication: validates Bearer token and returns authenticated UserRecord."""
    if not credentials:
        raise AuthenticationError("Authentication required. Please provide a Bearer token.")

    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except (TokenExpiredError, InvalidTokenError):
        raise
    except Exception as e:
        raise InvalidTokenError(f"Could not validate credentials: {str(e)}")

    user_id = payload.get("sub")
    if not user_id:
        raise InvalidTokenError("Token payload missing subject identifier.")

    user = await repo.get_user_by_id(user_id)
    if not user or not user.is_active:
        raise AuthenticationError("User account does not exist or has been disabled.")

    return user


async def require_workspace_membership(
    workspace_id: str,
    user: UserRecord,
    repo: DocumentRepository,
) -> WorkspaceMemberRecord:
    """Validates that the authenticated user is an active member of the requested workspace."""
    member = await repo.get_workspace_member(workspace_id, user.id)
    if not member:
        raise WorkspaceAccessDeniedError(workspace_id)
    return member
