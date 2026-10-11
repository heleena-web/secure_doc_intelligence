from fastapi import APIRouter, Depends, status

from app.api.deps import (
    get_current_user,
    get_repository,
    require_workspace_membership,
)
from app.models.auth import UserRecord
from app.models.qa import QARequest, QAResponse
from app.services.llm.qa_service import qa_service
from app.storage.repository import DocumentRepository

router = APIRouter(prefix="/qa", tags=["Question Answering"])


@router.post("/answer", response_model=QAResponse, status_code=status.HTTP_200_OK)
async def answer_question(
    req: QARequest,
    current_user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_repository),
) -> QAResponse:
    """Generates grounded answer with server-side pre-retrieval ACL enforcement."""
    # 1. Enforce workspace membership
    await require_workspace_membership(req.workspace_id, current_user, repo)

    # 2. Authoritative pre-retrieval permission resolution from SQLite
    user_permitted = await repo.get_user_permitted_document_ids(current_user.id, req.workspace_id)

    # Intersect with any caller-requested sub-filter
    if req.permitted_doc_ids is not None:
        req.permitted_doc_ids = [d for d in req.permitted_doc_ids if d in user_permitted]
    else:
        req.permitted_doc_ids = user_permitted

    return await qa_service.answer_question(req)
