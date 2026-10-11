from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class QAStatus(str, Enum):
    ANSWERED = "ANSWERED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NO_EVIDENCE_FOUND = "NO_EVIDENCE_FOUND"


class CitationItem(BaseModel):
    """Verified, application-grounded source citation metadata."""

    chunk_id: str
    document_id: str
    filename: str
    citation_label: str
    page_number: Optional[int] = None
    heading_path: Optional[str] = None
    snippet: str
    verified: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_type: Optional[str] = None
    source_url: Optional[str] = None


class QARequest(BaseModel):
    """Question answering request model."""

    question: str
    workspace_id: str = "default"
    top_k: int = 5
    # Pre-retrieval security filter hook for Phase 3
    permitted_doc_ids: Optional[list[str]] = None


class QAResponse(BaseModel):
    """Grounded question answering response."""

    question: str
    answer: str
    status: QAStatus
    has_sufficient_evidence: bool
    citations: list[CitationItem] = Field(default_factory=list)
    supporting_snippets: list[str] = Field(default_factory=list)
    model_used: str
    retrieved_chunks_count: int
