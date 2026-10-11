from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    BM25 = "bm25"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class SearchResultItem(BaseModel):
    """Ranked search result item returned to callers."""

    chunk_id: str
    document_id: str
    workspace_id: str
    filename: str
    content: str
    citation_label: str
    page_number: Optional[int] = None
    heading_path: Optional[str] = None
    score: float
    bm25_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchQuery(BaseModel):
    """Search request model."""

    query: str
    mode: SearchMode = SearchMode.HYBRID_RERANK
    top_k: int = 5
    workspace_id: str = "default"
    # Pre-retrieval security hook for Phase 3
    permitted_doc_ids: Optional[list[str]] = None


class SearchResponse(BaseModel):
    """Search response payload."""

    query: str
    mode: SearchMode
    workspace_id: str
    total_candidates: int
    results: list[SearchResultItem]
