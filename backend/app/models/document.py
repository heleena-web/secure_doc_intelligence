from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class IngestionStatus(str, Enum):
    PENDING = "PENDING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
    DUPLICATE = "DUPLICATE"


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    NOTION = "notion"


class ParsedBlock(BaseModel):
    """Represents a coherent structural block extracted from a raw document."""

    text: str
    page_number: Optional[int] = None        # 1-indexed for PDF
    heading_path: Optional[str] = None       # For DOCX, e.g., "1. Introduction > 1.1 Scope"
    paragraph_number: Optional[int] = None   # For DOCX
    table_number: Optional[int] = None       # For DOCX
    line_start: Optional[int] = None         # For TXT
    line_end: Optional[int] = None           # For TXT
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Normalized document representation emitted by format-specific parsers."""

    filename: str
    file_type: str
    char_count: int
    blocks: list[ParsedBlock]
    page_count: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    """A single tokenized chunk ready for storage and downstream retrieval."""

    id: str
    document_id: str
    workspace_id: str
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    content: str
    citation_label: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str


class DocumentRecord(BaseModel):
    """Authoritative document record stored in SQLite."""

    id: str
    workspace_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    sha256_hash: str
    status: IngestionStatus
    owner_id: Optional[str] = None
    is_private: bool = True
    page_count: Optional[int] = None
    chunk_count: int = 0
    char_count: int = 0
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class IngestionSummary(BaseModel):
    """User-facing ingestion result returned by the API."""

    document_id: str
    workspace_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    sha256_hash: str
    status: IngestionStatus
    owner_id: Optional[str] = None
    is_private: bool = True
    char_count: int
    chunk_count: int
    page_count: Optional[int] = None
    is_duplicate: bool = False
    sample_citations: list[str] = Field(default_factory=list)
    created_at: str
