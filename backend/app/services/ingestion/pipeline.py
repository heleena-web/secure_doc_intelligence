import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import aiosqlite

from app.core.config import settings
from app.core.errors import (
    FileSizeExceededError,
    UnsupportedFileTypeError,
    EmptyDocumentError,
)
from app.models.document import (
    DocumentRecord,
    DocumentType,
    IngestionStatus,
    IngestionSummary,
)
from app.services.ingestion.parser import ParserFactory
from app.services.ingestion.chunker import TokenizerAwareChunker
from app.storage.repository import DocumentRepository
from app.services.retrieval.search_service import search_service, SearchService


class IngestionPipeline:
    """Orchestrates validation, deduplication, parsing, chunking, and persistence."""

    def __init__(self, repo: DocumentRepository, search_svc: Optional[SearchService] = None):
        self.repo = repo
        self.search_svc = search_svc or search_service


    async def ingest_document(
        self,
        content: bytes,
        filename: str,
        workspace_id: str = "default",
        owner_id: Optional[str] = None,
        is_private: bool = True,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        temp_file_path: Path | None = None,
    ) -> IngestionSummary:
        """Executes full document ingestion lifecycle with cleanup guarantee."""
        try:
            # 1. Validation: File extension
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext not in settings.ALLOWED_EXTENSIONS:
                raise UnsupportedFileTypeError(ext, settings.ALLOWED_EXTENSIONS)

            # 2. Validation: File size
            file_size = len(content)
            if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
                raise FileSizeExceededError(file_size, settings.MAX_UPLOAD_SIZE_BYTES)

            # 3. Validation: Non-empty payload
            if file_size == 0 or len(content.strip()) == 0:
                raise EmptyDocumentError(filename)

            # 4. Content Hashing & Duplicate Detection
            sha256_hash = hashlib.sha256(content).hexdigest()
            existing_doc = await self.repo.get_document_by_hash(workspace_id, sha256_hash)

            if existing_doc and existing_doc.status == IngestionStatus.INDEXED:
                # Fetch sample citations for summary
                existing_chunks = await self.repo.get_chunks_by_document(existing_doc.id)
                sample_citations = [c.citation_label for c in existing_chunks[:3]]
                return IngestionSummary(
                    document_id=existing_doc.id,
                    workspace_id=existing_doc.workspace_id,
                    filename=existing_doc.filename,
                    file_type=existing_doc.file_type,
                    file_size_bytes=existing_doc.file_size_bytes,
                    sha256_hash=existing_doc.sha256_hash,
                    status=IngestionStatus.INDEXED,
                    owner_id=existing_doc.owner_id,
                    is_private=existing_doc.is_private,
                    char_count=existing_doc.char_count,
                    chunk_count=existing_doc.chunk_count,
                    page_count=existing_doc.page_count,
                    is_duplicate=True,
                    sample_citations=sample_citations,
                    created_at=existing_doc.created_at,
                )

            # 5. Parsing
            parser = ParserFactory.get_parser(filename)
            parsed_doc = parser.parse(content, filename)

            # 6. Initialize Document Record
            doc_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            doc_record = DocumentRecord(
                id=doc_id,
                workspace_id=workspace_id,
                filename=filename,
                file_type=parsed_doc.file_type,
                file_size_bytes=file_size,
                sha256_hash=sha256_hash,
                status=IngestionStatus.PENDING,
                owner_id=owner_id,
                is_private=is_private,
                page_count=parsed_doc.page_count,
                chunk_count=0,
                char_count=parsed_doc.char_count,
                error_message=None,
                created_at=now,
                updated_at=now,
            )
            await self.repo.create_document(doc_record)

            # 7. Tokenizer-Aware Chunking
            chunker = TokenizerAwareChunker(
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
            chunks = chunker.chunk_document(doc_id, workspace_id, parsed_doc)

            # 8. Persist Chunks and Mark Status as INDEXED
            await self.repo.save_chunks(chunks)
            await self.repo.update_document_status(
                document_id=doc_id,
                status=IngestionStatus.INDEXED,
                chunk_count=len(chunks),
                char_count=parsed_doc.char_count,
                page_count=parsed_doc.page_count,
            )

            # Index chunks in BM25 and ChromaDB for retrieval
            self.search_svc.index_document_chunks(chunks, filename)

            sample_citations = [c.citation_label for c in chunks[:3]]

            return IngestionSummary(
                document_id=doc_id,
                workspace_id=workspace_id,
                filename=filename,
                file_type=parsed_doc.file_type,
                file_size_bytes=file_size,
                sha256_hash=sha256_hash,
                status=IngestionStatus.INDEXED,
                owner_id=owner_id,
                is_private=is_private,
                char_count=parsed_doc.char_count,
                chunk_count=len(chunks),
                page_count=parsed_doc.page_count,
                is_duplicate=False,
                sample_citations=sample_citations,
                created_at=now,
            )

        except Exception as e:
            # If doc_id was assigned and document was created, update status to FAILED
            if "doc_id" in locals() and "doc_record" in locals():
                try:
                    await self.repo.update_document_status(
                        document_id=doc_id,
                        status=IngestionStatus.FAILED,
                        error_message=str(e),
                    )
                except Exception:
                    pass
            raise

        finally:
            # Guaranteed cleanup of temporary file if provided
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except OSError:
                    pass
