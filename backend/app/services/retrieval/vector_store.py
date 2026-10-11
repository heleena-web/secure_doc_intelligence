import json
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app.core.config import settings
from app.models.document import ChunkRecord
from app.models.search import SearchResultItem


class ChromaVectorStore:
    """ChromaDB vector store providing dense semantic retrieval with local all-MiniLM-L6-v2 embeddings."""

    def __init__(self, persist_dir: Path | str | None = None):
        path = str(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.client = chromadb.PersistentClient(path=path)
        self.embedding_fn = DefaultEmbeddingFunction()
        self.collection_name = "document_chunks"

    @property
    def _collection(self):
        """Dynamically ensures an active collection handle without stale UUID caching."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[ChunkRecord], filename: str = "") -> None:
        """Indexes chunk texts and metadata into ChromaDB using identical chunk UUIDs."""
        if not chunks:
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []

        for c in chunks:
            pages = c.metadata.get("pages", [])
            headings = c.metadata.get("headings", [])
            page_num = pages[0] if pages else -1
            heading = headings[0] if headings else ""

            meta: dict[str, str | int | float | bool] = {
                "doc_id": c.document_id,
                "workspace_id": c.workspace_id,
                "filename": filename,
                "citation_label": c.citation_label,
                "page_number": page_num,
                "heading_path": heading,
                "chunk_index": c.chunk_index,
                "source_type": str(c.metadata.get("source_type", "document")),
                "url": str(c.metadata.get("url", "")),
                "notion_page_id": str(c.metadata.get("notion_page_id", "")),
            }

            ids.append(c.id)
            documents.append(c.content)
            metadatas.append(meta)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_document(self, document_id: str) -> None:
        """Removes all vectors belonging to a document to eliminate stale search results."""
        try:
            self._collection.delete(where={"doc_id": document_id})
        except Exception:
            pass

    def clear(self) -> None:
        """Deletes all records from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

    def search(
        self,
        query: str,
        workspace_id: str = "default",
        top_k: int = 5,
        permitted_doc_ids: Optional[list[str]] = None,
    ) -> list[SearchResultItem]:
        """Performs dense vector cosine similarity search with workspace and permission filtering."""
        count = self._collection.count()
        if count == 0:
            return []

        # Construct pre-retrieval metadata filter
        if permitted_doc_ids is not None:
            if len(permitted_doc_ids) == 0:
                return []
            elif len(permitted_doc_ids) == 1:
                where_filter = {
                    "$and": [
                        {"workspace_id": workspace_id},
                        {"doc_id": permitted_doc_ids[0]},
                    ]
                }
            else:
                where_filter = {
                    "$and": [
                        {"workspace_id": workspace_id},
                        {"doc_id": {"$in": permitted_doc_ids}},
                    ]
                }
        else:
            where_filter = {"workspace_id": workspace_id}

        n_results = min(top_k, count)

        try:
            raw_res = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        results: list[SearchResultItem] = []
        ids = raw_res.get("ids", [[]])[0]
        docs = raw_res.get("documents", [[]])[0]
        metas = raw_res.get("metadatas", [[]])[0]
        dists = raw_res.get("distances", [[]])[0]

        for rank, (cid, doc_text, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
            # Cosine distance: similarity = 1 - distance (clamped to [0, 1])
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            page_num = int(meta.get("page_number", -1))
            heading = str(meta.get("heading_path", ""))

            results.append(
                SearchResultItem(
                    chunk_id=cid,
                    document_id=str(meta.get("doc_id", "")),
                    workspace_id=str(meta.get("workspace_id", workspace_id)),
                    filename=str(meta.get("filename", "Document")),
                    content=doc_text,
                    citation_label=str(meta.get("citation_label", "")),
                    page_number=page_num if page_num > 0 else None,
                    heading_path=heading if heading else None,
                    score=round(similarity, 4),
                    semantic_rank=rank,
                    metadata=dict(meta),
                )
            )

        return results
