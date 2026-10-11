from typing import Optional

from app.core.config import settings
from app.models.document import ChunkRecord
from app.models.search import SearchMode, SearchQuery, SearchResponse, SearchResultItem
from app.services.retrieval.bm25_indexer import BM25Retriever
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.reranker import BaseReranker, FlashRankReranker
from app.services.retrieval.vector_store import ChromaVectorStore


class SearchService:
    """Coordinates BM25, semantic, hybrid, and reranking retrieval pipelines."""

    def __init__(
        self,
        bm25_retriever: Optional[BM25Retriever] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        reranker: Optional[BaseReranker] = None,
    ):
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.vector_store = vector_store or ChromaVectorStore()
        self.reranker = reranker or FlashRankReranker()
        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            vector_store=self.vector_store,
            rrf_k=settings.DEFAULT_RRF_K,
        )

    def index_document_chunks(self, chunks: list[ChunkRecord], filename: str) -> None:
        """Indexes newly ingested chunks into both BM25 and ChromaDB."""
        self.bm25_retriever.add_chunks(chunks, filename)
        self.vector_store.add_chunks(chunks, filename)

    def remove_document(self, document_id: str) -> None:
        """Removes a document's chunks from both BM25 and ChromaDB to prevent stale results."""
        self.bm25_retriever.remove_document(document_id)
        self.vector_store.delete_document(document_id)

    def clear(self) -> None:
        """Clears all in-memory BM25 indices and ChromaDB collections."""
        self.bm25_retriever.clear()
        self.vector_store.clear()

    def search(self, query: SearchQuery) -> SearchResponse:
        """Executes search in requested mode with workspace and permission filtering."""
        results: list[SearchResultItem] = []

        if query.mode == SearchMode.BM25:
            results = self.bm25_retriever.search(
                query=query.query,
                workspace_id=query.workspace_id,
                top_k=query.top_k,
                permitted_doc_ids=query.permitted_doc_ids,
            )

        elif query.mode == SearchMode.SEMANTIC:
            results = self.vector_store.search(
                query=query.query,
                workspace_id=query.workspace_id,
                top_k=query.top_k,
                permitted_doc_ids=query.permitted_doc_ids,
            )

        elif query.mode == SearchMode.HYBRID:
            results = self.hybrid_retriever.search(
                query=query.query,
                workspace_id=query.workspace_id,
                top_k=query.top_k,
                permitted_doc_ids=query.permitted_doc_ids,
            )

        elif query.mode == SearchMode.HYBRID_RERANK:
            # First fetch candidate pool using hybrid search (up to max candidates)
            candidate_pool = self.hybrid_retriever.search(
                query=query.query,
                workspace_id=query.workspace_id,
                top_k=max(query.top_k * 3, settings.BM25_CANDIDATES),
                permitted_doc_ids=query.permitted_doc_ids,
            )
            # Second re-rank candidates with cross-encoder
            results = self.reranker.rerank(
                query=query.query,
                candidates=candidate_pool,
                top_k=query.top_k,
            )

        return SearchResponse(
            query=query.query,
            mode=query.mode,
            workspace_id=query.workspace_id,
            total_candidates=len(results),
            results=results,
        )


# Global singleton instance for runtime operations
search_service = SearchService()
