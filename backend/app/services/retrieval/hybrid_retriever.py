from typing import Optional

from app.core.config import settings
from app.models.search import SearchResultItem
from app.services.retrieval.bm25_indexer import BM25Retriever
from app.services.retrieval.vector_store import ChromaVectorStore


class HybridRetriever:
    """Combines BM25 keyword matching and dense vector semantic retrieval using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_store: ChromaVectorStore,
        rrf_k: int = 60,
    ):
        self.bm25_retriever = bm25_retriever
        self.vector_store = vector_store
        self.rrf_k = rrf_k or settings.DEFAULT_RRF_K

    def search(
        self,
        query: str,
        workspace_id: str = "default",
        top_k: int = 5,
        bm25_candidates: int | None = None,
        semantic_candidates: int | None = None,
        permitted_doc_ids: Optional[list[str]] = None,
    ) -> list[SearchResultItem]:
        """Executes hybrid retrieval and fuses candidate lists via Reciprocal Rank Fusion."""
        n_bm25 = bm25_candidates or settings.BM25_CANDIDATES
        n_semantic = semantic_candidates or settings.SEMANTIC_CANDIDATES

        bm25_results = self.bm25_retriever.search(
            query=query,
            workspace_id=workspace_id,
            top_k=n_bm25,
            permitted_doc_ids=permitted_doc_ids,
        )

        semantic_results = self.vector_store.search(
            query=query,
            workspace_id=workspace_id,
            top_k=n_semantic,
            permitted_doc_ids=permitted_doc_ids,
        )

        # Map by chunk_id to accumulate RRF scores and preserve metadata
        candidates_by_id: dict[str, SearchResultItem] = {}
        rrf_scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}
        semantic_ranks: dict[str, int] = {}

        # 1. Process BM25 rankings
        for item in bm25_results:
            cid = item.chunk_id
            rank = item.bm25_rank or 1
            bm25_ranks[cid] = rank
            rrf_score = 1.0 / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score
            candidates_by_id[cid] = item

        # 2. Process Semantic rankings
        for item in semantic_results:
            cid = item.chunk_id
            rank = item.semantic_rank or 1
            semantic_ranks[cid] = rank
            rrf_score = 1.0 / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score
            if cid not in candidates_by_id:
                candidates_by_id[cid] = item

        if not candidates_by_id:
            return []

        # 3. Sort candidates by combined RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        final_results: list[SearchResultItem] = []
        for cid in sorted_cids[:top_k]:
            base_item = candidates_by_id[cid]
            total_rrf = round(rrf_scores[cid], 6)

            final_results.append(
                SearchResultItem(
                    chunk_id=base_item.chunk_id,
                    document_id=base_item.document_id,
                    workspace_id=base_item.workspace_id,
                    filename=base_item.filename,
                    content=base_item.content,
                    citation_label=base_item.citation_label,
                    page_number=base_item.page_number,
                    heading_path=base_item.heading_path,
                    score=total_rrf,
                    bm25_rank=bm25_ranks.get(cid),
                    semantic_rank=semantic_ranks.get(cid),
                    rrf_score=total_rrf,
                    metadata=base_item.metadata,
                )
            )

        return final_results
