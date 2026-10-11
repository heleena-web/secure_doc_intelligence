import logging
from abc import ABC, abstractmethod
from typing import Optional

from app.core.config import settings
from app.models.search import SearchResultItem

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract base class for candidate passage rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[SearchResultItem],
        top_k: int = 5,
    ) -> list[SearchResultItem]:
        """Reranks candidate search result items and returns re-ordered top_k results."""
        pass


class FlashRankReranker(BaseReranker):
    """Lightweight, CPU-optimized cross-encoder reranker powered by FlashRank (ONNX Runtime)."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self._ranker = None
        self._initialized = False

    def _get_ranker(self):
        """Lazy initialization of FlashRank ranker."""
        if not self._initialized:
            try:
                from flashrank import Ranker
                self._ranker = Ranker(model_name=self.model_name)
                self._initialized = True
            except Exception as e:
                logger.warning("Failed to initialize FlashRank with model '%s': %s", self.model_name, e)
                self._ranker = None
                self._initialized = True
        return self._ranker

    def rerank(
        self,
        query: str,
        candidates: list[SearchResultItem],
        top_k: int = 5,
    ) -> list[SearchResultItem]:
        if not candidates:
            return []

        ranker = self._get_ranker()
        if ranker is None:
            # Graceful fallback: return original candidates if model is unavailable
            return candidates[:top_k]

        try:
            from flashrank import RerankRequest

            passages = [{"id": c.chunk_id, "text": c.content} for c in candidates]
            req = RerankRequest(query=query, passages=passages)
            ranked_raw = ranker.rerank(req)

            # Map results by chunk_id
            score_by_id = {str(r["id"]): float(r["score"]) for r in ranked_raw}
            candidate_map = {c.chunk_id: c for c in candidates}

            reranked_items: list[SearchResultItem] = []
            for raw in ranked_raw:
                cid = str(raw["id"])
                if cid in candidate_map:
                    base = candidate_map[cid]
                    r_score = float(raw["score"])
                    # Create updated item with rerank score
                    item_copy = base.model_copy()
                    item_copy.rerank_score = round(r_score, 6)
                    item_copy.score = round(r_score, 6)
                    reranked_items.append(item_copy)

            return reranked_items[:top_k]

        except Exception as e:
            logger.warning("Error during FlashRank reranking: %s. Falling back to hybrid score.", e)
            return candidates[:top_k]


class PassthroughReranker(BaseReranker):
    """Null reranker for benchmarking baseline retrieval without reranking."""

    def rerank(
        self,
        query: str,
        candidates: list[SearchResultItem],
        top_k: int = 5,
    ) -> list[SearchResultItem]:
        return candidates[:top_k]
