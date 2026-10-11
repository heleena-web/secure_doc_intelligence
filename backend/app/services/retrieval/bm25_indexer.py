import re
from typing import Optional
from rank_bm25 import BM25Plus

from app.models.document import ChunkRecord
from app.models.search import SearchResultItem



def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenizes text while preserving policy codes, hyphens, and exact identifiers (e.g. POL-402)."""
    tokens: list[str] = []
    # Match alphanumeric sequences with embedded hyphens, underscores, or periods
    for match in re.findall(r"[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*", text.lower()):
        tokens.append(match)
        # Also index components for hybrid matching
        if "-" in match or "_" in match or "." in match:
            for sub in re.split(r"[-_.]", match):
                if sub and sub not in tokens:
                    tokens.append(sub)
    return tokens


class BM25Retriever:
    """In-memory BM25 indexer synchronized with authoritative SQLite document storage."""

    def __init__(self):
        self.chunks_by_id: dict[str, ChunkRecord] = {}
        self.filenames_by_doc_id: dict[str, str] = {}
        self._bm25: Optional[BM25Plus] = None
        self._indexed_chunk_ids: list[str] = []
        self._corpus_tokens: list[set[str]] = []

    def _rebuild_index(self) -> None:
        """Rebuilds the BM25Plus model from current chunks."""
        self._indexed_chunk_ids = list(self.chunks_by_id.keys())
        if not self._indexed_chunk_ids:
            self._bm25 = None
            self._corpus_tokens = []
            return

        corpus = [
            tokenize_for_bm25(self.chunks_by_id[cid].content)
            for cid in self._indexed_chunk_ids
        ]
        self._corpus_tokens = [set(toks) for toks in corpus]
        self._bm25 = BM25Plus(corpus)

    def add_chunks(self, chunks: list[ChunkRecord], filename: str = "") -> None:
        """Adds new chunks to the BM25 index."""
        if not chunks:
            return
        for chunk in chunks:
            self.chunks_by_id[chunk.id] = chunk
            if filename:
                self.filenames_by_doc_id[chunk.document_id] = filename
        self._rebuild_index()

    def remove_document(self, document_id: str) -> None:
        """Removes all chunks associated with a deleted document to eliminate stale results."""
        to_delete = [
            cid for cid, chunk in self.chunks_by_id.items()
            if chunk.document_id == document_id
        ]
        for cid in to_delete:
            del self.chunks_by_id[cid]
        self.filenames_by_doc_id.pop(document_id, None)
        self._rebuild_index()

    def clear(self) -> None:
        """Clears all indexed chunks."""
        self.chunks_by_id.clear()
        self.filenames_by_doc_id.clear()
        self._bm25 = None
        self._indexed_chunk_ids.clear()
        self._corpus_tokens.clear()

    def search(
        self,
        query: str,
        workspace_id: str = "default",
        top_k: int = 5,
        permitted_doc_ids: Optional[list[str]] = None,
    ) -> list[SearchResultItem]:
        """Executes BM25 keyword search against indexed chunks."""
        if not self._bm25 or not self._indexed_chunk_ids:
            return []

        tokenized_query = tokenize_for_bm25(query)
        if not tokenized_query:
            return []

        raw_scores = self._bm25.get_scores(tokenized_query)
        query_set = set(tokenized_query)

        # Filter by workspace and optional permitted document IDs
        permitted_set = set(permitted_doc_ids) if permitted_doc_ids is not None else None
        scored_candidates: list[tuple[float, ChunkRecord]] = []

        for idx, (cid, score) in enumerate(zip(self._indexed_chunk_ids, raw_scores)):
            chunk = self.chunks_by_id.get(cid)
            if not chunk or chunk.workspace_id != workspace_id:
                continue
            if permitted_set is not None and chunk.document_id not in permitted_set:
                continue
            if score > 0.0 and query_set.intersection(self._corpus_tokens[idx]):
                scored_candidates.append((float(score), chunk))


        # Sort descending by BM25 score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:top_k]

        results: list[SearchResultItem] = []
        for rank, (score, chunk) in enumerate(top_candidates, start=1):
            filename = self.filenames_by_doc_id.get(chunk.document_id, "Document")
            pages = chunk.metadata.get("pages", [])
            headings = chunk.metadata.get("headings", [])

            results.append(
                SearchResultItem(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    workspace_id=chunk.workspace_id,
                    filename=filename,
                    content=chunk.content,
                    citation_label=chunk.citation_label,
                    page_number=pages[0] if pages else None,
                    heading_path=headings[0] if headings else None,
                    score=round(score, 4),
                    bm25_rank=rank,
                    metadata=chunk.metadata,
                )
            )

        return results
