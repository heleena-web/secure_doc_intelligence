import pytest
from app.models.document import ChunkRecord
from app.services.retrieval.bm25_indexer import BM25Retriever
from app.services.retrieval.vector_store import ChromaVectorStore
from app.services.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_rrf_scoring_and_deduplication(tmp_path):
    bm25 = BM25Retriever()
    vector_store = ChromaVectorStore(persist_dir=tmp_path / "chroma_hybrid")
    hybrid = HybridRetriever(bm25_retriever=bm25, vector_store=vector_store, rrf_k=60)

    # Chunk A: matches both keyword ("incident") and semantic concept
    chunk_a = ChunkRecord(
        id="c-both",
        document_id="doc-1",
        workspace_id="default",
        chunk_index=0,
        char_start=0,
        char_end=80,
        token_count=15,
        content="Cybersecurity breach reporting timeline requires incident response teams to notify within 24 hours.",
        citation_label="[Doc: Incident_Plan.pdf, Page: 5]",
        metadata={"pages": [5]},
        content_hash="ha" * 32,
    )

    # Chunk B: partial keyword match
    chunk_b = ChunkRecord(
        id="c-other",
        document_id="doc-2",
        workspace_id="default",
        chunk_index=0,
        char_start=0,
        char_end=60,
        token_count=12,
        content="General office security protocol and visitor badges.",
        citation_label="[Doc: Protocol.txt, Lines: 1-3]",
        metadata={"line_ranges": [(1, 3)]},
        content_hash="hb" * 32,
    )

    bm25.add_chunks([chunk_a, chunk_b], filename="Incident_Plan.pdf")
    vector_store.add_chunks([chunk_a, chunk_b], filename="Incident_Plan.pdf")

    results = hybrid.search(
        query="cybersecurity breach reporting timeline",
        workspace_id="default",
        top_k=5,
    )

    assert len(results) >= 1
    # Check deduplication: each chunk ID appears at most once
    chunk_ids = [r.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))

    # Chunk A (present in both rankers) should achieve top RRF score
    top_item = results[0]
    assert top_item.chunk_id == "c-both"
    assert top_item.rrf_score is not None
    assert top_item.bm25_rank is not None
    assert top_item.semantic_rank is not None
    # Combined RRF score must be strictly greater than single ranker reciprocal score
    assert top_item.rrf_score > (1.0 / (60 + 1))


def test_hybrid_empty_retrievers(tmp_path):
    bm25 = BM25Retriever()
    vector_store = ChromaVectorStore(persist_dir=tmp_path / "chroma_empty")
    hybrid = HybridRetriever(bm25_retriever=bm25, vector_store=vector_store)

    assert hybrid.search("any query", workspace_id="default") == []
