import pytest
from app.models.search import SearchResultItem
from app.services.retrieval.reranker import FlashRankReranker, PassthroughReranker


def test_flashrank_reranking_accuracy():
    reranker = FlashRankReranker()

    # Create candidate passages with varying relevance to query
    candidates = [
        SearchResultItem(
            chunk_id="c-low",
            document_id="doc-1",
            workspace_id="default",
            filename="general.txt",
            content="Employees should keep their office desks tidy and dispose of recyclables.",
            citation_label="[Doc: general.txt]",
            score=0.85,
        ),
        SearchResultItem(
            chunk_id="c-high",
            document_id="doc-2",
            workspace_id="default",
            filename="encryption.pdf",
            content="AES-256 encryption keys must be rotated every ninety days via hardware security module.",
            citation_label="[Doc: encryption.pdf, Page: 8]",
            page_number=8,
            score=0.72,
        ),
    ]

    # Query specifically about cryptographic key rotation
    reranked = reranker.rerank(
        query="What is the cryptographic key rotation interval for AES-256?",
        candidates=candidates,
        top_k=2,
    )

    assert len(reranked) == 2
    # c-high must be promoted to rank 1 by cross-encoder reranker
    assert reranked[0].chunk_id == "c-high"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score > reranked[1].rerank_score


def test_flashrank_graceful_fallback_on_error():
    # Initialize with non-existent model to trigger fallback
    faulty_reranker = FlashRankReranker(model_name="non_existent_fake_model_xyz")

    candidates = [
        SearchResultItem(
            chunk_id="c-1",
            document_id="doc-1",
            workspace_id="default",
            filename="doc.txt",
            content="Some document content.",
            citation_label="[Doc: doc.txt]",
            score=0.9,
        ),
    ]

    # Must not raise an exception; must return original candidates
    fallback_results = faulty_reranker.rerank("any query", candidates, top_k=1)
    assert len(fallback_results) == 1
    assert fallback_results[0].chunk_id == "c-1"


def test_passthrough_reranker():
    passthrough = PassthroughReranker()
    candidates = [
        SearchResultItem(
            chunk_id="c-1",
            document_id="doc-1",
            workspace_id="default",
            filename="a.txt",
            content="Text A",
            citation_label="[Doc: a.txt]",
            score=0.9,
        ),
        SearchResultItem(
            chunk_id="c-2",
            document_id="doc-2",
            workspace_id="default",
            filename="b.txt",
            content="Text B",
            citation_label="[Doc: b.txt]",
            score=0.8,
        ),
    ]

    results = passthrough.rerank("query", candidates, top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "c-1"
