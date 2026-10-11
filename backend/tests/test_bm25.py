import pytest
from app.models.document import ChunkRecord
from app.services.retrieval.bm25_indexer import BM25Retriever, tokenize_for_bm25


def test_tokenize_preserves_identifiers():
    tokens = tokenize_for_bm25("Compliance with POL-402, ISO-27001-A, and Section 3.2 is mandatory.")
    assert "pol-402" in tokens
    assert "pol" in tokens
    assert "402" in tokens
    assert "iso-27001-a" in tokens
    assert "iso" in tokens
    assert "27001" in tokens
    assert "section" in tokens
    assert "3.2" in tokens


def test_bm25_search_ranking_and_exact_policy():
    retriever = BM25Retriever()

    chunk1 = ChunkRecord(
        id="c-pol-402",
        document_id="doc-security",
        workspace_id="default",
        chunk_index=0,
        char_start=0,
        char_end=100,
        token_count=20,
        content="Policy POL-402 mandates multi-factor authentication on all company workstations.",
        citation_label="[Doc: Security_Manual.pdf, Page: 4]",
        metadata={"pages": [4]},
        content_hash="h1" * 32,
    )

    chunk2 = ChunkRecord(
        id="c-unrelated",
        document_id="doc-cafeteria",
        workspace_id="default",
        chunk_index=0,
        char_start=0,
        char_end=80,
        token_count=15,
        content="The cafeteria lunch schedule is posted on Mondays.",
        citation_label="[Doc: Cafeteria.txt, Lines: 1-5]",
        metadata={"line_ranges": [(1, 5)]},
        content_hash="h2" * 32,
    )

    retriever.add_chunks([chunk1, chunk2], filename="Security_Manual.pdf")

    # Query for exact policy code
    results = retriever.search("POL-402 authentication", workspace_id="default", top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "c-pol-402"
    assert results[0].bm25_rank == 1
    assert results[0].score > 0.0
    assert results[0].citation_label == "[Doc: Security_Manual.pdf, Page: 4]"
    assert results[0].page_number == 4


def test_bm25_stale_chunks_eliminated():
    retriever = BM25Retriever()

    chunk = ChunkRecord(
        id="c-stale",
        document_id="doc-old",
        workspace_id="default",
        chunk_index=0,
        char_start=0,
        char_end=50,
        token_count=10,
        content="Confidential corporate merger terms under project Titan.",
        citation_label="[Doc: Merger.docx, Section: Intro]",
        metadata={"headings": ["Intro"]},
        content_hash="h3" * 32,
    )

    retriever.add_chunks([chunk], filename="Merger.docx")
    assert len(retriever.search("project Titan", workspace_id="default")) == 1

    # Delete document
    retriever.remove_document("doc-old")

    # Assert search yields zero results (no stale chunks)
    stale_results = retriever.search("project Titan", workspace_id="default")
    assert len(stale_results) == 0


def test_bm25_empty_index_and_zero_matches():
    retriever = BM25Retriever()
    assert retriever.search("any query", workspace_id="default") == []

    chunk = ChunkRecord(
        id="c-apples",
        document_id="doc-fruit",
        workspace_id="default",
        chunk_index=0,
        char_start=0,
        char_end=30,
        token_count=5,
        content="Apples, oranges, and bananas.",
        citation_label="[Doc: Fruits.txt]",
        metadata={},
        content_hash="h4" * 32,
    )
    retriever.add_chunks([chunk])

    # Unrelated query should yield 0 results
    assert retriever.search("quantum mechanics satellite orbit", workspace_id="default") == []


def test_bm25_workspace_isolation():
    retriever = BM25Retriever()

    chunk_ws1 = ChunkRecord(
        id="c-ws1",
        document_id="doc-1",
        workspace_id="workspace_alpha",
        chunk_index=0,
        char_start=0,
        char_end=40,
        token_count=8,
        content="Confidential alpha payroll data.",
        citation_label="[Doc: Payroll.pdf]",
        metadata={},
        content_hash="h5" * 32,
    )

    retriever.add_chunks([chunk_ws1])

    # Query in workspace_beta should return 0 results
    assert retriever.search("payroll data", workspace_id="workspace_beta") == []
    # Query in workspace_alpha should return match
    assert len(retriever.search("payroll data", workspace_id="workspace_alpha")) == 1
