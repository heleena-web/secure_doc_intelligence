import pytest
from app.models.document import ChunkRecord
from app.models.search import SearchMode, SearchQuery
from app.services.retrieval.bm25_indexer import BM25Retriever
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.reranker import FlashRankReranker
from app.services.retrieval.search_service import SearchService
from app.services.retrieval.vector_store import ChromaVectorStore


@pytest.fixture
def golden_dataset(tmp_path):
    """Sets up a repeatable golden collection of chunks with known relevance labels for benchmarking."""
    bm25 = BM25Retriever()
    vector_store = ChromaVectorStore(persist_dir=tmp_path / "chroma_eval")
    reranker = FlashRankReranker()
    service = SearchService(bm25_retriever=bm25, vector_store=vector_store, reranker=reranker)

    chunks = [
        ChunkRecord(
            id="chunk-auth-802",
            document_id="doc-security",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=120,
            token_count=25,
            content="Policy POL-802 mandates biometric multi-factor authentication and zero-trust credential verification on all corporate devices.",
            citation_label="[Doc: Security_Policy.pdf, Page: 7]",
            metadata={"pages": [7]},
            content_hash="ha" * 32,
        ),
        ChunkRecord(
            id="chunk-sla-uptime",
            document_id="doc-sla",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=100,
            token_count=20,
            content="Service Level Agreements guarantee 99.95% API uptime availability during core operating hours.",
            citation_label="[Doc: SLA_Agreement.docx, Section: 3.1 Availability, Table: 1]",
            metadata={"headings": ["3.1 Availability"], "tables": [1]},
            content_hash="hb" * 32,
        ),
        ChunkRecord(
            id="chunk-retention-7y",
            document_id="doc-compliance",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=110,
            token_count=22,
            content="Tax and audit records must be retained in immutable storage for a duration of seven calendar years.",
            citation_label="[Doc: Retention_Rules.pdf, Page: 15]",
            metadata={"pages": [15]},
            content_hash="hc" * 32,
        ),
        ChunkRecord(
            id="chunk-incident-triage",
            document_id="doc-incident",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=115,
            token_count=24,
            content="Tier-3 cybersecurity incident response protocol requires notifying the CISO and legal counsel within thirty minutes of confirmation.",
            citation_label="[Doc: Incident_Plan.docx, Section: Escalation, Para: 2]",
            metadata={"headings": ["Escalation"], "paragraphs": [2]},
            content_hash="hd" * 32,
        ),
    ]

    service.index_document_chunks(chunks, "Corporate_Collection")
    return service


def compute_metrics(results, ground_truth_chunk_id, k=5):
    """Calculates Recall@K and Reciprocal Rank (RR@K) for a single query."""
    top_k_results = results[:k]
    matched_rank = None
    for rank, item in enumerate(top_k_results, start=1):
        if item.chunk_id == ground_truth_chunk_id:
            matched_rank = rank
            break

    recall = 1.0 if matched_rank is not None else 0.0
    reciprocal_rank = (1.0 / matched_rank) if matched_rank is not None else 0.0
    return recall, reciprocal_rank


def test_four_way_retrieval_comparison(golden_dataset: SearchService):
    """Benchmarks BM25, Semantic, Hybrid, and Hybrid+Rerank across golden queries, reporting Recall@5 and MRR@5."""
    test_cases = [
        {
            "query": "What is policy POL-802 regarding biometric authentication?",
            "target": "chunk-auth-802",
        },
        {
            "query": "What is the guaranteed service level agreement API uptime percentage?",
            "target": "chunk-sla-uptime",
        },
        {
            "query": "How many years must financial audit records be retained?",
            "target": "chunk-retention-7y",
        },
        {
            "query": "Who must be alerted within thirty minutes of a major tier-3 breach?",
            "target": "chunk-incident-triage",
        },
    ]

    modes = [
        SearchMode.BM25,
        SearchMode.SEMANTIC,
        SearchMode.HYBRID,
        SearchMode.HYBRID_RERANK,
    ]

    metrics_by_mode = {}

    for mode in modes:
        recalls = []
        reciprocal_ranks = []

        for case in test_cases:
            query_obj = SearchQuery(
                query=case["query"],
                mode=mode,
                top_k=5,
                workspace_id="default",
            )
            response = golden_dataset.search(query_obj)
            recall, rr = compute_metrics(response.results, case["target"], k=5)
            recalls.append(recall)
            reciprocal_ranks.append(rr)

        avg_recall = sum(recalls) / len(recalls)
        avg_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        metrics_by_mode[mode.value] = {
            "Recall@5": round(avg_recall, 4),
            "MRR@5": round(avg_mrr, 4),
        }

    # All modes must achieve high recall on this curated evaluation collection
    assert metrics_by_mode["bm25"]["Recall@5"] >= 0.75
    assert metrics_by_mode["semantic"]["Recall@5"] >= 0.75
    assert metrics_by_mode["hybrid"]["Recall@5"] == 1.0
    assert metrics_by_mode["hybrid_rerank"]["Recall@5"] == 1.0
    # Hybrid and Rerank must achieve top MRR
    assert metrics_by_mode["hybrid_rerank"]["MRR@5"] >= 0.90


def test_stale_chunks_and_empty_index_evaluation(golden_dataset: SearchService):
    """Verifies that deleted documents are never returned and empty indexes return cleanly."""
    # 1. Search existing document
    q = SearchQuery(query="biometric authentication POL-802", mode=SearchMode.HYBRID_RERANK)
    res_before = golden_dataset.search(q)
    assert len(res_before.results) > 0
    assert any(r.chunk_id == "chunk-auth-802" for r in res_before.results)

    # 2. Delete document
    golden_dataset.remove_document("doc-security")

    # 3. Search again: must have zero occurrences of deleted chunk
    res_after = golden_dataset.search(q)
    assert not any(r.chunk_id == "chunk-auth-802" for r in res_after.results)

    # 4. Clear all documents (empty index)
    golden_dataset.clear()
    empty_res = golden_dataset.search(SearchQuery(query="any search term", mode=SearchMode.HYBRID_RERANK))
    assert empty_res.total_candidates == 0
    assert empty_res.results == []


def test_citation_accuracy_in_search_results(golden_dataset: SearchService):
    """Verifies that search results preserve accurate PDF page numbers and DOCX section references."""
    q_pdf = SearchQuery(query="POL-802 biometric authentication", mode=SearchMode.HYBRID)
    res_pdf = golden_dataset.search(q_pdf)
    assert len(res_pdf.results) > 0
    assert res_pdf.results[0].citation_label == "[Doc: Security_Policy.pdf, Page: 7]"
    assert res_pdf.results[0].page_number == 7

    q_docx = SearchQuery(query="99.95% API uptime SLA", mode=SearchMode.HYBRID)
    res_docx = golden_dataset.search(q_docx)
    assert len(res_docx.results) > 0
    assert "Section: 3.1 Availability" in res_docx.results[0].citation_label
    assert "Table: 1" in res_docx.results[0].citation_label


def test_reproducible_benchmark_harness(tmp_path):
    """Executes the complete RetrievalBenchmarkHarness across all 4 modes and checks measured metric thresholds."""
    from app.services.evaluation.benchmark import (
        RetrievalBenchmarkHarness,
        get_reproducible_evaluation_dataset,
    )

    bm25 = BM25Retriever()
    vector_store = ChromaVectorStore(persist_dir=tmp_path / "chroma_bench")
    reranker = FlashRankReranker()
    service = SearchService(bm25_retriever=bm25, vector_store=vector_store, reranker=reranker)

    chunks, queries = get_reproducible_evaluation_dataset()
    service.index_document_chunks(chunks, "Evaluation_Collection")

    harness = RetrievalBenchmarkHarness(service)
    results = harness.run_full_benchmark(queries, k=5)

    assert "bm25" in results
    assert "semantic" in results
    assert "hybrid" in results
    assert "hybrid_rerank" in results

    assert results["bm25"].recall_at_5 >= 0.80
    assert results["semantic"].recall_at_5 >= 0.80
    assert results["hybrid"].recall_at_5 == 1.0
    assert results["hybrid_rerank"].recall_at_5 == 1.0
    assert results["hybrid_rerank"].mrr_at_5 >= 0.90
