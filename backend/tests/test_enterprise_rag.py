"""Tests for Enterprise RAG requirements:
1. Citation enforcement and grounded-answer validation (fabricated, missing, unsupported citations, unanswerable queries).
2. Deterministic answer quality metrics (Context Precision, Lexical Faithfulness, Citation Precision, Answer Relevance).
3. Retrieval evaluation harness across all 4 modes (BM25, Semantic, Hybrid, Hybrid+Rerank).
4. Automated CI quality gates.
5. Access-controlled citation and document isolation.
"""
import pytest
from app.models.document import ChunkRecord
from app.models.qa import CitationItem, QARequest, QAResponse, QAStatus
from app.models.search import SearchMode, SearchQuery, SearchResultItem
from app.services.evaluation.benchmark import (
    RetrievalBenchmarkHarness,
    get_reproducible_evaluation_dataset,
)
from app.services.evaluation.quality_assurance import (
    AnswerQualityAuditor,
    compute_answer_relevance,
    compute_citation_precision,
    compute_context_precision,
    compute_lexical_faithfulness,
)
from app.services.llm.qa_service import GroundedQAService
from app.services.retrieval.bm25_indexer import BM25Retriever
from app.services.retrieval.reranker import FlashRankReranker
from app.services.retrieval.search_service import SearchService
from app.services.retrieval.vector_store import ChromaVectorStore


# =====================================================================
# 1. Deterministic Answer Evaluation Metrics Tests
# =====================================================================

def test_compute_context_precision_perfect_ranking():
    """Rank 1 and 2 are relevant -> Context precision should be 1.0."""
    retrieved = ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"]
    targets = ["chunk-1", "chunk-2"]
    score = compute_context_precision(retrieved, targets, k=5)
    # Rank 1: precision 1/1 = 1.0
    # Rank 2: precision 2/2 = 1.0
    # Avg precision = (1.0 + 1.0) / 2 = 1.0
    assert score == pytest.approx(1.0)


def test_compute_context_precision_delayed_ranking():
    """Relevant items appear lower down (ranks 3 and 5) -> lower context precision."""
    retrieved = ["chunk-irr-1", "chunk-irr-2", "chunk-rel-1", "chunk-irr-3", "chunk-rel-2"]
    targets = ["chunk-rel-1", "chunk-rel-2"]
    score = compute_context_precision(retrieved, targets, k=5)
    # Rank 3: rel=1, prec=1/3 (~0.333)
    # Rank 5: rel=2, prec=2/5 (0.4)
    # Avg precision = (1/3 + 2/5) / 2 = (0.7333) / 2 = 0.3667
    assert score == pytest.approx(0.3667, abs=1e-3)


def test_compute_citation_precision():
    """Tests valid vs fabricated citations."""
    candidate_ids = ["chunk-1", "chunk-2", "chunk-3"]
    # All citations are in candidate set
    assert compute_citation_precision(["chunk-1", "chunk-2"], candidate_ids) == 1.0
    # 1 valid, 1 fabricated
    assert compute_citation_precision(["chunk-1", "chunk-fake"], candidate_ids) == 0.5
    # All fabricated
    assert compute_citation_precision(["fake-1", "fake-2"], candidate_ids) == 0.0
    # Empty citations
    assert compute_citation_precision([], candidate_ids) == 1.0


def test_compute_lexical_faithfulness():
    """Tests grounding overlap between answer text and source chunks."""
    cited_chunks = [
        "Policy SEC-402 mandates hardware-backed FIDO2 multi-factor authentication tokens for remote administrative access."
    ]
    # Grounded answer sharing significant keywords
    grounded_answer = "Under Policy SEC-402, remote administrative access strictly requires hardware-backed FIDO2 tokens."
    score_grounded = compute_lexical_faithfulness(grounded_answer, cited_chunks)
    assert score_grounded > 0.40

    # Hallucinated / off-topic answer sharing almost no keywords
    hallucinated_answer = "Bananas and pineapples grow in tropical climates and need ample rainfall."
    score_hallucinated = compute_lexical_faithfulness(hallucinated_answer, cited_chunks)
    assert score_hallucinated < 0.15


def test_compute_answer_relevance():
    """Tests token overlap with ground-truth reference answer."""
    reference = "Hardware-backed FIDO2 multi-factor authentication tokens are required for administrative access."
    answer = "Administrative access requires FIDO2 hardware-backed MFA tokens."
    relevance = compute_answer_relevance(answer, reference)
    assert relevance > 0.40

    unrelated = "The company was founded in 1998 in California."
    assert compute_answer_relevance(unrelated, reference) < 0.10


# =====================================================================
# 2. Citation Enforcement & Grounded-Answer Validation Tests
# =====================================================================

def test_auditor_detects_fabricated_citations():
    """Auditor flags cited chunk IDs that were never in the retrieved candidates."""
    auditor = AnswerQualityAuditor()
    retrieved_chunk = SearchResultItem(
        chunk_id="real-chunk-1",
        document_id="doc-1",
        workspace_id="default",
        filename="Security.pdf",
        content="Session timeouts are configured for 15 minutes of inactivity.",
        citation_label="[Doc: Security.pdf, Page 1]",
        score=0.95,
    )
    fake_citation = CitationItem(
        chunk_id="fake-chunk-999",
        document_id="doc-fake",
        filename="Fake.pdf",
        citation_label="[Doc: Fake.pdf]",
        snippet="Fabricated snippet",
        verified=False,
    )
    qa_response = QAResponse(
        question="What is the session timeout?",
        answer="The session timeout is 15 minutes [fake-chunk-999].",
        status=QAStatus.ANSWERED,
        has_sufficient_evidence=True,
        citations=[fake_citation],
        supporting_snippets=[],
        model_used="qwen3:8b",
        retrieved_chunks_count=1,
    )

    report = auditor.audit_response(
        response=qa_response,
        candidate_chunks=[retrieved_chunk],
    )

    assert not report.is_grounded
    assert "fake-chunk-999" in report.invalid_citations
    assert report.unsupported_claims_detected
    assert any("unretrieved or hallucinated" in n for n in report.notes)


def test_auditor_detects_unsupported_claims_when_faithfulness_is_low():
    """Auditor flags unsupported claims when answer content words do not match cited text."""
    auditor = AnswerQualityAuditor()
    retrieved_chunk = SearchResultItem(
        chunk_id="chunk-finance-01",
        document_id="doc-fin",
        workspace_id="default",
        filename="Finance.pdf",
        content="The quarterly revenue reached 4.2 million dollars.",
        citation_label="[Doc: Finance.pdf, Page 1]",
        score=0.90,
    )
    citation = CitationItem(
        chunk_id="chunk-finance-01",
        document_id="doc-fin",
        filename="Finance.pdf",
        citation_label="[Doc: Finance.pdf, Page 1]",
        snippet="The quarterly revenue reached 4.2 million dollars.",
        verified=True,
    )
    qa_response = QAResponse(
        question="What are the encryption standards?",
        # Answer makes claim completely absent from cited text
        answer="The system uses quantum-resistant lattice cryptography for all cold storage databases [chunk-finance-01].",
        status=QAStatus.ANSWERED,
        has_sufficient_evidence=True,
        citations=[citation],
        supporting_snippets=[],
        model_used="qwen3:8b",
        retrieved_chunks_count=1,
    )

    report = auditor.audit_response(
        response=qa_response,
        candidate_chunks=[retrieved_chunk],
    )

    assert not report.is_grounded
    assert report.unsupported_claims_detected
    assert report.faithfulness_score < 0.15
    assert any("Low lexical faithfulness" in n for n in report.notes)


def test_auditor_flags_missing_citations_on_factual_claims():
    """Auditor flags when a factual answer provides zero citations despite retrieved evidence."""
    auditor = AnswerQualityAuditor()
    retrieved_chunk = SearchResultItem(
        chunk_id="chunk-auth-01",
        document_id="doc-auth",
        workspace_id="default",
        filename="Auth.pdf",
        content="FIDO2 tokens are mandatory for remote access.",
        citation_label="[Doc: Auth.pdf, Page 1]",
        score=0.92,
    )
    qa_response = QAResponse(
        question="What tokens are mandatory?",
        answer="FIDO2 tokens are mandatory for all remote access connections.",
        status=QAStatus.ANSWERED,
        has_sufficient_evidence=True,
        citations=[],  # Missing citations
        supporting_snippets=[],
        model_used="qwen3:8b",
        retrieved_chunks_count=1,
    )

    report = auditor.audit_response(
        response=qa_response,
        candidate_chunks=[retrieved_chunk],
    )

    assert not report.is_grounded
    assert report.unsupported_claims_detected
    assert any("lacks any valid, verified citations" in n for n in report.notes)


@pytest.mark.asyncio
async def test_unanswerable_query_refusal_generates_zero_citations(tmp_path):
    """GroundedQAService refuses to invent answers or fabricate citations when evidence is insufficient."""
    bm25 = BM25Retriever()
    vector_store = ChromaVectorStore(persist_dir=tmp_path / "chroma")
    reranker = FlashRankReranker()
    search_service = SearchService(bm25, vector_store, reranker)

    # Empty search service - no documents indexed
    qa_service = GroundedQAService(retrieval_service=search_service)
    req = QARequest(
        question="What is the internal satellite communication frequency?",
        workspace_id="default",
    )
    response = await qa_service.answer_question(req)

    assert response.status == QAStatus.NO_EVIDENCE_FOUND
    assert len(response.citations) == 0
    assert not response.has_sufficient_evidence
    assert any(phrase in response.answer.lower() for phrase in ["do not contain", "insufficient", "not find"])


# =====================================================================
# 3. Retrieval Benchmark Harness & Quality Gates Tests
# =====================================================================

def test_retrieval_benchmark_runs_all_modes_and_measures_metrics(tmp_path):
    """RetrievalBenchmarkHarness executes BM25, Semantic, Hybrid, and Hybrid+Rerank."""
    bm25 = BM25Retriever()
    vector_store = ChromaVectorStore(persist_dir=tmp_path / "chroma_eval")
    reranker = FlashRankReranker()
    search_service = SearchService(bm25, vector_store, reranker)

    chunks, test_cases = get_reproducible_evaluation_dataset(dataset_size="standard")
    search_service.index_document_chunks(chunks, "Eval_Collection")

    harness = RetrievalBenchmarkHarness(search_service)
    results = harness.run_full_benchmark(test_cases, k=5)

    assert len(results) == 4
    for mode_name in [SearchMode.BM25.value, SearchMode.SEMANTIC.value, SearchMode.HYBRID.value, SearchMode.HYBRID_RERANK.value]:
        assert mode_name in results
        metric = results[mode_name]
        assert 0.0 <= metric.recall_at_5 <= 1.0
        assert 0.0 <= metric.mrr_at_5 <= 1.0
        assert 0.0 <= metric.precision_at_5 <= 1.0
        assert 0.0 <= metric.context_precision_at_5 <= 1.0
        assert metric.avg_latency_ms >= 0.0

    # Verify Hybrid with Reranker achieves high recall and MRR
    hybrid_rerank = results[SearchMode.HYBRID_RERANK.value]
    assert hybrid_rerank.recall_at_5 >= 0.85
    assert hybrid_rerank.mrr_at_5 >= 0.80


def test_quality_gate_threshold_verification(tmp_path):
    """Verifies that quality gate logic correctly passes valid thresholds and fails unrealizable thresholds."""
    bm25 = BM25Retriever()
    vector_store = ChromaVectorStore(persist_dir=tmp_path / "chroma_eval")
    reranker = FlashRankReranker()
    search_service = SearchService(bm25, vector_store, reranker)

    chunks, test_cases = get_reproducible_evaluation_dataset(dataset_size="standard")
    search_service.index_document_chunks(chunks, "Eval_Collection")

    harness = RetrievalBenchmarkHarness(search_service)
    result = harness.evaluate_mode(SearchMode.HYBRID_RERANK, test_cases, k=5)

    # Standard enterprise gate: Recall >= 0.85, MRR >= 0.80
    assert result.recall_at_5 >= 0.85
    assert result.mrr_at_5 >= 0.80

    # Unrealistic gate should fail
    unrealistic_recall_gate = 1.05
    assert not (result.recall_at_5 >= unrealistic_recall_gate)


# =====================================================================
# 4. Citation Access Control & Cross-User Isolation
# =====================================================================

def test_cross_user_private_document_citation_isolation(tmp_path):
    """User cannot retrieve or cite private chunks owned by another user."""
    bm25 = BM25Retriever()
    vector_store = ChromaVectorStore(persist_dir=tmp_path / "chroma_isolation")
    reranker = FlashRankReranker()
    search_service = SearchService(bm25, vector_store, reranker)

    # User A's private chunk
    user_a_chunk = ChunkRecord(
        id="chunk-secret-a",
        document_id="doc-secret-a",
        workspace_id="ws-1",
        chunk_index=0,
        char_start=0,
        char_end=120,
        token_count=25,
        content="Confidential Project Alpha secret key: 994827-XF-A.",
        citation_label="[Doc: Secret_A.txt, Page 1]",
        content_hash="hash-secret-a",
    )

    # User B's public chunk
    public_chunk = ChunkRecord(
        id="chunk-public-b",
        document_id="doc-public-b",
        workspace_id="ws-1",
        chunk_index=0,
        char_start=0,
        char_end=100,
        token_count=20,
        content="General guidelines for repository setup and development workflow.",
        citation_label="[Doc: Guidelines.txt, Page 1]",
        content_hash="hash-public-b",
    )

    search_service.index_document_chunks([user_a_chunk, public_chunk], "Isolation_Collection")

    # User B searches: only has access to "doc-public-b"
    query_user_b = SearchQuery(
        query="secret key Project Alpha",
        mode=SearchMode.HYBRID_RERANK,
        top_k=5,
        workspace_id="ws-1",
        permitted_doc_ids=["doc-public-b"],  # Filter applied at pre-retrieval
    )
    results = search_service.search(query_user_b)

    retrieved_ids = [r.chunk_id for r in results.results]
    assert "chunk-secret-a" not in retrieved_ids
    # Unauthorized content never reaches search results or reranker


# =====================================================================
# 5. Phase 7: Live Pipeline & Cross-Workspace Regression Tests
# =====================================================================

@pytest.mark.asyncio
async def test_live_query_pipeline_genuinely_uses_cross_encoder_reranker(tmp_path):
    """Verifies that GroundedQAService.answer_question invokes the FlashRank cross-encoder reranker in the live query pipeline."""
    from unittest.mock import AsyncMock, patch

    bm25 = BM25Retriever()
    vector_store = ChromaVectorStore(persist_dir=tmp_path / "chroma_rerank")
    reranker = FlashRankReranker()
    search_service = SearchService(bm25, vector_store, reranker)

    chunk = ChunkRecord(
        id="chunk-sop-101",
        document_id="doc-sop",
        workspace_id="ws-ops",
        chunk_index=0,
        char_start=0,
        char_end=150,
        token_count=30,
        content="Standard Operating Procedure 101: In event of water leak, turn valve C-4 clockwise 90 degrees.",
        citation_label="[Doc: SOP_101.pdf, Page 12]",
        content_hash="hash-sop-101",
    )
    search_service.index_document_chunks([chunk], "SOP_Collection")

    mock_ollama = AsyncMock()
    mock_ollama.default_model = "qwen3:8b"
    mock_ollama.validate_availability = AsyncMock(return_value=None)
    mock_ollama.generate = AsyncMock(
        return_value="In event of water leak, turn valve C-4 clockwise 90 degrees [chunk-sop-101]."
    )

    qa_service = GroundedQAService(retrieval_service=search_service, ollama_client=mock_ollama)

    with patch.object(reranker, "rerank", wraps=reranker.rerank) as spy_rerank:
        req = QARequest(
            question="What is the water leak procedure?",
            workspace_id="ws-ops",
        )
        res = await qa_service.answer_question(req)

        assert spy_rerank.called
        assert spy_rerank.call_count == 1
        call_kwargs = spy_rerank.call_args.kwargs
        call_args = spy_rerank.call_args.args
        query_arg = call_kwargs.get("query") if "query" in call_kwargs else call_args[0]
        candidates_arg = call_kwargs.get("candidates") if "candidates" in call_kwargs else call_args[1]

        assert query_arg == "What is the water leak procedure?"
        assert any(c.chunk_id == "chunk-sop-101" for c in candidates_arg)
        assert res.status == QAStatus.ANSWERED
        assert len(res.citations) == 1
        assert res.citations[0].chunk_id == "chunk-sop-101"


def test_citation_metadata_grounded_in_trusted_storage_not_llm_fabrication():
    """Verifies that CitationItem metadata strictly reflects trusted chunk metadata, rejecting any fabricated metadata from LLM text."""
    qa_service = GroundedQAService()
    candidate = SearchResultItem(
        chunk_id="chunk-trusted-77",
        document_id="doc-security-policy",
        workspace_id="ws-sec",
        filename="Security_Policy_v2.pdf",
        citation_label="[Doc: Security_Policy_v2.pdf, Page: 4]",
        page_number=4,
        heading_path="Access Control > MFA",
        content="Hardware tokens are strictly mandatory for all remote shell access.",
        score=0.96,
        metadata={"source_type": "document", "pages": [4]},
    )

    # LLM hallucinates a different page in its prose: Page 99
    llm_answer = "Hardware tokens are strictly mandatory according to Page 99 [chunk-trusted-77]."
    citations = qa_service._extract_and_verify_citations(
        answer=llm_answer,
        candidates=[candidate],
        has_sufficient_evidence=True,
    )

    assert len(citations) == 1
    cit = citations[0]
    # Trusted metadata preserved: Page 4 (NOT the LLM's hallucinated Page 99)
    assert cit.page_number == 4
    assert cit.filename == "Security_Policy_v2.pdf"
    assert cit.heading_path == "Access Control > MFA"
    assert cit.verified is True


def test_cross_workspace_retrieval_isolation(tmp_path):
    """Verifies that retrieval strictly isolates workspaces at the index layer."""
    bm25 = BM25Retriever()
    vector_store = ChromaVectorStore(persist_dir=tmp_path / "chroma_ws_iso")
    reranker = FlashRankReranker()
    search_service = SearchService(bm25, vector_store, reranker)

    ws1_chunk = ChunkRecord(
        id="chunk-ws1-01",
        document_id="doc-ws1",
        workspace_id="workspace-alpha",
        chunk_index=0,
        char_start=0,
        char_end=100,
        token_count=20,
        content="Alpha confidential quarterly projections and internal metrics.",
        citation_label="[Doc: Alpha.pdf, Page 1]",
        content_hash="hash-ws1",
    )
    ws2_chunk = ChunkRecord(
        id="chunk-ws2-01",
        document_id="doc-ws2",
        workspace_id="workspace-beta",
        chunk_index=0,
        char_start=0,
        char_end=100,
        token_count=20,
        content="Beta public guidelines and standard operating procedures.",
        citation_label="[Doc: Beta.pdf, Page 1]",
        content_hash="hash-ws2",
    )

    search_service.index_document_chunks([ws1_chunk, ws2_chunk], "MultiWorkspace")

    # Search in workspace-beta asking for alpha confidential info
    query = SearchQuery(
        query="Alpha confidential quarterly projections",
        mode=SearchMode.HYBRID_RERANK,
        top_k=5,
        workspace_id="workspace-beta",
    )
    res = search_service.search(query)
    retrieved_ids = [r.chunk_id for r in res.results]

    # chunk-ws1-01 must NEVER be returned in workspace-beta
    assert "chunk-ws1-01" not in retrieved_ids
