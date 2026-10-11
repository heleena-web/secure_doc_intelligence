import pytest
from app.models.qa import CitationItem, QAResponse, QAStatus
from app.models.search import SearchResultItem
from app.services.evaluation.quality_assurance import AnswerQualityAuditor


@pytest.fixture
def sample_candidate_chunks() -> list[SearchResultItem]:
    return [
        SearchResultItem(
            chunk_id="chunk_sec_001",
            document_id="doc_sec",
            workspace_id="default",
            filename="Security_Policy.pdf",
            citation_label="[Doc: Security_Policy.pdf, Page: 4]",
            page_number=4,
            heading_path=None,
            content="Hardware-backed MFA tokens are mandatory for all remote administrative access.",
            score=0.95,
        ),
        SearchResultItem(
            chunk_id="chunk_hr_002",
            document_id="doc_hr",
            workspace_id="default",
            filename="Employee_Handbook.docx",
            citation_label="[Doc: Employee_Handbook.docx, Section: 2.1 Benefits]",
            page_number=None,
            heading_path="2.1 Benefits",
            content="Parental leave provides 16 weeks of fully paid leave for primary caregivers.",
            score=0.88,
        ),
    ]


def test_grounded_answer_audit_success(sample_candidate_chunks):
    response = QAResponse(
        question="What are the MFA requirements?",
        answer="Hardware-backed MFA tokens are mandatory for all remote administrative access [chunk_sec_001].",
        status=QAStatus.ANSWERED,
        has_sufficient_evidence=True,
        citations=[
            CitationItem(
                chunk_id="chunk_sec_001",
                document_id="doc_sec",
                filename="Security_Policy.pdf",
                citation_label="[Doc: Security_Policy.pdf, Page: 4]",
                page_number=4,
                snippet="Hardware-backed MFA tokens are mandatory...",
                verified=True,
            )
        ],
        supporting_snippets=["Hardware-backed MFA tokens are mandatory..."],
        model_used="qwen3:8b",
        retrieved_chunks_count=2,
    )

    report = AnswerQualityAuditor.audit_response(response, sample_candidate_chunks, is_known_unanswerable=False)

    assert report.is_grounded is True
    assert report.unsupported_claims_detected is False
    assert len(report.invalid_citations) == 0
    assert len(report.valid_citations) == 1
    assert report.evidence_coverage_ratio > 0.0


def test_hallucinated_citation_detected_and_flagged(sample_candidate_chunks):
    # Simulated response citing a fake, unretrieved chunk
    response = QAResponse(
        question="What are the encryption requirements?",
        answer="Encryption must use AES-256 [chunk_fake_999].",
        status=QAStatus.ANSWERED,
        has_sufficient_evidence=True,
        citations=[
            CitationItem(
                chunk_id="chunk_fake_999",
                document_id="doc_fake",
                filename="Fake_Doc.pdf",
                citation_label="Page 99",
                snippet="Fake content",
                verified=True,
            )
        ],
        model_used="qwen3:8b",
        retrieved_chunks_count=2,
    )

    report = AnswerQualityAuditor.audit_response(response, sample_candidate_chunks, is_known_unanswerable=False)

    assert report.is_grounded is False
    assert "chunk_fake_999" in report.invalid_citations
    assert any("hallucinated chunk ID" in note for note in report.notes)


def test_citation_metadata_mismatch_detected(sample_candidate_chunks):
    # Response references a real chunk ID but has an incorrect fabricated page number
    response = QAResponse(
        question="What are the MFA requirements?",
        answer="MFA is mandatory [chunk_sec_001].",
        status=QAStatus.ANSWERED,
        has_sufficient_evidence=True,
        citations=[
            CitationItem(
                chunk_id="chunk_sec_001",
                document_id="doc_sec",
                filename="Security_Policy.pdf",
                citation_label="Page 99",
                page_number=99,  # Genuine chunk is Page 4
                snippet="Hardware-backed MFA...",
                verified=True,
            )
        ],
        model_used="qwen3:8b",
        retrieved_chunks_count=2,
    )

    report = AnswerQualityAuditor.audit_response(response, sample_candidate_chunks, is_known_unanswerable=False)

    assert report.is_grounded is False
    assert "chunk_sec_001" in report.invalid_citations
    assert any("Citation page mismatch" in note for note in report.notes)


def test_unanswerable_question_audit_success(sample_candidate_chunks):
    # Unanswerable question properly flagged with INSUFFICIENT_EVIDENCE
    response = QAResponse(
        question="What is the meal reimbursement allowance?",
        answer="The available documents do not contain sufficient evidence to answer this question.",
        status=QAStatus.INSUFFICIENT_EVIDENCE,
        has_sufficient_evidence=False,
        citations=[],
        supporting_snippets=[],
        model_used="qwen3:8b",
        retrieved_chunks_count=2,
    )

    report = AnswerQualityAuditor.audit_response(response, sample_candidate_chunks, is_known_unanswerable=True)

    assert report.is_grounded is True
    assert report.insufficient_evidence_handled_correctly is True
    assert report.unsupported_claims_detected is False
    assert len(report.invalid_citations) == 0


def test_unanswerable_question_hallucination_detected(sample_candidate_chunks):
    # Model hallucinated an answer for an unanswerable question
    response = QAResponse(
        question="What is the meal reimbursement allowance?",
        answer="The allowance is $50 per day [chunk_sec_001].",
        status=QAStatus.ANSWERED,
        has_sufficient_evidence=True,
        citations=[
            CitationItem(
                chunk_id="chunk_sec_001",
                document_id="doc_sec",
                filename="Security_Policy.pdf",
                citation_label="Page 4",
                snippet="Hardware-backed MFA...",
                verified=True,
            )
        ],
        model_used="qwen3:8b",
        retrieved_chunks_count=2,
    )

    report = AnswerQualityAuditor.audit_response(response, sample_candidate_chunks, is_known_unanswerable=True)

    assert report.is_grounded is False
    assert report.insufficient_evidence_handled_correctly is False
    assert report.unsupported_claims_detected is True
    assert any("Model failed to refuse unanswerable query" in note for note in report.notes)
