import re
from dataclasses import dataclass, field
from typing import Any, Optional, Set

from app.models.qa import CitationItem, QAResponse, QAStatus
from app.models.search import SearchResultItem

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "don", "should", "now", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "at", "by", "with", "must"
}


def _tokenize(text: str) -> list[str]:
    """Extracts normalized alphanumeric words."""
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", text)]


def _content_words(text: str) -> set[str]:
    """Returns set of normalized content words excluding common stopwords."""
    return {w for w in _tokenize(text) if w not in STOPWORDS and len(w) > 1}


def compute_context_precision(
    retrieved_chunk_ids: list[str],
    target_chunk_ids: list[str],
    k: int = 5,
) -> float:
    """Computes Mean Average Precision (MAP) / Context Precision at K.
    Measures whether relevant chunks are ranked higher than irrelevant ones.
    """
    if not target_chunk_ids:
        return 1.0 if not retrieved_chunk_ids else 0.0

    target_set = set(target_chunk_ids)
    sub_retrieved = retrieved_chunk_ids[:k]
    if not sub_retrieved:
        return 0.0

    hits = 0
    precisions: list[float] = []
    for idx, cid in enumerate(sub_retrieved, start=1):
        if cid in target_set:
            hits += 1
            precisions.append(hits / idx)

    if not precisions:
        return 0.0

    return round(sum(precisions) / len(target_set), 4)


def compute_citation_precision(
    citations: list[Any],
    candidate_chunk_ids: list[str],
) -> float:
    """Computes citation precision: fraction of reported citations matching genuine candidate chunks."""
    if not citations:
        return 1.0
    candidate_set = set(candidate_chunk_ids)
    valid_count = sum(
        1 for c in citations
        if (c.chunk_id if hasattr(c, "chunk_id") else str(c)) in candidate_set
    )
    return round(valid_count / len(citations), 4)


def compute_lexical_faithfulness(
    answer: str,
    cited_chunks: list[Any],
) -> float:
    """Computes lexical faithfulness: proportion of substantive content words in the answer
    that appear in the supporting context chunks.
    """
    answer_words = _content_words(answer)
    if not answer_words:
        return 1.0
    if not cited_chunks:
        return 0.0

    context_words: set[str] = set()
    for chunk in cited_chunks:
        content_text = chunk.content if hasattr(chunk, "content") else str(chunk)
        context_words.update(_content_words(content_text))

    overlap = answer_words.intersection(context_words)
    return round(len(overlap) / len(answer_words), 4)


def compute_answer_relevance(
    answer: str,
    reference_answer: str,
) -> float:
    """Computes Jaccard lexical overlap between generated answer and ground-truth reference answer."""
    ans_words = _content_words(answer)
    ref_words = _content_words(reference_answer)
    if not ans_words or not ref_words:
        return 0.0
    intersection = ans_words.intersection(ref_words)
    union = ans_words.union(ref_words)
    return round(len(intersection) / len(union), 4)


@dataclass
class QualityAuditReport:
    """Detailed audit report assessing answer grounding, citation validity, and hallucination absence."""

    is_grounded: bool
    unsupported_claims_detected: bool
    invalid_citations: list[str] = field(default_factory=list)
    valid_citations: list[CitationItem] = field(default_factory=list)
    insufficient_evidence_handled_correctly: bool = True
    evidence_coverage_ratio: float = 0.0
    faithfulness_score: float = 0.0
    citation_precision: float = 1.0
    notes: list[str] = field(default_factory=list)


class AnswerQualityAuditor:
    """Evaluates whether generated answers are supported by retrieved evidence and have valid citations."""

    @staticmethod
    def audit_response(
        response: QAResponse,
        candidate_chunks: list[SearchResultItem],
        is_known_unanswerable: bool = False,
    ) -> QualityAuditReport:
        """Audits a QA response against retrieved candidate chunks and ground-truth answerability."""
        candidate_map = {c.chunk_id: c for c in candidate_chunks}
        invalid_citations: list[str] = []
        valid_citations: list[CitationItem] = []
        notes: list[str] = []

        # 1. Unanswerable question evaluation
        if is_known_unanswerable:
            insufficient_handled = (
                response.status in (QAStatus.INSUFFICIENT_EVIDENCE, QAStatus.NO_EVIDENCE_FOUND)
                and not response.has_sufficient_evidence
                and len(response.citations) == 0
            )
            if not insufficient_handled:
                notes.append("Model failed to refuse unanswerable query; generated unsupported claims or citations.")
            return QualityAuditReport(
                is_grounded=insufficient_handled,
                unsupported_claims_detected=not insufficient_handled,
                invalid_citations=[c.chunk_id for c in response.citations],
                valid_citations=[],
                insufficient_evidence_handled_correctly=insufficient_handled,
                evidence_coverage_ratio=0.0,
                faithfulness_score=1.0 if insufficient_handled else 0.0,
                citation_precision=0.0 if response.citations else 1.0,
                notes=notes,
            )

        # 2. Citation validity and metadata audit
        for citation in response.citations:
            if citation.chunk_id not in candidate_map:
                invalid_citations.append(citation.chunk_id)
                notes.append(f"Citation references unretrieved or hallucinated chunk ID: {citation.chunk_id}")
            else:
                chunk = candidate_map[citation.chunk_id]
                # Validate filename and provenance accuracy
                if citation.filename != chunk.filename:
                    invalid_citations.append(citation.chunk_id)
                    notes.append(f"Citation filename mismatch: '{citation.filename}' != '{chunk.filename}'")
                elif (
                    citation.page_number is not None
                    and chunk.page_number is not None
                    and citation.page_number != chunk.page_number
                ):
                    invalid_citations.append(citation.chunk_id)
                    notes.append(f"Citation page mismatch: {citation.page_number} != {chunk.page_number}")
                else:
                    valid_citations.append(citation)

        # 3. Grounding and claim support check
        unsupported_claims = False
        if response.status == QAStatus.ANSWERED and not valid_citations:
            unsupported_claims = True
            notes.append("Answer marked as ANSWERED but lacks any valid, verified citations.")

        # Evaluate lexical support if citations were valid
        cited_items = [candidate_map[c.chunk_id] for c in valid_citations if c.chunk_id in candidate_map]
        faithfulness = compute_lexical_faithfulness(response.answer, cited_items) if cited_items else 0.0

        if response.status == QAStatus.ANSWERED and valid_citations and faithfulness < 0.15:
            unsupported_claims = True
            notes.append(f"Low lexical faithfulness ({faithfulness}): answer content has minimal overlap with cited evidence.")

        cit_prec = compute_citation_precision(response.citations, list(candidate_map.keys()))
        coverage = len(valid_citations) / len(candidate_chunks) if candidate_chunks else 0.0

        return QualityAuditReport(
            is_grounded=(not unsupported_claims and len(invalid_citations) == 0),
            unsupported_claims_detected=unsupported_claims,
            invalid_citations=invalid_citations,
            valid_citations=valid_citations,
            insufficient_evidence_handled_correctly=True,
            evidence_coverage_ratio=round(coverage, 4),
            faithfulness_score=faithfulness,
            citation_precision=cit_prec,
            notes=notes,
        )
