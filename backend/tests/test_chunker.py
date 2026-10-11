import pytest

from app.models.document import DocumentType, ParsedBlock, ParsedDocument
from app.services.ingestion.chunker import TokenizerAwareChunker


def test_chunker_initialization_validation():
    chunker = TokenizerAwareChunker(max_tokens=256, overlap_tokens=32)
    assert chunker.max_tokens == 256
    assert chunker.overlap_tokens == 32

    # Overlap >= max_tokens must raise ValueError
    with pytest.raises(ValueError):
        TokenizerAwareChunker(max_tokens=100, overlap_tokens=100)
    with pytest.raises(ValueError):
        TokenizerAwareChunker(max_tokens=50, overlap_tokens=80)


def test_chunker_token_budget_and_counts():
    chunker = TokenizerAwareChunker(max_tokens=60, overlap_tokens=15)
    
    # Create long repetitive block
    long_text = "Corporate governance requires robust auditing and security monitoring controls. " * 15
    parsed_doc = ParsedDocument(
        filename="compliance.pdf",
        file_type=DocumentType.PDF.value,
        char_count=len(long_text),
        blocks=[ParsedBlock(text=long_text, page_number=1)],
        page_count=1,
    )

    chunks = chunker.chunk_document("doc-123", "ws-1", parsed_doc)

    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.token_count <= 60
        # Token count matches tokenizer evaluation
        assert c.token_count == chunker.count_tokens(c.content)
        assert c.char_start >= 0
        assert c.char_end > c.char_start
        assert len(c.content_hash) == 64  # SHA-256 hex length
        assert "[Doc: compliance.pdf, Page: 1]" in c.citation_label


def test_pdf_page_citation_preservation():
    chunker = TokenizerAwareChunker(max_tokens=500, overlap_tokens=50)
    parsed_doc = ParsedDocument(
        filename="manual.pdf",
        file_type=DocumentType.PDF.value,
        char_count=200,
        blocks=[
            ParsedBlock(text="Overview of authentication on page 1.", page_number=1),
            ParsedBlock(text="Role-based permissions on page 2.", page_number=2),
        ],
        page_count=2,
    )

    chunks = chunker.chunk_document("doc-pdf", "ws-1", parsed_doc)
    # Page boundary triggered separate chunks
    assert len(chunks) == 2
    assert chunks[0].citation_label == "[Doc: manual.pdf, Page: 1]"
    assert chunks[1].citation_label == "[Doc: manual.pdf, Page: 2]"


def test_docx_section_and_table_citations():
    chunker = TokenizerAwareChunker(max_tokens=500, overlap_tokens=50)
    parsed_doc = ParsedDocument(
        filename="agreement.docx",
        file_type=DocumentType.DOCX.value,
        char_count=300,
        blocks=[
            ParsedBlock(
                text="This contract binds both parties to non-disclosure obligations.",
                heading_path="Article 1: Confidentiality",
                paragraph_number=1,
            ),
            ParsedBlock(
                text="| Service | SLA |\n|---|---|\n| API | 99.9% |",
                heading_path="Article 2: SLA Commitments",
                table_number=1,
            ),
        ],
    )

    chunks = chunker.chunk_document("doc-docx", "ws-1", parsed_doc)
    assert len(chunks) == 2
    assert "Section: Article 1: Confidentiality" in chunks[0].citation_label
    assert "Para: 1" in chunks[0].citation_label
    assert "Section: Article 2: SLA Commitments" in chunks[1].citation_label
    assert "Table: 1" in chunks[1].citation_label


def test_txt_line_range_citations():
    chunker = TokenizerAwareChunker(max_tokens=500, overlap_tokens=50)
    parsed_doc = ParsedDocument(
        filename="config.txt",
        file_type=DocumentType.TXT.value,
        char_count=150,
        blocks=[
            ParsedBlock(text="DATABASE_PORT=5432\nDB_POOL=20", line_start=1, line_end=2),
        ],
    )

    chunks = chunker.chunk_document("doc-txt", "ws-1", parsed_doc)
    assert len(chunks) == 1
    assert chunks[0].citation_label == "[Doc: config.txt, Lines: 1-2]"


def test_no_text_loss_across_chunks():
    chunker = TokenizerAwareChunker(max_tokens=40, overlap_tokens=10)
    sentences = [
        "Alpha section introduces the core algorithms.",
        "Beta section analyzes the runtime complexity.",
        "Gamma section provides the empirical measurements.",
        "Delta section concludes with future research directions.",
    ]
    full_text = " ".join(sentences)
    parsed_doc = ParsedDocument(
        filename="research.pdf",
        file_type=DocumentType.PDF.value,
        char_count=len(full_text),
        blocks=[ParsedBlock(text=full_text, page_number=1)],
        page_count=1,
    )

    chunks = chunker.chunk_document("doc-loss-test", "ws-1", parsed_doc)
    combined_content = " ".join(c.content for c in chunks)

    # Every key sentence must be preserved in at least one chunk
    for s in sentences:
        assert s in combined_content
