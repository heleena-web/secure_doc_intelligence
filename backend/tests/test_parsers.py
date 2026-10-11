import io
import docx
import pymupdf
import pytest

from app.core.errors import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedEncodingError,
    UnsupportedFileTypeError,
)
from app.services.ingestion.parser import (
    DOCXParser,
    PDFParser,
    ParserFactory,
    TXTParser,
)


def test_parser_factory_supported_types():
    assert isinstance(ParserFactory.get_parser("report.pdf"), PDFParser)
    assert isinstance(ParserFactory.get_parser("POLICY.DOCX"), DOCXParser)
    assert isinstance(ParserFactory.get_parser("notes.txt"), TXTParser)


def test_parser_factory_unsupported_type():
    with pytest.raises(UnsupportedFileTypeError) as exc:
        ParserFactory.get_parser("archive.zip")
    assert exc.value.status_code == 415
    assert "zip" in exc.value.message


# ---------------------------------------------------------------------------
# PDF Parser Tests
# ---------------------------------------------------------------------------

def test_pdf_parser_valid_multi_page(sample_pdf_bytes):
    parser = PDFParser()
    doc = parser.parse(sample_pdf_bytes, "test_security.pdf")

    assert doc.filename == "test_security.pdf"
    assert doc.file_type == "pdf"
    assert doc.page_count == 2
    assert doc.char_count > 0
    assert len(doc.blocks) >= 2

    # Verify physical page numbers (1-indexed)
    page_numbers = {b.page_number for b in doc.blocks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert any("Compliance Report" in b.text for b in doc.blocks)
    assert any("Zero-trust" in b.text for b in doc.blocks)


def test_pdf_parser_empty_bytes():
    parser = PDFParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(b"", "empty.pdf")


def test_pdf_parser_blank_pages():
    parser = PDFParser()
    doc = pymupdf.open()
    doc.new_page()  # Blank page without text
    blank_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(EmptyDocumentError):
        parser.parse(blank_bytes, "blank.pdf")


def test_pdf_parser_corrupt_file():
    parser = PDFParser()
    with pytest.raises(DocumentParsingError) as exc:
        parser.parse(b"%PDF-1.5 CORRUPTED_DATA_NOT_A_REAL_STREAM", "corrupt.pdf")
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# DOCX Parser Tests
# ---------------------------------------------------------------------------

def test_docx_parser_valid(sample_docx_bytes):
    parser = DOCXParser()
    doc = parser.parse(sample_docx_bytes, "framework.docx")

    assert doc.filename == "framework.docx"
    assert doc.file_type == "docx"
    assert doc.page_count is None  # Physical pages not determinable in DOCX
    assert doc.char_count > 0

    # Verify heading hierarchy and table extraction
    headings = [b.heading_path for b in doc.blocks if b.heading_path]
    assert any("Corporate Security Framework" in h for h in headings)
    assert any("Section 2: Data Retention Guidelines" in h for h in headings)

    # Verify table block was extracted
    table_blocks = [b for b in doc.blocks if b.table_number is not None]
    assert len(table_blocks) == 1
    assert "Classification" in table_blocks[0].text
    assert "Restricted" in table_blocks[0].text


def test_docx_parser_empty_bytes():
    parser = DOCXParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(b"", "empty.docx")


def test_docx_parser_blank_document():
    parser = DOCXParser()
    doc = docx.Document()
    buf = io.BytesIO()
    doc.save(buf)

    with pytest.raises(EmptyDocumentError):
        parser.parse(buf.getvalue(), "blank.docx")


def test_docx_parser_corrupt_file():
    parser = DOCXParser()
    with pytest.raises(DocumentParsingError):
        parser.parse(b"PK\x03\x04INVALID_ZIP_NOT_A_DOCX_PAYLOAD", "corrupt.docx")


# ---------------------------------------------------------------------------
# TXT Parser Tests
# ---------------------------------------------------------------------------

def test_txt_parser_valid(sample_txt_bytes):
    parser = TXTParser()
    doc = parser.parse(sample_txt_bytes, "hardening.txt")

    assert doc.filename == "hardening.txt"
    assert doc.file_type == "txt"
    assert doc.char_count > 0
    assert len(doc.blocks) >= 2

    # Verify line range preservation
    for b in doc.blocks:
        assert b.line_start is not None
        assert b.line_end is not None
        assert b.line_start <= b.line_end


def test_txt_parser_empty_bytes():
    parser = TXTParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(b"   \n  \t  ", "empty.txt")


def test_txt_parser_unsupported_encoding():
    parser = TXTParser()
    # Construct invalid byte sequence that fails UTF-8 and chardet
    invalid_bytes = b"\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89" * 10
    with pytest.raises((UnsupportedEncodingError, DocumentParsingError, EmptyDocumentError)):
        parser.parse(invalid_bytes, "bad_encoding.txt")
