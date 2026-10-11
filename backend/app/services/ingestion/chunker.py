import hashlib
import re
import uuid
from typing import Any
import tiktoken

from app.core.config import settings
from app.models.document import ChunkRecord, ParsedBlock, ParsedDocument, DocumentType


class TokenizerAwareChunker:
    """Configurable, tokenizer-aware chunker that preserves page numbers, section paths,
    table locations, and line references with sliding window overlap.
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        encoding_name: str | None = None,
    ):
        self.max_tokens = max_tokens or settings.DEFAULT_CHUNK_SIZE_TOKENS
        self.overlap_tokens = overlap_tokens or settings.DEFAULT_CHUNK_OVERLAP_TOKENS
        self.encoding_name = encoding_name or settings.TOKENIZER_ENCODING
        self.tokenizer = tiktoken.get_encoding(self.encoding_name)

        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be strictly less than max_tokens ({self.max_tokens})"
            )

    def count_tokens(self, text: str) -> int:
        """Returns exact token count for string using the configured tokenizer."""
        return len(self.tokenizer.encode(text))

    def _split_text_into_token_chunks(self, text: str) -> list[str]:
        """Splits a single large text string into chunks respecting max_tokens and overlap_tokens."""
        if self.count_tokens(text) <= self.max_tokens:
            return [text]

        # Split into sentences or lines
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
        if not raw_sentences:
            raw_sentences = [text.strip()]

        chunks: list[str] = []
        curr_sentences: list[str] = []
        curr_tokens = 0

        for s in raw_sentences:
            s_tokens = self.count_tokens(s)

            # If a single sentence exceeds max_tokens, split by words
            if s_tokens > self.max_tokens:
                words = s.split(" ")
                sub_chunk: list[str] = []
                sub_tokens = 0
                for w in words:
                    w_tok = self.count_tokens(w + " ")
                    if sub_tokens + w_tok > self.max_tokens and sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = [w]
                        sub_tokens = w_tok
                    else:
                        sub_chunk.append(w)
                        sub_tokens += w_tok
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                continue

            if curr_tokens + s_tokens > self.max_tokens and curr_sentences:
                chunk_str = " ".join(curr_sentences)
                chunks.append(chunk_str)

                # Compute overlap sentences from tail
                overlap_sentences: list[str] = []
                overlap_cnt = 0
                for prev_s in reversed(curr_sentences):
                    p_tok = self.count_tokens(prev_s)
                    if overlap_cnt + p_tok <= self.overlap_tokens:
                        overlap_sentences.insert(0, prev_s)
                        overlap_cnt += p_tok
                    else:
                        break

                curr_sentences = overlap_sentences + [s]
                curr_tokens = sum(self.count_tokens(x) for x in curr_sentences)
            else:
                curr_sentences.append(s)
                curr_tokens += s_tokens

        if curr_sentences:
            chunks.append(" ".join(curr_sentences))

        return chunks

    def _format_citation(
        self,
        filename: str,
        file_type: str,
        pages: list[int],
        headings: list[str],
        paragraphs: list[int],
        tables: list[int],
        lines: list[tuple[int, int]],
    ) -> str:
        """Constructs format-compliant citation label."""
        if file_type == DocumentType.PDF.value:
            if pages:
                min_p, max_p = min(pages), max(pages)
                if min_p == max_p:
                    return f"[Doc: {filename}, Page: {min_p}]"
                return f"[Doc: {filename}, Pages: {min_p}-{max_p}]"
            return f"[Doc: {filename}]"

        elif file_type == DocumentType.DOCX.value:
            heading = headings[0] if headings else "Document Body"
            if tables:
                min_t, max_t = min(tables), max(tables)
                tbl_str = f"Table: {min_t}" if min_t == max_t else f"Tables: {min_t}-{max_t}"
                return f"[Doc: {filename}, Section: {heading}, {tbl_str}]"
            elif paragraphs:
                min_p, max_p = min(paragraphs), max(paragraphs)
                p_str = f"Para: {min_p}" if min_p == max_p else f"Paras: {min_p}-{max_p}"
                return f"[Doc: {filename}, Section: {heading}, {p_str}]"
            return f"[Doc: {filename}, Section: {heading}]"

        elif file_type == DocumentType.TXT.value:
            if lines:
                start_l = min(l[0] for l in lines)
                end_l = max(l[1] for l in lines)
                return f"[Doc: {filename}, Lines: {start_l}-{end_l}]"
            return f"[Doc: {filename}]"

        elif file_type in (DocumentType.NOTION.value, "notion"):
            heading = headings[0] if headings else ""
            if heading:
                return f"[Notion: {filename}, Section: {heading}]"
            return f"[Notion: {filename}]"

        return f"[Doc: {filename}]"

    def chunk_document(
        self,
        document_id: str,
        workspace_id: str,
        parsed_doc: ParsedDocument,
    ) -> list[ChunkRecord]:
        """Splits a ParsedDocument into an ordered sequence of ChunkRecords with full provenance."""
        chunks: list[ChunkRecord] = []
        chunk_idx = 0
        char_offset = 0

        # Process blocks
        # We group contiguous blocks sharing the same structural context until max_tokens is reached
        accumulated_text: list[str] = []
        accumulated_pages: list[int] = []
        accumulated_headings: list[str] = []
        accumulated_paras: list[int] = []
        accumulated_tables: list[int] = []
        accumulated_lines: list[tuple[int, int]] = []
        accumulated_meta: dict[str, Any] = {}
        accumulated_tokens = 0

        def emit_accumulated_chunk():
            nonlocal chunk_idx, char_offset, accumulated_text, accumulated_pages
            nonlocal accumulated_headings, accumulated_paras, accumulated_tables
            nonlocal accumulated_lines, accumulated_meta, accumulated_tokens

            if not accumulated_text:
                return

            full_text = " ".join(accumulated_text).strip()
            if not full_text:
                return

            # If accumulated text is within max_tokens, emit directly
            # If larger (e.g. single huge block), split with overlap
            sub_chunks = self._split_text_into_token_chunks(full_text)

            for sub_text in sub_chunks:
                tok_cnt = self.count_tokens(sub_text)
                c_start = char_offset
                c_end = char_offset + len(sub_text)
                char_offset = c_end + 1  # +1 for spacing

                c_hash = hashlib.sha256(sub_text.encode("utf-8")).hexdigest()
                citation = self._format_citation(
                    filename=parsed_doc.filename,
                    file_type=parsed_doc.file_type,
                    pages=accumulated_pages,
                    headings=accumulated_headings,
                    paragraphs=accumulated_paras,
                    tables=accumulated_tables,
                    lines=accumulated_lines,
                )

                chunk_meta = {
                    **accumulated_meta,
                    "pages": sorted(list(set(accumulated_pages))),
                    "headings": list(set(accumulated_headings)),
                    "paragraphs": sorted(list(set(accumulated_paras))),
                    "tables": sorted(list(set(accumulated_tables))),
                    "line_ranges": accumulated_lines,
                }

                chunks.append(
                    ChunkRecord(
                        id=str(uuid.uuid4()),
                        document_id=document_id,
                        workspace_id=workspace_id,
                        chunk_index=chunk_idx,
                        char_start=c_start,
                        char_end=c_end,
                        token_count=tok_cnt,
                        content=sub_text,
                        citation_label=citation,
                        metadata=chunk_meta,
                        content_hash=c_hash,
                    )
                )
                chunk_idx += 1

            # Reset accumulated state
            accumulated_text = []
            accumulated_pages = []
            accumulated_headings = []
            accumulated_paras = []
            accumulated_tables = []
            accumulated_lines = []
            accumulated_meta = {}
            accumulated_tokens = 0

        for block in parsed_doc.blocks:
            block_tokens = self.count_tokens(block.text)

            # Check if adding this block exceeds max_tokens or changes primary context
            # Note: For PDF, keep same-page blocks together if possible
            is_new_page = (
                parsed_doc.file_type == DocumentType.PDF.value
                and accumulated_pages
                and block.page_number
                and block.page_number != accumulated_pages[-1]
            )
            is_new_heading = (
                parsed_doc.file_type in (DocumentType.DOCX.value, DocumentType.NOTION.value, "notion")
                and accumulated_headings
                and block.heading_path
                and block.heading_path != accumulated_headings[-1]
            )

            if (accumulated_tokens + block_tokens > self.max_tokens) or is_new_page or is_new_heading:
                emit_accumulated_chunk()

            accumulated_text.append(block.text)
            accumulated_tokens += block_tokens

            if block.page_number is not None:
                accumulated_pages.append(block.page_number)
            if block.heading_path is not None:
                accumulated_headings.append(block.heading_path)
            if block.paragraph_number is not None:
                accumulated_paras.append(block.paragraph_number)
            if block.table_number is not None:
                accumulated_tables.append(block.table_number)
            if block.line_start is not None and block.line_end is not None:
                accumulated_lines.append((block.line_start, block.line_end))

            accumulated_meta.update(block.metadata)

        # Flush any remaining accumulated text
        emit_accumulated_chunk()

        return chunks
