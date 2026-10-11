# System Architecture: Secure Document Intelligence

This document specifies the technical architecture, security model, storage responsibilities, ingestion pipeline, retrieval mechanics, and lifecycle management for the Secure Document Intelligence platform.

---

## 1. Security & Access Control Architecture

### 1.1 Authentication & Authorization Flow

```
[ Client Request ]
       |
       v
+-----------------------------------------------------------------------------------+
| FastAPI Request Pipeline                                                         |
|                                                                                   |
|  1. Extract Bearer Token (JWT)                                                    |
|  2. Verify Signature & Expiry against SQLite User Records                         |
|  3. Resolve Identity:                                                             |
|     - user_id: UUID                                                               |
|     - active_workspace_id: UUID                                                   |
|     - user_roles: ["finance_viewer", "compliance_officer"]                        |
|  4. Authorize Workspace Membership:                                               |
|     Check: (user_id, active_workspace_id) in SQLite `workspace_members`           |
|     Fail -> 403 Forbidden                                                         |
+-----------------------------------------------------------------------------------+
       |
       | Query: "What are the Q3 financial covenants?"
       v
+-----------------------------------------------------------------------------------+
| Pre-Retrieval Security Filter Construction                                       |
|                                                                                   |
|  Query SQLite for permitted document IDs:                                         |
|  SELECT d.id FROM documents d                                                     |
|  JOIN document_permissions dp ON d.id = dp.document_id                            |
|  WHERE d.workspace_id = :active_workspace_id                                      |
|    AND (dp.role IN (:user_roles) OR dp.user_id = :user_id)                        |
|                                                                                   |
|  Resulting Permitted Doc Set: S_permitted = {doc_1, doc_4, doc_8}                 |
+-----------------------------------------------------------------------------------+
       |
       +------------------------------------+
       |                                    |
       v                                    v
+----------------------------+   +--------------------------------------------------+
| SQLite FTS5 (BM25 Index)   |   | ChromaDB (Vector Store)                          |
|                            |   |                                                  |
| Filter:                    |   | Metadata Filter:                                 |
| document_id IN S_permitted |   | { "$and": [                                      |
|                            |   |     { "workspace_id": active_workspace_id },     |
|                            |   |     { "doc_id": { "$in": S_permitted } }         |
|                            |   | ] }                                              |
+----------------------------+   +--------------------------------------------------+
       |                                    |
       +-----------------+------------------+
                         |
                         v
          [ Authorized Top-K Candidates Only ]
                         |
                         v
          +-------------------------------+
          | FlashRank Reranker            | (Zero unpermitted data ever reaches reranker)
          +---------------+---------------+
                          |
                          v
          +-------------------------------+
          | Ollama LLM Context Window     | (Zero unpermitted data ever reaches LLM prompt)
          +-------------------------------+
```

### 1.2 Multi-Tenant Isolation Guarantees
- **No Shared Execution Spaces**: Every query requires an explicit `active_workspace_id` header or session state.
- **Pre-Retrieval Boundary**: Retrieval engines (both vector and keyword) never execute open queries. Filtering by authorized document IDs occurs *prior* to or *at* query execution time.
- **Fail-Closed Policy**: If a user belongs to no roles or has no document access granted in the workspace, `S_permitted` is empty (`∅`), and retrieval immediately short-circuits with 0 candidates without invoking ChromaDB, FTS5, Reranker, or LLM.

### 1.3 Security Testing Strategy
1. **Direct Document Access Tests**: Attempting to fetch document metadata, raw text, or download links for documents in another workspace or without role permissions must immediately yield `HTTP 403 Forbidden`.
2. **Unauthorized Keyword & Semantic Query Leak Tests**: Ingest document A (restricted to `legal`) containing a canary token (e.g., `"CANARY_TOP_SECRET_COVENANT_XYZ"`). Query with user B (role: `sales`) targeting this canary phrase. Assert that:
   - BM25 candidate list contains 0 results.
   - ChromaDB returns 0 results.
   - Cross-encoder reranker receives 0 chunks.
   - LLM prompt context contains 0 occurrences of the canary.
3. **Dynamic Permission Revocation Tests**: Ingest document A, query with user B (initially granted role `analyst`). Verify successful retrieval. Update SQLite to revoke `analyst` access from document A. Re-run query immediately without touching vector store. Assert that user B's retrieval instantly drops to 0 results.
4. **Cross-User Workspace Boundary Tests**: Users authenticated in Workspace 1 can never view or search chunks belonging to Workspace 2, even if both workspaces contain identically named documents or identical user role labels.

---

## 2. Storage Responsibilities & Data Consistency Model

The system utilizes a dual-engine architecture with **SQLite as the single authoritative source of truth**.

```
+------------------------------------------------------------------------------------+
|                         SQLite (Authoritative Master DB)                           |
|                                                                                    |
|  - Users & Credentials (hashed passwords, active status)                           |
|  - Workspaces & Workspace Memberships (user_id, workspace_id, role)                |
|  - Documents (id, workspace_id, name, hash_sha256, file_type, size, status)       |
|  - Document Permissions (id, document_id, role, user_id)                          |
|  - Ingestion Jobs (id, document_id, status, error_message, created_at)             |
|  - Chunks (id, document_id, workspace_id, chunk_index, page_num, section_heading,  |
|            char_start, char_end, token_count, content, content_hash)               |
|  - FTS5 Virtual Table (BM25 keyword search index, indexed on chunks.content)       |
+------------------------------------------------------------------------------------+
                                      |
                      Transactional Sync & Ingestion
                                      |
                                      v
+------------------------------------------------------------------------------------+
|                         ChromaDB (Derived Vector Store)                            |
|                                                                                    |
|  - Collection per workspace OR unified collection with metadata tags:             |
|    - id: matches SQLite `chunks.id` (UUID)                                         |
|    - document: `chunks.content`                                                    |
|    - metadata: { "doc_id": ..., "workspace_id": ..., "chunk_index": ... }         |
+------------------------------------------------------------------------------------+
```

### 2.1 Storage Roles & Boundary Rules
1. **SQLite (Authoritative)**:
   - All relational entities, access control lists (ACLs), user profiles, and ingestion state machines live exclusively in SQLite.
   - Contains the authoritative master copy of all chunks and their extracted text.
   - Hosts the SQLite **FTS5 (Full-Text Search)** virtual table for BM25 keyword search, enabling transactional joins between full-text queries and permission tables.
2. **ChromaDB (Derived Index)**:
   - Operates purely as a secondary semantic index.
   - ChromaDB IDs are guaranteed to be identical to SQLite `chunks.id`.
   - ChromaDB stores metadata (`doc_id`, `workspace_id`) solely to support vector pre-filtering.
   - If ChromaDB is ever corrupted or lost, it can be 100% reconstructed deterministically from SQLite records without re-parsing raw files.

### 2.2 Consistency & Transactional Rollback
- **Two-Phase Write during Ingestion**:
  1. Parse document, calculate content hash.
  2. Insert document record and chunk records into SQLite in a single transaction with status `INGESTING`.
  3. Insert corresponding chunk embeddings into ChromaDB tagged with the generated SQLite chunk IDs.
  4. Update SQLite document status to `INDEXED`.
  5. If embedding or ChromaDB fails, roll back SQLite transaction and delete partial vectors, setting status to `FAILED`.
- **Cascading Deletions**:
  - When a document is deleted:
    1. Query all `chunks.id` for `document_id` from SQLite.
    2. Delete those IDs from ChromaDB.
    3. Delete document record in SQLite (cascading foreign keys remove chunk records and FTS5 entries).

---

## 3. Parsing & Citation Specification

Physical page numbers cannot be reliably extracted from all file formats. The platform defines strict format-specific citation semantics:

```
+-----------+-------------------------+--------------------------------------------------------------+
| File Type | Primary Locator         | Exact Citation Format Example                                |
+-----------+-------------------------+--------------------------------------------------------------+
| PDF       | Physical Page Number    | [Doc: Security_Policy.pdf, Page 14]                          |
| DOCX      | Heading / Section / Para| [Doc: Employment_Contract.docx, Section: 3.2 Benefits, P: 4] |
| TXT       | Line Range & Heading    | [Doc: server_config.txt, Lines 45-72]                        |
+-----------+-------------------------+--------------------------------------------------------------+
```

### 3.1 PDF Extraction (PyMuPDF / Fitz)
- Iterates over document page by page (`doc.pages()`).
- Yields `page_number` (1-indexed).
- Tracks bounding rectangles and blocks to preserve reading order across multi-column layouts.
- Output chunks retain `page_start` and `page_end` (in case a chunk spans across a page boundary).

### 3.2 DOCX Extraction (python-docx)
- **Technical Reality**: `python-docx` parses XML structures (`w:p`, `w:tbl`, `w:tr`). Physical page numbers do not exist in DOCX files because pages are rendered dynamically based on fonts, page margins, and client viewport.
- **Section-Based Citation Model (v1)**:
  - Traverses the document while maintaining an active heading stack (e.g., `["1. Overview", "1.2 Scope"]`).
  - Identifies paragraph index (`paragraph_index`) and table index (`table_index`).
  - Emits location metadata:
    - `heading_path`: `"Section 3 > Subsection A"`
    - `paragraph_number`: `12`
    - `table_number`: `2` (if inside a table)
  - Citations reference the hierarchical section and paragraph: `[Doc: Contract.docx, Section: Termination, Para: 3]`.

### 3.3 TXT Extraction
- Streams content with automatic encoding detection (`chardet` / UTF-8 fallback).
- Tracks 1-indexed line numbers (`line_start`, `line_end`) and character offsets.
- Detects Markdown-style headers (`#`, `##`) if present to enrich chunk breadcrumbs.

---

## 4. Tokenizer-Aware Chunking Strategy

Naive character count chunking (e.g., 2000 characters) leads to unpredictable token lengths, overflowing embedding models and LLM context budgets. The chunking engine is **tokenizer-aware**:

```
+-----------------------------------------------------------------------------------+
| Chunker Configuration                                                             |
|                                                                                   |
|  - tokenizer: "cl100k_base" (tiktoken) or HuggingFace AutoTokenizer               |
|  - max_tokens: 512 tokens                                                         |
|  - overlap_tokens: 64 tokens                                                      |
|  - delimiters: ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]                   |
|  - metadata_prefix: True (injects Doc Name & Section into chunk context)           |
+-----------------------------------------------------------------------------------+
```

### 4.1 Recursive Token-Aware Splitting Pipeline
1. **Structural Segmentation**: Split raw document into logical blocks based on document hierarchy (pages for PDF, headings/paragraphs for DOCX).
2. **Tokenizer Counting**: Count tokens using an exact tokenizer matching the embedding model / LLM tokenizer.
3. **Recursive Division**: If a block exceeds `max_tokens`:
   - Split hierarchically by delimiters (`\n\n` -> `\n` -> sentence boundaries -> words).
   - Merge split units until adding the next unit would exceed `max_tokens`.
   - Apply `overlap_tokens` sliding window from the end of the previous chunk to maintain context across boundaries.
4. **Header Breadcrumb Injection**:
   - For DOCX: prepend `[Context: Document Title > Heading Name]\n` to the chunk text before embedding, ensuring dense vectors capture high-level section semantics.

---

## 5. Swappable Reranking Architecture

### 5.1 Reranker Selection Rationale: FlashRank
For local-first operation without requiring a dedicated GPU or multi-gigabyte PyTorch weight downloads, **FlashRank** is selected as the default local reranker:
- **Footprint**: Ultra-lightweight (runs on ONNX Runtime, models ~4MB–100MB e.g. `ms-marco-TinyBERT-L-2-v2` or `ms-marco-MiniLM-L-12-v2`).
- **Speed**: Sub-50ms CPU latency per query for 25 candidate chunks.
- **Independence**: Zero PyTorch / CUDA runtime dependency required.

### 5.2 Pluggable Abstract Interface
To prevent vendor lock-in and enable zero-code swap to Cross-Encoders or local models, the reranker adheres to an abstract interface:

```python
from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

class CandidateChunk(BaseModel):
    chunk_id: str
    doc_id: str
    document_name: str
    location_label: str  # "Page 14" or "Section 3.2, Para 4"
    content: str
    initial_score: float

class RerankedChunk(CandidateChunk):
    rerank_score: float

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: List[CandidateChunk], top_k: int) -> List[RerankedChunk]:
        """Reranks candidate chunks based on relevance to query."""
        pass
```

Alternative implementations (`SentenceTransformerCrossEncoderReranker`, `FlashRankReranker`, `PassthroughReranker`) can be selected via environment configuration (`RERANKER_TYPE=flashrank`).

---

## 6. Document Lifecycle Management

```
[ Upload Document ]
        |
        v
+--------------------------+
| 1. SHA-256 Hash Check    | === (Hash matches existing doc in workspace) ===> [ Return Existing / 304 ]
+------------+-------------+
             | New Hash
             v
+--------------------------+
| 2. Ingestion Job Queued  |
+------------+-------------+
             |
             v
+--------------------------+
| 3. Parse & Token Chunk   |
+------------+-------------+
             |
             v
+--------------------------+
| 4. Atomic Multi-Store    |
|    - SQLite Master Chunks|
|    - SQLite FTS5 Index   |
|    - ChromaDB Embeddings |
+------------+-------------+
             |
             v
+--------------------------+
| 5. Status: INDEXED       |
+--------------------------+
```

### 6.1 Duplicate Detection
- At upload time, compute `sha256(file_bytes)`.
- Query SQLite: `SELECT id, status FROM documents WHERE workspace_id = :ws_id AND hash_sha256 = :hash`.
- If an active document exists with the identical hash, bypass re-processing and return existing document record.

### 6.2 Document Updates & Versioning
- When a document is re-uploaded with the same filename but different content hash:
  1. Ingest new version as a staging record.
  2. In an atomic transaction: mark old version as `SUPERSEDED` or delete its chunks from SQLite and ChromaDB.
  3. Promote new version to `INDEXED`.

### 6.3 Immediate Permission Updates
- Document permissions (`document_permissions`) are evaluated at the SQLite query level during pre-retrieval.
- Granting or revoking a role takes effect **immediately** on the very next query.
- **No re-embedding or re-indexing is required** upon permission changes, because vector and keyword indices do not bake static ACL rules into frozen chunk embeddings.

---

## 7. Retrieval Evaluation & Benchmark Framework

To scientifically validate search quality and ensure regression resistance, the platform includes an automated evaluation harness:

```
+-----------------------------------------------------------------------------------+
| Retrieval Strategy Evaluation Matrix                                              |
|                                                                                   |
|  Pipeline A: BM25 Keyword Search only (SQLite FTS5)                               |
|  Pipeline B: Dense Vector Semantic Search only (ChromaDB)                         |
|  Pipeline C: Hybrid Retrieval (Reciprocal Rank Fusion - RRF)                      |
|  Pipeline D: Hybrid Retrieval + Reranking (RRF + FlashRank)                       |
+-----------------------------------------------------------------------------------+
```

### 7.1 Evaluation Metrics
For a curated golden dataset of test queries with known ground-truth source chunks:
- **Precision@K** (\(K \in \{3, 5\}\)): Proportion of retrieved chunks that are relevant.
- **Recall@K** (\(K \in \{3, 5\}\)): Proportion of all relevant chunks successfully retrieved.
- **MRR (Mean Reciprocal Rank)**: Position of the first relevant chunk in candidate list.
- **Citation Accuracy**: Verifies whether cited `[Doc, Page/Section]` matches the ground-truth location of the evidence text.

### 7.2 Zero-Leakage Security Verification Gate
- Automated negative tests execute queries designed to trigger cross-workspace and unauthorized hits.
- Assertion: Retrieval metrics for unauthorized documents must equal exactly **0.00** across all four pipelines.
