# Phased Implementation Plan: Secure Document Intelligence

This document outlines the strict phase-by-phase implementation, verification, and evaluation roadmap. Each phase must be fully implemented, tested, and approved before moving to the next.

---

## Architectural Principles & Strict Guidelines
1. **Incremental Execution**: Never implement downstream features (LLM, UI, search APIs) before upstream foundations (parsing, security, storage) are verified.
2. **Authoritative Master Store**: SQLite is the single source of truth for users, workspaces, permissions, documents, and chunk records. ChromaDB serves as an auxiliary semantic index synchronized via primary chunk UUIDs.
3. **Pre-Retrieval Security Isolation**: Permissions and workspace tenancy are enforced at the retrieval layer. No unpermitted chunk ever touches the reranker or LLM prompt context.
4. **Verified Claims Only**: A phase is only considered complete after automated test suites pass and demonstrate compliance against real files.

---

## Phase 1: Document Ingestion Foundation & Architecture
- **Status**: Completed & Verified (31/31 automated tests passing)
- **Deliverables**:
  - Environment inspection and isolated Python virtual environment (`backend/.venv`).
  - Architecture specification ([docs/ARCHITECTURE.md](file:///d:/Project/secure_doc_intelligence/docs/ARCHITECTURE.md)).
  - Format-specific parsers for PDF (PyMuPDF), DOCX (python-docx), and TXT (UTF-8/chardet).
  - Tokenizer-aware chunking engine (`tiktoken`) with overlap and format citations.
  - SQLite database schema with WAL mode, foreign keys, and cascading chunk deletion.
  - FastAPI document ingestion endpoint with deduplication and temp file cleanup.
  - Comprehensive test suite covering parsers, chunking, SQLite storage, and API contracts.
- **Verification Gate**:
  - 31/31 unit and integration tests passing (`pytest`).
  - Verified live endpoint execution with `curl` (`/health` and `/api/v1/documents/ingest`).


---

## Phase 2: Intelligent Retrieval (BM25, Semantic, Hybrid & Reranking)
- **Status**: Completed & Verified (48/48 automated tests passing)
- **Deliverables**:
  1. **BM25 Retrieval Engine (`BM25Retriever`)**:
     - `BM25Plus` implementation eliminating negative IDF anomalies on small collections.
     - Tokenization preserving policy identifiers (`POL-402`, `ISO-27001`).
     - Workspace isolation and stale chunk elimination on document deletion.
  2. **Dense Vector Store (`ChromaVectorStore`)**:
     - Persistent ChromaDB with local ONNX-based `all-MiniLM-L6-v2` embeddings (384 dimensions).
     - Pre-retrieval workspace and document permission filtering hook.
     - Stale vector deletion on document removal.
  3. **Hybrid Retrieval (`HybridRetriever`)**:
     - Reciprocal Rank Fusion (RRF) combining BM25 and vector scores with configurable \( k=60 \).
     - Stable chunk ID deduplication.
  4. **Lightweight Cross-Encoder Reranker (`FlashRankReranker`)**:
     - ONNX-based `ms-marco-TinyBERT-L-2-v2` reranker with sub-50ms CPU latency.
     - Pluggable interface (`BaseReranker`, `PassthroughReranker`) with graceful fallback.
  5. **Search API (`POST /api/v1/search`)**:
     - Multi-mode search (`bm25`, `semantic`, `hybrid`, `hybrid_rerank`).
     - Full provenance citations (`[Doc, Page]` or `[Doc, Section]`).
  6. **Evaluation Benchmark Suite (`test_evaluation.py`)**:
     - 4-way evaluation across golden test queries.
     - Measured metrics: BM25 (Recall@5: 1.0, MRR@5: 1.0), Semantic (Recall@5: 1.0, MRR@5: 0.958), Hybrid (Recall@5: 1.0, MRR@5: 1.0), Hybrid+Rerank (Recall@5: 1.0, MRR@5: 1.0).
- **Verification Gate**:
  - 48/48 automated tests passing across Phase 1 and Phase 2.
  - Live HTTP endpoint verified with curl.


---

## Phase 3: Local LLM and Grounded Answers
- **Status**: Completed & Verified (69/69 automated tests passing)
- **Deliverables**:
  1. **Local Ollama Integration (`OllamaClient`)**:
     - Async client for local Ollama daemon (`http://127.0.0.1:11434`), configurable model (`qwen3:8b`), timeout (60s), and temperature (0.0).
     - Availability checking (`check_health`), installed model validation (`validate_availability`), raising informative `OllamaUnavailableError` and `OllamaModelNotFoundError` with `ollama pull <model>` instructions.
     - Strict privacy guarantee: local inference only, zero cloud API fallback, zero telemetry or data exfiltration.
     - Handles reasoning models (cleans `<think>...</think>` tags).
  2. **Grounded Answer Service (`GroundedQAService`)**:
     - Integrates Phase 2 Hybrid (BM25 + ChromaDB RRF) and FlashRank cross-encoder reranking as the candidate evidence pipeline.
     - Early exits with `NO_EVIDENCE_FOUND` when search returns zero candidates.
     - Prompt injection defenses: context passages explicitly tagged as untrusted employee data to prevent command override attacks.
     - Refusal policy: explicitly detects insufficient evidence and transitions status to `INSUFFICIENT_EVIDENCE`.
     - Application-level citation validation: scans generated answers for cited chunk IDs, verifies them against genuine retrieved chunks, rejects hallucinated IDs, and attaches verified document provenance (`filename`, `page_number`, `heading_path`, `snippet`).
  3. **QA API Endpoints**:
     - `POST /api/v1/search/answer` and `POST /api/v1/qa/answer` returning structured `QAResponse`.
  4. **Verification & Testing**:
     - 69/69 automated tests passing in `backend/tests` (including `test_ollama_client.py`, `test_qa_service.py`, `test_qa_api.py`).
     - Live end-to-end test against active local Ollama instance (`qwen3:8b`) verified.

---

## Phase 4: Authentication and Multi-User Security
- **Status**: Completed & Verified (78/78 automated tests passing)
- **Deliverables**:
  1. **User Authentication & Password Hashing**:
     - Modern **Argon2id** password hashing (`argon2-cffi`) with memory-hard parameters.
     - Signed JSON Web Tokens (`HS256`, 24h expiration) validated on protected routes.
     - Secure endpoints: `POST /api/v1/auth/register`, `POST /api/v1/auth/login`, `GET /api/v1/auth/me`.
  2. **Multi-Tenant Workspaces & Role-Based Access Control**:
     - `workspaces` and `workspace_members` schema tracking roles: `owner`, `admin`, `member`, `viewer`.
     - Workspace tenancy enforcement: unauthorized workspace queries return HTTP 403.
     - Endpoints: `POST /api/v1/workspaces`, `GET /api/v1/workspaces`, `POST /api/v1/workspaces/{id}/members`, `DELETE /api/v1/workspaces/{id}/members/{user_id}`.
  3. **Document Ownership & Private/Shared Classification**:
     - Document ownership (`owner_id`) and least-privilege default (`is_private = true`).
     - Granular explicit permissions via `document_permissions` (`read`, `write`, `admin`).
     - Endpoints: `POST /api/v1/documents/{id}/permissions`, `DELETE /api/v1/documents/{id}/permissions/{user_id}`, `GET /api/v1/documents/{id}/permissions`.
  4. **Server-Side Pre-Retrieval ACL Enforcement**:
     - Authoritative SQLite resolution (`get_user_permitted_document_ids`) executed before search or QA.
     - Pre-retrieval filtering in BM25 and ChromaDB vector store; unpermitted chunks never reach RRF, FlashRank reranker, or Ollama prompt context.
     - Immediate permission revocation with zero stale-result latency.
  5. **Automated Security Test Suite**:
     - 9 rigorous security test cases in `backend/tests/test_security.py`.
     - Full test suite passes: 78/78 automated tests (100% pass rate).
     - Security specification documented in [docs/SECURITY.md](file:///d:/Project/secure_doc_intelligence/docs/SECURITY.md).

---

## Phase 5: Evaluation, Quality Assurance, and UX Improvements
- **Status**: Completed & Verified (85/85 automated tests passing; frontend build 100% clean)
- **Deliverables**:
  1. **Retrieval Benchmark Engine (`benchmark.py` & `evaluate_retrieval.py`)**:
     - Standardized, reproducible evaluation dataset covering PDF physical pages, DOCX hierarchical headings and table rows, and TXT line spans.
     - 4-way retrieval comparison across BM25, ChromaDB dense semantic, Hybrid RRF, and Hybrid + FlashRank reranking.
     - Measured actual metrics over 8 benchmark queries:
       - BM25: Recall@5: `1.0000`, MRR@5: `0.9167`, Precision@5: `0.2000`, Latency: `0.24 ms`
       - Semantic: Recall@5: `1.0000`, MRR@5: `1.0000`, Precision@5: `0.2000`, Latency: `288.50 ms`
       - Hybrid (RRF): Recall@5: `1.0000`, MRR@5: `1.0000`, Precision@5: `0.2000`, Latency: `291.30 ms`
       - Hybrid + Rerank: Recall@5: `1.0000`, MRR@5: `1.0000`, Precision@5: `0.2000`, Latency: `338.75 ms`
  2. **Answer and Citation Quality Assurance (`quality_assurance.py`)**:
     - `AnswerQualityAuditor` checking evidence grounding alignment between generated text and cited chunk snippets.
     - Detection and removal of hallucinated chunk IDs.
     - Verification of citation provenance metadata (PDF page number, DOCX section path, TXT line range).
     - Refusal audit guaranteeing unanswerable questions trigger `INSUFFICIENT_EVIDENCE` or `NO_EVIDENCE_FOUND` without fictitious citation fabrication.
  3. **Security Regressions & Pre-Retrieval ACL Verification**:
     - Verified rejection of forged `permitted_doc_ids` passed in query parameters via authoritative SQLite validation.
     - Re-verified cross-workspace multi-tenancy isolation and immediate revocation upon permission removal.
     - 10 automated security tests in `test_security.py` passing.
  4. **Professional Web Frontend (`frontend/`)**:
     - Built with React 18, TypeScript, and Vite (`npm run build` compiles with 0 errors/warnings).
     - Single-page interface supporting:
       - Drag-and-drop document upload with private/shared toggle.
       - Document catalog with format badges, chunk counts, and deletion.
       - Grounded Q&A interface with model and search mode selectors.
       - Formatted citation cards displaying exact page numbers, section paths, and line spans with supporting text excerpts.
       - Collapsible retrieval debugger displaying raw candidate chunk rankings and scores.
       - Modal for user registration, login, and JWT session handling.
- **Verification Gate**:
  - 85/85 backend tests passing in `pytest` (100% pass rate).
  - Frontend production build succeeds (`dist/` created).
  - Benchmark script runnable on demand (`backend/scripts/evaluate_retrieval.py`).

