# Secure Document Intelligence

A privacy-focused, permission-aware document intelligence platform that enables employees to securely search internal organizational documents and receive evidence-based answers with verified citations.

---

## 1. Project Overview

**Secure Document Intelligence** is built around three foundational requirements:
1. **Security-by-Design & Pre-Retrieval Enforcement**: Document permissions and workspace isolation are enforced *at the retrieval layer*. Unauthorized document content is guaranteed to never reach candidate lists, rerankers, or LLM context prompts.
2. **Authoritative Master Storage**: SQLite acts as the authoritative single source of truth for users, workspaces, permissions, documents, chunks, and BM25 keyword indexing (via SQLite FTS5). ChromaDB serves as an auxiliary, synchronized semantic vector store.
3. **Format-Aware Provenance & Citations**:
   - **PDF**: Physical page numbers (1-indexed) and bounding coordinates via PyMuPDF (`fitz`).
   - **DOCX**: Hierarchical section headers, paragraph indices, and table locations via `python-docx` (section-based citations; no false claims of physical page recovery).
   - **TXT**: 1-indexed line spans and character offsets.
   - **Notion**: Block-tree hierarchy (`heading_1-3`, paragraphs, tables, code blocks, lists) with direct canonical Notion page URLs.
4. **Local-First Privacy**: Local embeddings and local LLM inference via Ollama (`llama3.2`, `qwen3`, `mistral`) ensure complete data residency and zero exfiltration.
5. **Notion Knowledge Integration (Phase 6)**: Official Notion API v1 integration with incremental synchronization, content hashing deduplication, automatic cleanup of stale chunks and deleted pages, and pre-retrieval ACL enforcement.

---

## 2. Core Architecture

```
                                  +-----------------------------+
                                  |     Clean Web Frontend      |
                                  |   (React / Vite / Tailwind) |
                                  +--------------+--------------+
                                                 |
                                         REST / Streaming API
                                                 |
                                  +--------------v--------------+
                                  |       FastAPI Backend       |
                                  |  - JWT & Workspace Auth     |
                                  |  - Pre-Retrieval ACL Guard  |
                                  +--------------+--------------+
                                                 |
                      Query SQLite for authorized document IDs in active workspace
                                                 |
                         +-----------------------+-----------------------+
                         |                                               |
                         v                                               v
       +-----------------------------------+           +-----------------------------------+
       |     SQLite FTS5 (BM25 Engine)     |           |     ChromaDB (Dense Vector Store) |
       |                                   |           |                                   |
       | Filter: doc_id IN :permitted_ids  |           | Filter: doc_id IN :permitted_ids  |
       +-----------------+-----------------+           +-----------------+-----------------+
                         |                                               |
                         +-----------------------+-----------------------+
                                                 |
                                         Reciprocal Rank Fusion (RRF)
                                                 |
                                                 v
                                  +-----------------------------+
                                  | Swappable Local Reranker    |
                                  | (FlashRank - Lightweight)   |
                                  +--------------+--------------+
                                                 | Top-K Authorized Chunks
                                                 v
                                  +-----------------------------+
                                  | Local Ollama LLM Inference  |
                                  | (Evidence Citations Only)   |
                                  +-----------------------------+
```

---

## 3. Storage Responsibilities & Data Synchronization

| Storage Engine | Role | Responsibilities |
| :--- | :--- | :--- |
| **SQLite (Master)** | Authoritative Single Source of Truth | - User authentication, passwords, and tokens<br>- Workspace hierarchies and user memberships<br>- Document records, SHA-256 deduplication hashes, and status<br>- Role-based access control (ACL) permissions<br>- Master chunk texts, token counts, and provenance metadata<br>- FTS5 full-text search index for BM25 keyword retrieval |
| **ChromaDB** | Secondary Semantic Vector Store | - Chunk vector embeddings<br>- Synchronized using identical chunk UUIDs from SQLite<br>- Workspace and document ID metadata for vector pre-filtering<br>- Fully rebuildable on demand from SQLite records |

### Document Lifecycle & Consistency
- **Deduplication**: Ingested files are hashed (`sha256`). Re-uploading an unchanged file returns the existing document record immediately.
- **Atomic Deletions**: Deleting a document removes its vectors from ChromaDB and cascades through SQLite to delete all associated chunk records, permissions, and FTS5 index rows.
- **Instant Permission Revocation**: Because permissions are resolved dynamically from SQLite during pre-retrieval filtering, permission changes take effect on the very next query without requiring re-indexing or re-embedding.

---

## 4. Citation Specifications by Document Format

| Format | Parsing Engine | Provenance Strategy | Citation Example |
| :--- | :--- | :--- | :--- |
| **PDF** | PyMuPDF (`fitz`) | 1-indexed physical page numbers & text block geometry | `[Doc: Financial_Report.pdf, Page: 14]` |
| **DOCX** | `python-docx` | Heading hierarchy (`heading_path`), paragraph number, table index | `[Doc: Master_Agreement.docx, Section: 4.1 Termination, Para: 3]` |
| **TXT** | UTF-8 Stream | Line spans (`line_start`-`line_end`) & section headers | `[Doc: config_guide.txt, Lines: 45-68]` |
| **Notion** | Notion API v1 Blocks | Block tree traversal, heading hierarchy, and source page link | `[Notion: Engineering Handbook, Section: Incident Management]` |

---

## 5. Tokenizer-Aware Chunking Strategy

Unlike naive character-splitting approaches, the chunking pipeline uses tokenizer-aware splitting:
- **Tokenizer**: Compatible with embedding model / LLM token boundaries (e.g. `cl100k_base` via `tiktoken` or HuggingFace tokenizers).
- **Configurable Parameters**:
  - `max_tokens`: Default 512 tokens per chunk.
  - `overlap_tokens`: Default 64 tokens sliding window.
  - `split_delimiters`: Hierarchical boundaries (`\n\n`, `\n`, sentence endings `. `, `? `, `! `, words).
- **Context Injection**: Prepends document title and section breadcrumbs to chunks to preserve topical context during dense retrieval.

---

## 6. Swappable Reranking Architecture

- **Default Engine**: **FlashRank**
  - **Rationale**: Ultra-lightweight (runs on ONNX Runtime), tiny model footprint (4MB–100MB), zero PyTorch/GPU dependency, sub-50ms CPU execution time.
- **Pluggable Interface**: Abstract `BaseReranker` class allows seamless swapping with HuggingFace Cross-Encoders, Cohere local, or alternative rerankers via configuration without changing retrieval code.

---

## 7. Retrieval Evaluation & Measured Benchmark Results

The platform contains an automated, reproducible benchmark harness (`backend/app/services/evaluation/benchmark.py`) evaluated across 8 multi-format test queries (PDF physical pages, DOCX hierarchical sections/tables, and TXT line spans).

You can run the benchmark at any time via:
```powershell
.\backend\.venv\Scripts\python backend/scripts/evaluate_retrieval.py
```

### Measured Retrieval Performance (8 Ground-Truth Queries, K=5)

The metrics below represent **actual measured results** executed against the reproducible multi-format document corpus on Windows 11 / Python 3.11:

| Mode | Recall@5 | MRR@5 | Precision@5 | Latency (avg) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BM25 Only** (SQLite FTS5) | **1.0000** | 0.9167 | 0.2000 | **0.24 ms** | Sub-millisecond keyword lookup; outstanding for exact policy IDs (`POL-402`, `ISO-27001`). Lower MRR when natural language wording differs from document text. |
| **Semantic Only** (ChromaDB) | **1.0000** | **1.0000** | 0.2000 | 288.50 ms | Captures conceptual semantic intent (`all-MiniLM-L6-v2`); perfect ranking on benchmark queries. |
| **Hybrid (RRF)** (BM25 + Chroma) | **1.0000** | **1.0000** | 0.2000 | 291.30 ms | Reciprocal Rank Fusion ($k=60$) balances keyword precision and dense semantics. Resilient to vocabulary mismatch. |
| **Hybrid + Rerank** (FlashRank) | **1.0000** | **1.0000** | 0.2000 | 338.75 ms | Re-ranks top fusion candidates using cross-encoder (`ms-marco-TinyBERT-L-2-v2`) on CPU with minimal latency overhead. |

> [!NOTE]
> All four retrieval configurations achieved 100% Recall@5 on the curated evaluation set because the ground truth chunks fell within the top 5 candidates. Hybrid and Hybrid+Rerank yielded optimal MRR@5 (1.0000) by consistently positioning the exact ground-truth passage in rank 1.

---

## 8. Answer & Citation Quality Assurance

The system implements automated quality auditing (`AnswerQualityAuditor` in `backend/app/services/evaluation/quality_assurance.py`) to guarantee factual grounding and eliminate hallucinations:

1. **Evidence Grounding Verification**: Evaluates whether statements generated by the LLM are supported by cited retrieved snippets, checking semantic lexical overlap and n-gram alignment.
2. **Hallucinated Citation Rejection**: Scans cited chunk IDs against genuine chunk IDs retrieved in the authorized pre-retrieval candidate set. Unrecognized or forged IDs are flagged and discarded.
3. **Provenance Validation**: Validates that citation metadata points to real documents, genuine page numbers (1 to total pages for PDF), existing section headings (DOCX), or valid line spans (TXT).
4. **Refusal on Unanswerable Queries**: For out-of-domain or unanswerable queries, the pipeline guarantees refusal (status `INSUFFICIENT_EVIDENCE` or `NO_EVIDENCE_FOUND`) without hallucinating facts or fabricated citations.

---

## 9. API Endpoints

### Authentication & Workspaces
- `POST /api/v1/auth/register`: User registration with Argon2id password hashing.
- `POST /api/v1/auth/login`: User login returning signed JWT access token.
- `GET /api/v1/auth/me`: Current user profile.
- `POST /api/v1/workspaces`: Create a new workspace (creator is assigned as `owner`).
- `GET /api/v1/workspaces`: List workspaces user belongs to.
- `POST /api/v1/workspaces/{id}/members`: Add user to workspace (`owner`/`admin` only).
- `DELETE /api/v1/workspaces/{id}/members/{user_id}`: Remove user from workspace.

### Document Management & Permissions
- `POST /api/v1/documents/ingest`: Ingest PDF, DOCX, or TXT documents (private by default, with `is_private` toggle).
- `GET /api/v1/documents`: List authorized documents in workspace.
- `GET /api/v1/documents/{id}`: Get document metadata (permission-verified).
- `GET /api/v1/documents/{id}/chunks`: Inspect chunks and citations (permission-verified).
- `DELETE /api/v1/documents/{id}`: Delete document and cascade delete chunks/indices.
- `POST /api/v1/documents/{id}/permissions`: Grant explicit document permissions (`read`, `write`, `admin`).
- `DELETE /api/v1/documents/{id}/permissions/{user_id}`: Revoke explicit document permission.
- `GET /api/v1/documents/{id}/permissions`: List explicit document permissions.

### Notion Knowledge Integration
- `GET /api/v1/notion/status`: Inspect active workspace Notion connection and bot details.
- `POST /api/v1/notion/connect`: Connect workspace via Notion internal integration secret.
- `POST /api/v1/notion/disconnect`: Disconnect Notion workspace integration.
- `GET /api/v1/notion/pages`: List Notion pages shared with the integration bot.
- `POST /api/v1/notion/sync`: Incrementally synchronize pages, update vector and BM25 indexes, and prune deleted pages.

### Intelligent Retrieval & Grounded QA
- `POST /api/v1/search`: Multi-mode search (`bm25`, `semantic`, `hybrid`, `hybrid_rerank`) with server-side pre-retrieval ACL filtering.
- `POST /api/v1/search/answer` (alias: `POST /api/v1/qa/answer`):
  - Accepts employee question, workspace ID, and optional permission filter.
  - Enforces pre-retrieval ACL filtering; unauthorized passages never reach reranker or Ollama.
  - Retrieves candidates via Hybrid RRF + FlashRank reranker.
  - Executes local Ollama inference (`qwen3:8b`).
  - Verifies citations against authoritative chunk metadata.
  - Returns answer, verified citations, supporting snippets, and evidence status (`ANSWERED`, `INSUFFICIENT_EVIDENCE`, `NO_EVIDENCE_FOUND`).

---

## 10. Web User Experience (Frontend)

The application provides a clean, professional, enterprise-grade React/TypeScript frontend located in `frontend/`:
- **Document Ingestion Dropzone**: Single-click file selection for PDF, DOCX, and TXT with a private/workspace-shared toggle.
- **Document Library**: Displays document title, format badge, chunk count, file size, privacy status, and deletion action.
- **Notion Knowledge Base Card**: Displays Notion connection status, bot information, "Sync Now" button with spinner, and collapsible page discovery list.
- **Intelligent Grounded Q&A**: Question input bar with mode selector (Hybrid+Rerank, Hybrid, Semantic, BM25) and real-time query latency badge.
- **Verified Citation Cards**: Visual cards displaying document title, physical page number (PDF), section path (DOCX), line range (TXT), or Notion page badge and direct URL link, with supporting chunk excerpts.
- **Collapsible Retrieval Debugger**: Live inspection drawer displaying raw retrieved candidate chunks, relevance scores, and provenance metadata.

---

## 11. Quickstart & Local Setup

### Prerequisites
- Python 3.11+
- Node.js 20+ and npm
- [Ollama](https://ollama.com/) running locally with model installed:
  ```powershell
  ollama pull qwen3:8b
  ```

### Backend Setup
```powershell
# Navigate to backend directory
cd D:\Project\secure_doc_intelligence\backend

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Run the test suite (111 automated tests covering ACLs, parsers, Notion, QA, and Enterprise RAG)
pytest -v

# Run the reproducible retrieval evaluation benchmark (Expanded dataset: 24 chunks, 20 queries)
python scripts/evaluate_retrieval.py --dataset expanded

# Run with CI quality gate enforcement (Recall@5 >= 0.85, MRR@5 >= 0.80)
python scripts/evaluate_retrieval.py --ci --gate-recall 0.85 --gate-mrr 0.80

# Start the FastAPI server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend Setup
```powershell
# Navigate to frontend directory
cd D:\Project\secure_doc_intelligence\frontend

# Install dependencies (already installed)
npm install

# Build production bundle
npm run build

# Start development server
npm run dev
```

---

## 12. Deployment Considerations & Known Caveats

While designed with strict security and privacy standards, note the following operational considerations before deploying in production:
1. **JWT Secret Configuration**: Default JWT signing key uses a development secret; set `JWT_SECRET_KEY` in production environment variables.
2. **SQLite Concurrency**: SQLite with WAL mode supports high-throughput concurrent reads and single-writer transactions. For massive distributed write concurrency, migrate the authoritative schema to PostgreSQL while preserving the identical query interface.
3. **ChromaDB Scaling**: Current setup uses embedded ChromaDB with persistent local storage. For multi-node clustering, use Chroma's client/server architecture.
4. **Ollama Hardware Allocation**: Local inference latency depends directly on available GPU memory (e.g. 8GB+ VRAM for `qwen3:8b` quantization).
5. **HTTPS / Reverse Proxy**: Always place FastAPI behind an SSL-terminating reverse proxy (Nginx, Caddy, or Cloudflare Tunnel) to ensure all authentication tokens and document payloads are encrypted in transit.

---

## 13. Detailed Documentation
- [System Architecture Specification](file:///d:/Project/secure_doc_intelligence/docs/ARCHITECTURE.md)
- [Security Architecture & Authorization Guide](file:///d:/Project/secure_doc_intelligence/docs/SECURITY.md)
- [Phased Implementation Plan & Test Gates](file:///d:/Project/secure_doc_intelligence/docs/IMPLEMENTATION_PLAN.md)

