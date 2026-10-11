# Security Architecture & Authorization Specification

This document details the security model, authentication flows, authorization policies, pre-retrieval access control, and deployment assumptions for the **Secure Document Intelligence** platform.

---

## 1. Authentication Flow

Authentication is designed to be modern, local-first, and robust against credential exposure:

```
+---------------+           +------------------------+           +----------------------+
|  HTTP Client  | --------> |  POST /auth/register   | --------> | SQLite `users` Table |
|               |           |  Argon2id Hash Gen     |           | Stores: $argon2id$...|
+---------------+           +------------------------+           +----------------------+
        |
        | Credentials (Username/Email + Password)
        v
+------------------------+           +------------------------+
|    POST /auth/login    | --------> |  Argon2id Verification |
|                        |           |  Issue Signed JWT      |
+------------------------+           +------------------------+
        |
        | Authorization: Bearer <JWT>
        v
+-------------------------------------------------------------+
| Protected Endpoints (Ingest, Search, QA, Document Access)   |
| Dependencies (`app.api.deps.get_current_user`):             |
| 1. Decodes JWT using HMAC-SHA256 and validates expiration.  |
| 2. Resolves user UUID from token subject claim.             |
| 3. Confirms user status is active in SQLite.                |
+-------------------------------------------------------------+
```

### Key Implementation Details:
- **Password Hashing**: Implemented via modern **Argon2id** (`argon2-cffi`), configuring memory-hard parameters resistant to GPU/ASIC cracking. Plaintext passwords are never logged, cached, or persisted.
- **Session Tokens**: Cryptographically signed JSON Web Tokens (JWT) using `HS256`, configured via `JWT_SECRET_KEY` and expiring in 24 hours.
- **Safe Database Access**: Parameterized queries across all SQLite interactions eliminate SQL injection vulnerabilities.
- **Error Neutrality**: Generic invalid credential responses prevent account enumeration.

---

## 2. Multi-Tenancy & Workspace Authorization

The platform supports multi-tenant workspace isolation:
1. **Workspace Tenancy**:
   - Every workspace is tracked in `workspaces`.
   - Users are enrolled in workspaces via `workspace_members` with roles: `owner`, `admin`, `member`, or `viewer`.
   - Access to any workspace requires verified membership. Modifying `workspace_id` in a request to an unauthorized workspace results in an immediate **HTTP 403 Forbidden** (`WorkspaceAccessDeniedError`).
2. **Role Hierarchy**:
   - **Owner / Admin**: Full administrative rights within the workspace, including member management, viewing all workspace documents, and document deletion.
   - **Member**: Can upload documents, search authorized documents, and receive grounded answers. Cannot elevate roles or add workspace members.
   - **Viewer**: Read-only access to authorized workspace documents.

---

## 3. Document Access & Privacy Policies

The platform enforces **least-privilege defaults** for all ingested documents:

### Classification:
- **Private Documents (`is_private = true`)** *(Default)*:
  - Accessible strictly by:
    1. The document creator/owner (`owner_id`).
    2. Workspace `owner` or `admin`.
    3. Users explicitly granted permissions in `document_permissions`.
  - Hidden from all other members of the workspace.
- **Workspace-Shared Documents (`is_private = false`)**:
  - Accessible by all verified members of the workspace.

### Explicit Document Permissions:
Document owners and workspace administrators can grant granular permissions via `POST /api/v1/documents/{document_id}/permissions`:
- **Read**: Allows keyword search, semantic search, snippet inspection, and question answering.
- **Write**: Allows updating document content.
- **Admin**: Allows managing document permissions and deleting the document.

---

## 4. Pre-Retrieval Access Control (Zero-Leakage Guarantee)

Access control is enforced **at the server layer before document chunks are retrieved, reranked, or presented to the LLM**. We do not rely on frontend filtering, post-retrieval stripping, or LLM system prompt instructions.

```
                    +------------------------------------+
                    | Authenticated Request (User + WS)  |
                    +-----------------+------------------+
                                      |
                                      v
          +-------------------------------------------------------+
          | SQLite Authoritative Pre-Retrieval ACL Resolution     |
          | `DocumentRepository.get_user_permitted_document_ids`  |
          +---------------------------+---------------------------+
                                      |
                           Permitted Document IDs
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
+-----------------------------------+     +-----------------------------------+
|      BM25 Keyword Retriever       |     |        ChromaDB Vector Store      |
| Chunk filter:                     |     | Pre-retrieval metadata filter:    |
| chunk.doc_id IN :permitted_doc_ids|     | doc_id IN :permitted_doc_ids      |
+-----------------+-----------------+     +-----------------+-----------------+
                  |                                         |
                  +--------------------+--------------------+
                                       |
                                       v
                     +-----------------------------------+
                     | Reciprocal Rank Fusion (RRF)      |
                     | Fuses only authorized candidates  |
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     | FlashRank Cross-Encoder Reranker  |
                     | Evaluates only authorized chunks  |
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     | Local Ollama LLM Inference        |
                     | Zero unauthorized tokens in context|
                     +-----------------------------------+
```

### Verification:
Negative test suites verify that queries by unauthorized users:
1. Return **0 search results**.
2. Pass **0 candidate passages** to the cross-encoder reranker.
3. Pass **0 unpermitted tokens** to the Ollama prompt context.

---

## 5. Permission Revocation & Data Lifecycle

- **Immediate Revocation**: When permission is revoked via `DELETE /api/v1/documents/{id}/permissions/{user_id}`, the change is committed to SQLite immediately. The user's next search or QA request re-evaluates `get_user_permitted_document_ids` against SQLite; the revoked document is excluded with zero cache latency.
- **Cascading Deletion**: Deleting a document removes its record from SQLite and triggers cascading deletion of its chunk records and permissions. Synchronously, `search_service.remove_document(doc_id)` purges its entries from BM25 and ChromaDB, preventing stale results.

---

## 6. Known Limitations & Deployment Assumptions

> [!WARNING]
> While this system enforces strict server-side pre-retrieval authorization and passes all automated security tests, it should **not** be considered fully hardened for production deployment without addressing the following operational factors:

1. **Secret Management**: In this local development phase, the JWT secret key is read from configuration with a local default. In production, `JWT_SECRET_KEY` must be injected via a secure secrets vault or environment variable.
2. **Transport Security**: All production traffic must terminate behind HTTPS/TLS 1.3 to prevent token interception over plaintext HTTP.
3. **Brute-Force & Rate Limiting**: The current authentication endpoints do not implement account lockout or rate limiting (e.g., via Redis or token-bucket middleware).
4. **Local Single-Node Indices**: The BM25 index and ChromaDB collection reside on the local filesystem. A multi-instance cluster deployment would require distributed vector/search storage and cache invalidation messaging.
