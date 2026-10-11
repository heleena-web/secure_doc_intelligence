export interface User {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  created_at: string;
}

export interface DocumentRecord {
  id: string;
  workspace_id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  status: string;
  owner_id?: string;
  is_private: boolean;
  page_count?: number;
  chunk_count: number;
  created_at: string;
}

export interface CitationItem {
  chunk_id: string;
  document_id: string;
  filename: string;
  citation_label: string;
  page_number?: number;
  heading_path?: string;
  snippet: string;
  verified: boolean;
  source_type?: string;
  source_url?: string;
}

export interface QAResponse {
  question: string;
  answer: string;
  status: 'ANSWERED' | 'INSUFFICIENT_EVIDENCE' | 'NO_EVIDENCE_FOUND';
  has_sufficient_evidence: boolean;
  citations: CitationItem[];
  supporting_snippets: string[];
  model_used: string;
  retrieved_chunks_count: number;
}

export interface SearchResultItem {
  chunk_id: string;
  document_id: string;
  filename: string;
  content: string;
  citation_label: string;
  score: number;
  bm25_rank?: number;
  semantic_rank?: number;
  rrf_score?: number;
}

export interface SearchResponse {
  query: string;
  mode: string;
  workspace_id: string;
  total_candidates: number;
  results: SearchResultItem[];
}

const API_BASE = '/api/v1';

function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem('token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export function extractErrorMessage(err: any, fallback: string): string {
  if (!err) return fallback;
  if (typeof err === 'string') return err;
  if (typeof err.detail === 'string') return err.detail;
  if (Array.isArray(err.detail)) {
    return err.detail
      .map((item: any) => {
        if (typeof item === 'string') return item;
        const field = Array.isArray(item.loc)
          ? item.loc.filter((part: any) => part !== 'body').join('.')
          : '';
        return field ? `${field}: ${item.msg}` : (item.msg || JSON.stringify(item));
      })
      .join('; ');
  }
  if (typeof err.detail === 'object' && err.detail !== null) {
    if (err.detail.message) return err.detail.message;
    if (err.detail.msg) return err.detail.msg;
    return JSON.stringify(err.detail);
  }
  if (typeof err.message === 'string') return err.message;
  return fallback;
}

export async function login(username: string, password: string): Promise<{ access_token: string; user: User }> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(err, 'Login failed'));
  }
  return res.json();
}

export async function register(username: string, email: string, password: string, full_name?: string): Promise<{ access_token: string; user: User }> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password, full_name }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(err, 'Registration failed'));
  }
  return res.json();
}

export async function getMe(): Promise<User> {
  const res = await fetch(`${API_BASE}/auth/me`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error('Unauthenticated');
  return res.json();
}

export async function listDocuments(workspaceId: string = 'default'): Promise<DocumentRecord[]> {
  const res = await fetch(`${API_BASE}/documents?workspace_id=${workspaceId}`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error('Failed to fetch documents');
  return res.json();
}

export async function ingestDocument(
  file: File,
  workspaceId: string = 'default',
  isPrivate: boolean = true
): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('workspace_id', workspaceId);
  formData.append('is_private', isPrivate.toString());

  const res = await fetch(`${API_BASE}/documents/ingest`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(err, 'Upload failed'));
  }
  return res.json();
}

export async function deleteDocument(documentId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${documentId}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(err, 'Failed to delete document'));
  }
}

export async function askQuestion(question: string, workspaceId: string = 'default'): Promise<QAResponse> {
  const res = await fetch(`${API_BASE}/search/answer`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({ question, workspace_id: workspaceId, top_k: 5 }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(err, 'Question answering failed'));
  }
  return res.json();
}

export async function executeSearch(query: string, workspaceId: string = 'default'): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({ query, workspace_id: workspaceId, mode: 'hybrid_rerank', top_k: 5 }),
  });
  if (!res.ok) throw new Error('Search failed');
  return res.json();
}

// ---------------------------------------------------------------------------
// Notion Knowledge Integration
// ---------------------------------------------------------------------------

export interface NotionConnectionResponse {
  workspace_id: string;
  status: 'connected' | 'disconnected' | 'error';
  is_connected: boolean;
  workspace_name?: string;
  bot_id?: string;
  bot_name?: string;
  last_synced_at?: string;
  page_count: number;
  has_configured_token: boolean;
}

export interface NotionPageSummary {
  page_id: string;
  title: string;
  url: string;
  last_edited_time: string;
  is_synced: boolean;
  document_id?: string;
  synced_at?: string;
  status: string;
}

export interface NotionSyncResponse {
  workspace_id: string;
  total_pages_found: number;
  pages_synced: number;
  pages_updated: number;
  pages_removed: number;
  pages_skipped: number;
  synced_at: string;
  errors: string[];
}

export async function getNotionStatus(workspaceId: string = 'default'): Promise<NotionConnectionResponse> {
  const res = await fetch(`${API_BASE}/notion/status?workspace_id=${workspaceId}`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(err, 'Failed to fetch Notion status'));
  }
  return res.json();
}

export async function connectNotion(
  apiKey?: string,
  workspaceId: string = 'default'
): Promise<NotionConnectionResponse> {
  const res = await fetch(`${API_BASE}/notion/connect`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({
      workspace_id: workspaceId,
      api_key: apiKey ? apiKey.trim() : undefined,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(err, 'Failed to connect to Notion'));
  }
  return res.json();
}

export async function disconnectNotion(workspaceId: string = 'default'): Promise<void> {
  const res = await fetch(`${API_BASE}/notion/disconnect?workspace_id=${workspaceId}`, {
    method: 'POST',
    headers: getAuthHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(err, 'Failed to disconnect Notion'));
  }
}

export async function listNotionPages(workspaceId: string = 'default'): Promise<NotionPageSummary[]> {
  const res = await fetch(`${API_BASE}/notion/pages?workspace_id=${workspaceId}`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(err, 'Failed to fetch Notion pages'));
  }
  return res.json();
}

export async function syncNotionPages(
  workspaceId: string = 'default',
  pageIds?: string[],
  isPrivate: boolean = true,
  force: boolean = false
): Promise<NotionSyncResponse> {
  const res = await fetch(`${API_BASE}/notion/sync`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({
      workspace_id: workspaceId,
      page_ids: pageIds,
      is_private: isPrivate,
      force: force,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(err, 'Notion sync failed'));
  }
  return res.json();
}
