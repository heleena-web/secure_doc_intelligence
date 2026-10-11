import React, { useState, useEffect } from 'react';
import {
  Shield,
  FileText,
  Search,
  UploadCloud,
  Lock,
  Trash2,
  CheckCircle,
  AlertCircle,
  HelpCircle,
  ChevronDown,
  ChevronRight,
  LogIn,
  LogOut,
  Sparkles,
  X,
  BookOpen,
  RefreshCw,
  ExternalLink,
  Layers,
} from 'lucide-react';
import {
  User,
  DocumentRecord,
  QAResponse,
  NotionConnectionResponse,
  NotionPageSummary,
  login,
  register,
  getMe,
  listDocuments,
  ingestDocument,
  deleteDocument,
  askQuestion,
  getNotionStatus,
  connectNotion,
  disconnectNotion,
  listNotionPages,
  syncNotionPages,
} from './api';

export function App() {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [authUsername, setAuthUsername] = useState('demo_admin');
  const [authEmail, setAuthEmail] = useState('admin@company.com');
  const [authPassword, setAuthPassword] = useState('Password123!');
  const [authError, setAuthError] = useState('');

  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isPrivateUpload, setIsPrivateUpload] = useState(true);
  const [uploadNotice, setUploadNotice] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);

  const [question, setQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [qaResult, setQaResult] = useState<QAResponse | null>(null);
  const [qaError, setQaError] = useState('');
  const [showDebug, setShowDebug] = useState(false);

  // Notion Integration State
  const [notionStatus, setNotionStatus] = useState<NotionConnectionResponse | null>(null);
  const [isSyncingNotion, setIsSyncingNotion] = useState(false);
  const [isConnectingNotion, setIsConnectingNotion] = useState(false);
  const [notionApiKey, setNotionApiKey] = useState('');
  const [showNotionConnectModal, setShowNotionConnectModal] = useState(false);
  const [notionPages, setNotionPages] = useState<NotionPageSummary[]>([]);
  const [showNotionPages, setShowNotionPages] = useState(false);
  const [isLoadingNotionPages, setIsLoadingNotionPages] = useState(false);

  // Initialize Auth & Status
  useEffect(() => {
    getMe()
      .then((user) => {
        setCurrentUser(user);
        loadDocuments();
        loadNotionStatus();
      })
      .catch(() => {
        // Attempt auto-login with default demo credentials for local developer experience
        login('test_admin', 'adminpassword123')
          .then((res) => {
            localStorage.setItem('token', res.access_token);
            setCurrentUser(res.user);
            loadDocuments();
            loadNotionStatus();
          })
          .catch(() => {
            setIsAuthOpen(true);
          });
      });
  }, []);

  const loadDocuments = async () => {
    try {
      const docs = await listDocuments('default');
      setDocuments(docs);
    } catch {
      // Ignored if unauthenticated
    }
  };

  const loadNotionStatus = async () => {
    try {
      const status = await getNotionStatus('default');
      setNotionStatus(status);
    } catch {
      // Ignored if unauthenticated or disconnected
    }
  };

  const handleConnectNotion = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsConnectingNotion(true);
    try {
      const res = await connectNotion(notionApiKey || undefined, 'default');
      setNotionStatus(res);
      setShowNotionConnectModal(false);
      setNotionApiKey('');
      setUploadNotice({
        type: 'success',
        message: `Successfully connected to Notion workspace '${res.workspace_name || 'Notion'}'.`,
      });
    } catch (err: any) {
      setUploadNotice({
        type: 'error',
        message: err.message || 'Failed to connect to Notion',
      });
    } finally {
      setIsConnectingNotion(false);
    }
  };

  const handleDisconnectNotion = async () => {
    if (!confirm('Are you sure you want to disconnect Notion for this workspace?')) return;
    try {
      await disconnectNotion('default');
      await loadNotionStatus();
      setUploadNotice({
        type: 'info',
        message: 'Notion integration disconnected.',
      });
    } catch (err: any) {
      setUploadNotice({
        type: 'error',
        message: err.message || 'Failed to disconnect Notion',
      });
    }
  };

  const handleSyncNotion = async () => {
    setIsSyncingNotion(true);
    try {
      const res = await syncNotionPages('default', undefined, isPrivateUpload, true);
      await loadDocuments();
      await loadNotionStatus();
      setUploadNotice({
        type: 'success',
        message: `Notion Sync Complete: ${res.pages_synced} synced, ${res.pages_updated} updated, ${res.pages_skipped} unchanged, ${res.pages_removed} removed.`,
      });
    } catch (err: any) {
      setUploadNotice({
        type: 'error',
        message: err.message || 'Notion synchronization failed',
      });
    } finally {
      setIsSyncingNotion(false);
    }
  };

  const handleToggleNotionPages = async () => {
    if (!showNotionPages) {
      setIsLoadingNotionPages(true);
      try {
        const pages = await listNotionPages('default');
        setNotionPages(pages);
        setShowNotionPages(true);
      } catch (err: any) {
        setUploadNotice({
          type: 'error',
          message: err.message || 'Could not discover Notion pages',
        });
      } finally {
        setIsLoadingNotionPages(false);
      }
    } else {
      setShowNotionPages(false);
    }
  };

  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError('');
    try {
      let res;
      if (authMode === 'login') {
        res = await login(authUsername, authPassword);
      } else {
        res = await register(authUsername, authEmail, authPassword);
      }
      localStorage.setItem('token', res.access_token);
      setCurrentUser(res.user);
      setIsAuthOpen(false);
      loadDocuments();
    } catch (err: any) {
      setAuthError(err.message || 'Authentication failed');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setCurrentUser(null);
    setDocuments([]);
    setQaResult(null);
    setIsAuthOpen(true);
  };

  const uploadFile = async (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!ext || !['pdf', 'docx', 'txt'].includes(ext)) {
      setUploadNotice({
        type: 'error',
        message: `Unsupported file type '.${ext || 'unknown'}'. Please upload a PDF, DOCX, or TXT file.`,
      });
      return;
    }

    if (file.size === 0) {
      setUploadNotice({
        type: 'error',
        message: `File '${file.name}' contains no readable content (0 bytes).`,
      });
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setUploadNotice({
        type: 'error',
        message: `File '${file.name}' exceeds the 50MB maximum size limit.`,
      });
      return;
    }

    setIsUploading(true);
    setUploadNotice(null);
    try {
      const summary = await ingestDocument(file, 'default', isPrivateUpload);
      await loadDocuments();
      if (summary.is_duplicate) {
        setUploadNotice({
          type: 'info',
          message: `Document '${file.name}' was already indexed in this workspace.`,
        });
      } else {
        setUploadNotice({
          type: 'success',
          message: `Successfully indexed '${file.name}' (${summary.chunk_count} chunks).`,
        });
      }
    } catch (err: any) {
      setUploadNotice({
        type: 'error',
        message: err.message || 'Upload failed',
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadFile(file);
    }
    e.target.value = '';
  };

  const handleDelete = async (docId: string, filename: string) => {
    if (!confirm(`Are you sure you want to delete '${filename}'? Chunks and search indices will be purged immediately.`)) return;
    try {
      await deleteDocument(docId);
      await loadDocuments();
      setUploadNotice({
        type: 'info',
        message: `Deleted '${filename}' and purged all indexed chunks.`,
      });
    } catch (err: any) {
      setUploadNotice({
        type: 'error',
        message: err.message || 'Failed to delete document',
      });
    }
  };

  const handleAsk = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!question.trim() || isAsking) return;

    setIsAsking(true);
    setQaError('');
    try {
      const res = await askQuestion(question.trim(), 'default');
      setQaResult(res);
    } catch (err: any) {
      setQaError(err.message || 'Error communicating with AI service');
    } finally {
      setIsAsking(false);
    }
  };

  const sampleQuestions = [
    'What are the password security requirements?',
    'What is the parental leave policy?',
    'What is the international travel expense allowance?',
  ];

  return (
    <div>
      {/* Navigation Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">
            <Shield size={22} />
          </div>
          <div>
            <div className="brand-title">Secure Document Intelligence</div>
            <div className="brand-subtitle">Privacy-First • Permission-Aware • Local Grounded AI</div>
          </div>
        </div>

        <div className="user-bar">
          {currentUser ? (
            <>
              <div className="user-badge">
                <Lock size={14} />
                <span>Workspace: default</span>
                <span style={{ color: 'var(--text-muted)' }}>|</span>
                <span>{currentUser.username}</span>
              </div>
              <button className="btn-secondary" onClick={handleLogout} title="Log Out">
                <LogOut size={14} />
                <span>Sign Out</span>
              </button>
            </>
          ) : (
            <button className="btn-secondary" onClick={() => setIsAuthOpen(true)}>
              <LogIn size={14} />
              <span>Sign In</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Container */}
      <main className="app-container">
        {/* Left Column: Document Management */}
        <section>
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <FileText size={18} />
                <span>Document Library ({documents.length})</span>
              </div>
            </div>
            <div className="card-body">
              {/* Upload Dropzone */}
              <label
                className={`dropzone ${isUploading ? 'active' : ''} ${isDragging ? 'dragging' : ''}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  const file = e.dataTransfer.files?.[0];
                  if (file && currentUser) {
                    uploadFile(file);
                  }
                }}
              >
                <input
                  type="file"
                  accept=".pdf,.docx,.txt"
                  style={{ display: 'none' }}
                  onChange={handleFileUpload}
                  disabled={isUploading || !currentUser}
                />
                <UploadCloud size={30} className="dropzone-icon" />
                <div className="dropzone-label">
                  {isUploading ? 'Ingesting, Chunking & Indexing...' : 'Click or Drag Document to Upload'}
                </div>
                <div className="dropzone-hint">Supports PDF, DOCX, and TXT files (Max 50MB)</div>
              </label>

              {/* Upload Notice Banner */}
              {uploadNotice && (
                <div className={`toast-banner toast-banner-${uploadNotice.type}`}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                    {uploadNotice.type === 'success' && <CheckCircle size={15} />}
                    {uploadNotice.type === 'error' && <AlertCircle size={15} />}
                    {uploadNotice.type === 'info' && <HelpCircle size={15} />}
                    <span>{uploadNotice.message}</span>
                  </div>
                  <button
                    type="button"
                    className="toast-close-btn"
                    onClick={() => setUploadNotice(null)}
                    title="Dismiss"
                  >
                    <X size={14} />
                  </button>
                </div>
              )}

              {/* Privacy Toggle Card */}
              <div className="privacy-toggle-card">
                <span className="privacy-toggle-title">
                  <Lock size={13} />
                  <span>Document Privacy</span>
                </span>
                <label className="privacy-toggle-input">
                  <input
                    type="checkbox"
                    checked={isPrivateUpload}
                    onChange={(e) => setIsPrivateUpload(e.target.checked)}
                  />
                  <span>Restrict to Owner</span>
                </label>
              </div>

              {/* Document List */}
              <div className="doc-list">
                {documents.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '2rem 1rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    No documents uploaded yet in this workspace.
                  </div>
                ) : (
                  documents.map((doc) => (
                    <div key={doc.id} className="doc-item">
                      <div className="doc-info">
                        <span className={`badge badge-${doc.file_type}`}>{doc.file_type}</span>
                        <div>
                          <div className="doc-name" title={doc.filename}>{doc.filename}</div>
                          <div className="doc-meta">
                            {doc.chunk_count} chunks {doc.page_count ? `• ${doc.page_count} pages` : ''}
                            {doc.is_private && ' • Private'}
                          </div>
                        </div>
                      </div>
                      <button
                        className="btn-icon-danger"
                        title="Delete Document"
                        onClick={() => handleDelete(doc.id, doc.filename)}
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Notion Knowledge Base Card */}
          <div className="card" style={{ marginTop: '1.25rem' }}>
            <div className="card-header">
              <div className="card-title">
                <BookOpen size={18} style={{ color: '#d97706' }} />
                <span>Notion Knowledge Base</span>
              </div>
              <div>
                {notionStatus?.is_connected ? (
                  <span className="notion-status-pill connected">
                    <span className="status-dot green"></span>
                    <span>Connected</span>
                  </span>
                ) : (
                  <span className="notion-status-pill disconnected">
                    <span className="status-dot gray"></span>
                    <span>Not Connected</span>
                  </span>
                )}
              </div>
            </div>
            <div className="card-body">
              {notionStatus?.is_connected ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                  <div className="notion-info-row">
                    <div>
                      <div style={{ fontSize: '0.825rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                        {notionStatus.workspace_name || 'Notion Workspace'}
                      </div>
                      <div style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>
                        Bot: {notionStatus.bot_name || notionStatus.bot_id || 'Internal Integration'} • {notionStatus.page_count} pages indexed
                      </div>
                    </div>
                    {notionStatus.last_synced_at && (
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'right' }}>
                        Last sync: {new Date(notionStatus.last_synced_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    )}
                  </div>

                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <button
                      type="button"
                      className="btn-primary"
                      style={{ padding: '0.5rem 0.85rem', fontSize: '0.8rem' }}
                      onClick={handleSyncNotion}
                      disabled={isSyncingNotion}
                    >
                      <RefreshCw size={14} className={isSyncingNotion ? 'spin' : ''} />
                      <span>{isSyncingNotion ? 'Syncing...' : 'Sync Notion'}</span>
                    </button>

                    <button
                      type="button"
                      style={{
                        padding: '0.5rem 0.85rem',
                        fontSize: '0.8rem',
                        background: 'var(--bg-surface-subtle)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.4rem',
                        color: 'var(--text-secondary)'
                      }}
                      onClick={handleToggleNotionPages}
                      disabled={isLoadingNotionPages}
                    >
                      <Layers size={14} className={isLoadingNotionPages ? 'spin' : ''} />
                      <span>{showNotionPages ? 'Hide Pages' : 'Discovered Pages'}</span>
                    </button>

                    <button
                      type="button"
                      style={{
                        padding: '0.5rem 0.85rem',
                        fontSize: '0.8rem',
                        background: 'transparent',
                        border: '1px solid var(--danger-border)',
                        color: 'var(--danger-text)',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        marginLeft: 'auto'
                      }}
                      onClick={handleDisconnectNotion}
                    >
                      Disconnect
                    </button>
                  </div>

                  {showNotionPages && (
                    <div className="notion-pages-drawer">
                      <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                        Accessible Notion Pages ({notionPages.length})
                      </div>
                      {notionPages.length === 0 ? (
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                          No shared pages found. Ensure pages are shared with your Notion integration bot.
                        </div>
                      ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '180px', overflowY: 'auto' }}>
                          {notionPages.map((page) => (
                            <div key={page.page_id} className="notion-page-item">
                              <a
                                href={page.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="notion-page-link"
                                title={page.title}
                              >
                                <ExternalLink size={11} />
                                <span>{page.title}</span>
                              </a>
                              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                                {page.last_edited_time ? new Date(page.last_edited_time).toLocaleDateString() : ''}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '1rem 0.5rem' }}>
                  <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', marginBottom: '1rem', lineHeight: 1.5 }}>
                    Connect your workspace to Notion to index internal pages and retrieve team documentation securely.
                  </p>
                  <button
                    type="button"
                    className="btn-primary"
                    style={{ margin: '0 auto', padding: '0.55rem 1rem', fontSize: '0.825rem' }}
                    onClick={() => setShowNotionConnectModal(true)}
                  >
                    <BookOpen size={15} />
                    <span>Connect Notion Integration</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Right Column: Grounded Search & Answers */}
        <section>
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <Sparkles size={18} style={{ color: 'var(--accent-primary)' }} />
                <span>Evidence-Based Document Assistant</span>
              </div>
            </div>
            <div className="card-body">
              {/* Question Input */}
              <form onSubmit={handleAsk} className="query-bar">
                <input
                  type="text"
                  className="input-text"
                  placeholder="Ask a question grounded in your internal documents..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  disabled={isAsking}
                />
                <button type="submit" className="btn-primary" disabled={isAsking || !question.trim()}>
                  <Search size={16} />
                  <span>{isAsking ? 'Thinking...' : 'Search & Answer'}</span>
                </button>
              </form>

              {/* Sample Queries */}
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', alignSelf: 'center' }}>Examples:</span>
                {sampleQuestions.map((q, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      setQuestion(q);
                    }}
                    style={{
                      background: 'var(--bg-surface-subtle)',
                      border: '1px solid var(--border-color)',
                      padding: '0.25rem 0.6rem',
                      borderRadius: '9999px',
                      fontSize: '0.75rem',
                      cursor: 'pointer',
                      color: 'var(--text-secondary)',
                    }}
                  >
                    {q}
                  </button>
                ))}
              </div>

              {/* Error Message */}
              {qaError && (
                <div style={{ padding: '0.75rem 1rem', background: 'var(--danger-bg)', color: 'var(--danger-text)', border: '1px solid var(--danger-border)', borderRadius: '8px', fontSize: '0.85rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <AlertCircle size={16} />
                  <span>{qaError}</span>
                </div>
              )}

              {/* Answer Panel */}
              {qaResult && (
                <div className="answer-panel">
                  <div className="answer-header">
                    <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                      GROUNDED RESPONSE
                    </span>
                    {qaResult.status === 'ANSWERED' && (
                      <span className="status-pill status-answered">
                        <CheckCircle size={13} />
                        <span>Evidence Verified ({qaResult.citations.length} Sources)</span>
                      </span>
                    )}
                    {qaResult.status === 'INSUFFICIENT_EVIDENCE' && (
                      <span className="status-pill status-insufficient">
                        <AlertCircle size={13} />
                        <span>Insufficient Evidence (Refusal)</span>
                      </span>
                    )}
                    {qaResult.status === 'NO_EVIDENCE_FOUND' && (
                      <span className="status-pill status-no-evidence">
                        <HelpCircle size={13} />
                        <span>No Documents Matched</span>
                      </span>
                    )}
                  </div>

                  <div className="answer-text">
                    {qaResult.answer}
                  </div>

                  {/* Verified Citations */}
                  {qaResult.citations && qaResult.citations.length > 0 && (
                    <div className="citations-section">
                      <div className="citations-title">Verified Source Citations</div>
                      <div className="citations-grid">
                        {qaResult.citations.map((c, i) => (
                          <div key={i} className="citation-card">
                            <div className="citation-header">
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                {c.source_type === 'notion' && (
                                  <span className="badge badge-notion" style={{ fontSize: '0.65rem', padding: '0.1rem 0.4rem' }}>
                                    Notion
                                  </span>
                                )}
                                <span style={{ color: 'var(--accent-primary)' }}>{c.filename}</span>
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{c.citation_label}</span>
                                {c.source_url && (
                                  <a
                                    href={c.source_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="citation-ext-link"
                                    title="Open original page in Notion"
                                  >
                                    <ExternalLink size={10} style={{ display: 'inline', marginRight: '2px', verticalAlign: '-1px' }} />
                                    Open
                                  </a>
                                )}
                              </div>
                            </div>
                            <div className="citation-snippet">{c.snippet}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Collapsible Retrieval & Debug Drawer */}
                  <div className="debug-drawer">
                    <button
                      type="button"
                      className="debug-toggle"
                      onClick={() => setShowDebug(!showDebug)}
                    >
                      {showDebug ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                      <span>{showDebug ? 'Hide' : 'Show'} Retrieval & Technical Details</span>
                    </button>
                    {showDebug && (
                      <pre className="debug-content">
                        {JSON.stringify(
                          {
                            model_used: qaResult.model_used,
                            evidence_status: qaResult.status,
                            retrieved_candidate_count: qaResult.retrieved_chunks_count,
                            verified_citations_count: qaResult.citations.length,
                            raw_supporting_snippets: qaResult.supporting_snippets,
                          },
                          null,
                          2
                        )}
                      </pre>
                    )}
                  </div>
                </div>
              )}

              {!qaResult && !isAsking && (
                <div className="assistant-empty-state">
                  <div className="assistant-empty-icon">
                    <Sparkles size={24} />
                  </div>
                  <div className="assistant-empty-title">
                    Evidence-Based Grounded Answers
                  </div>
                  <div className="assistant-empty-desc">
                    Ask questions across your ingested organization documents. Answers are generated by local Ollama AI and strictly verified against retrieved passage citations.
                  </div>
                  <div className="assistant-badge-row">
                    <div className="assistant-feature-badge">
                      <Lock size={12} style={{ color: 'var(--accent-primary)' }} />
                      <span>Pre-Retrieval ACL Guard</span>
                    </div>
                    <div className="assistant-feature-badge">
                      <Search size={12} style={{ color: 'var(--accent-primary)' }} />
                      <span>Hybrid BM25 + Vector Search</span>
                    </div>
                    <div className="assistant-feature-badge">
                      <Shield size={12} style={{ color: 'var(--accent-primary)' }} />
                      <span>100% Local (Zero Cloud Leakage)</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      </main>

      {/* Authentication Modal */}
      {isAuthOpen && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50 }}>
          <div style={{ background: 'white', padding: '2rem', borderRadius: '12px', width: '100%', maxWidth: '380px', boxShadow: 'var(--shadow-lg)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
              <Shield size={24} style={{ color: 'var(--accent-primary)' }} />
              <h2 style={{ fontSize: '1.15rem', fontWeight: 700 }}>
                {authMode === 'login' ? 'Sign In to Workspace' : 'Create Account'}
              </h2>
            </div>

            {authError && (
              <div style={{ padding: '0.5rem 0.75rem', background: 'var(--danger-bg)', color: 'var(--danger-text)', borderRadius: '6px', fontSize: '0.8rem', marginBottom: '1rem' }}>
                {authError}
              </div>
            )}

            <form onSubmit={handleAuthSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Username</label>
                <input
                  type="text"
                  className="input-text"
                  style={{ width: '100%', marginTop: '0.25rem' }}
                  value={authUsername}
                  onChange={(e) => setAuthUsername(e.target.value)}
                  minLength={3}
                  maxLength={50}
                  required
                />
              </div>

              {authMode === 'register' && (
                <div>
                  <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Email</label>
                  <input
                    type="email"
                    className="input-text"
                    style={{ width: '100%', marginTop: '0.25rem' }}
                    value={authEmail}
                    onChange={(e) => setAuthEmail(e.target.value)}
                    required
                  />
                </div>
              )}

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Password</label>
                  {authMode === 'register' && (
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Min 8 characters</span>
                  )}
                </div>
                <input
                  type="password"
                  className="input-text"
                  style={{ width: '100%', marginTop: '0.25rem' }}
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  minLength={authMode === 'register' ? 8 : 1}
                  required
                />
              </div>

              <button type="submit" className="btn-primary" style={{ justifyContent: 'center', marginTop: '0.5rem' }}>
                {authMode === 'login' ? 'Authenticate' : 'Register Account'}
              </button>
            </form>

            <div style={{ textAlign: 'center', marginTop: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {authMode === 'login' ? (
                <span>
                  Don't have an account?{' '}
                  <button type="button" onClick={() => { setAuthMode('register'); setAuthError(''); }} style={{ color: 'var(--accent-primary)', border: 'none', background: 'none', cursor: 'pointer', fontWeight: 600 }}>
                    Register
                  </button>
                </span>
              ) : (
                <span>
                  Already registered?{' '}
                  <button type="button" onClick={() => { setAuthMode('login'); setAuthError(''); }} style={{ color: 'var(--accent-primary)', border: 'none', background: 'none', cursor: 'pointer', fontWeight: 600 }}>
                    Sign In
                  </button>
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Notion Connect Modal */}
      {showNotionConnectModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50 }}>
          <div style={{ background: 'white', padding: '2rem', borderRadius: '12px', width: '100%', maxWidth: '440px', boxShadow: 'var(--shadow-lg)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <BookOpen size={20} style={{ color: '#d97706' }} />
                <h2 style={{ fontSize: '1.15rem', fontWeight: 700 }}>Connect Notion Workspace</h2>
              </div>
              <button
                type="button"
                onClick={() => setShowNotionConnectModal(false)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
              >
                <X size={18} />
              </button>
            </div>

            <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', marginBottom: '1.25rem', lineHeight: 1.5 }}>
              Enter an Internal Integration Secret from your Notion developer portal, or leave blank to use the server-configured environment secret.
            </p>

            <form onSubmit={handleConnectNotion} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  Internal Integration Secret (Optional)
                </label>
                <input
                  type="password"
                  className="input-text"
                  placeholder="secret_... (or leave empty if configured on server)"
                  style={{ width: '100%', marginTop: '0.35rem' }}
                  value={notionApiKey}
                  onChange={(e) => setNotionApiKey(e.target.value)}
                />
              </div>

              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', background: 'var(--bg-surface-subtle)', padding: '0.6rem 0.75rem', borderRadius: '6px' }}>
                Tip: Remember to open the Notion pages you wish to sync and share them with your internal integration bot via the page's "..." menu.
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', marginTop: '0.5rem' }}>
                <button
                  type="button"
                  style={{
                    padding: '0.6rem 1rem',
                    background: 'var(--bg-surface-subtle)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '8px',
                    fontSize: '0.85rem',
                    cursor: 'pointer'
                  }}
                  onClick={() => setShowNotionConnectModal(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={isConnectingNotion}
                  style={{ padding: '0.6rem 1.25rem' }}
                >
                  {isConnectingNotion ? 'Connecting...' : 'Connect Workspace'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

