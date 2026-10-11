import pytest
from app.models.document import ChunkRecord
from app.services.retrieval.vector_store import ChromaVectorStore


def test_chroma_semantic_retrieval(tmp_path):
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma")

    chunk1 = ChunkRecord(
        id="c-sem-1",
        document_id="doc-cloud",
        workspace_id="default",
        chunk_index=0,
        char_start=0,
        char_end=100,
        token_count=20,
        content="Virtual private clouds must enforce zero trust network segmentation and microperimeters.",
        citation_label="[Doc: Cloud_Architecture.pdf, Page: 12]",
        metadata={"pages": [12]},
        content_hash="h1" * 32,
    )

    chunk2 = ChunkRecord(
        id="c-sem-2",
        document_id="doc-recipes",
        workspace_id="default",
        chunk_index=0,
        char_start=0,
        char_end=80,
        token_count=15,
        content="Bake the chocolate cake at 350 degrees for twenty-five minutes.",
        citation_label="[Doc: Cookbook.txt, Lines: 10-14]",
        metadata={"line_ranges": [(10, 14)]},
        content_hash="h2" * 32,
    )

    store.add_chunks([chunk1, chunk2], filename="Cloud_Architecture.pdf")

    # Semantic query with distinct vocabulary from chunk1
    results = store.search("secure cloud networking boundaries", workspace_id="default", top_k=5)
    assert len(results) > 0
    assert results[0].chunk_id == "c-sem-1"
    assert results[0].score > 0.4
    assert results[0].semantic_rank == 1
    assert results[0].page_number == 12


def test_chroma_permission_and_workspace_filtering(tmp_path):
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma_perm")

    chunk_a = ChunkRecord(
        id="c-doc-a",
        document_id="doc-a",
        workspace_id="ws-main",
        chunk_index=0,
        char_start=0,
        char_end=50,
        token_count=10,
        content="Audit logs are retained for seven years.",
        citation_label="[Doc: Audit.pdf]",
        metadata={},
        content_hash="ha" * 32,
    )
    chunk_b = ChunkRecord(
        id="c-doc-b",
        document_id="doc-b",
        workspace_id="ws-main",
        chunk_index=0,
        char_start=0,
        char_end=50,
        token_count=10,
        content="Secret revenue projection figures.",
        citation_label="[Doc: Revenue.pdf]",
        metadata={},
        content_hash="hb" * 32,
    )

    store.add_chunks([chunk_a, chunk_b])

    # 1. Workspace isolation
    res_other_ws = store.search("audit logs", workspace_id="ws-other")
    assert len(res_other_ws) == 0

    # 2. Permitted doc filtering: user only has permission for doc-a
    res_perm_a = store.search(
        "financial revenue and audit logs",
        workspace_id="ws-main",
        permitted_doc_ids=["doc-a"],
    )
    assert len(res_perm_a) == 1
    assert res_perm_a[0].chunk_id == "c-doc-a"

    # 3. Empty permitted doc list returns 0 candidates immediately
    res_empty_perm = store.search(
        "audit logs",
        workspace_id="ws-main",
        permitted_doc_ids=[],
    )
    assert res_empty_perm == []


def test_chroma_stale_chunks_deletion(tmp_path):
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma_stale")

    chunk = ChunkRecord(
        id="c-ephemeral",
        document_id="doc-ephemeral",
        workspace_id="default",
        chunk_index=0,
        char_start=0,
        char_end=50,
        token_count=10,
        content="Temporary server credentials rotated daily.",
        citation_label="[Doc: Temp.txt]",
        metadata={},
        content_hash="he" * 32,
    )

    store.add_chunks([chunk])
    assert len(store.search("server credentials", workspace_id="default")) == 1

    # Delete document
    store.delete_document("doc-ephemeral")

    # Assert vector index has 0 stale results
    assert len(store.search("server credentials", workspace_id="default")) == 0
