from dataclasses import dataclass
from typing import Any, Optional
import time

from app.models.document import ChunkRecord
from app.models.search import SearchMode, SearchQuery, SearchResponse
from app.services.evaluation.quality_assurance import (
    compute_answer_relevance,
    compute_citation_precision,
    compute_context_precision,
    compute_lexical_faithfulness,
)
from app.services.retrieval.bm25_indexer import BM25Retriever
from app.services.retrieval.reranker import FlashRankReranker
from app.services.retrieval.search_service import SearchService
from app.services.retrieval.vector_store import ChromaVectorStore


@dataclass
class EvaluationTestCase:
    """Represents a benchmark query with ground truth relevant chunk IDs and reference answer."""

    query: str
    target_chunk_ids: list[str]
    category: str  # e.g. "exact_code", "semantic", "numerical", "multi_hop", "unanswerable"
    description: str
    reference_answer: Optional[str] = None


@dataclass
class BenchmarkMetricResult:
    """Measured benchmark performance metrics for a retrieval mode."""

    mode: str
    recall_at_5: float
    mrr_at_5: float
    precision_at_5: float
    avg_latency_ms: float
    total_queries: int
    context_precision_at_5: float = 0.0


def get_reproducible_evaluation_dataset(dataset_size: str = "standard") -> tuple[list[ChunkRecord], list[EvaluationTestCase]]:
    """Returns a reproducible evaluation collection across PDF, DOCX, and TXT documents.
    Supports 'standard' (6 chunks, 8 queries) and 'expanded' (24 chunks, 20 queries).
    """
    if dataset_size == "expanded":
        return get_expanded_evaluation_dataset()

    chunks = [
        # --- Document 1: Security & Compliance (PDF) ---
        ChunkRecord(
            id="eval-sec-001",
            document_id="doc-security-pdf",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=160,
            token_count=35,
            content="Policy SEC-402: All remote administrative access requires hardware-backed FIDO2 multi-factor authentication tokens. SMS and email codes are prohibited.",
            citation_label="[Doc: Security_Compliance.pdf, Page: 4]",
            metadata={"pages": [4]},
            content_hash="h1" * 32,
        ),
        ChunkRecord(
            id="eval-sec-002",
            document_id="doc-security-pdf",
            workspace_id="default",
            chunk_index=1,
            char_start=161,
            char_end=320,
            token_count=34,
            content="Policy SEC-405: Encryption keys must be rotated at least every 90 days. Backups must be encrypted at rest using AES-256-GCM algorithms.",
            citation_label="[Doc: Security_Compliance.pdf, Page: 8]",
            metadata={"pages": [8]},
            content_hash="h2" * 32,
        ),
        # --- Document 2: Employee Handbook & Benefits (DOCX) ---
        ChunkRecord(
            id="eval-hr-001",
            document_id="doc-hr-docx",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=175,
            token_count=38,
            content="Section 2.1 Benefits: The company provides 16 weeks of fully paid parental leave for primary caregivers, applicable for birth, adoption, or foster placement.",
            citation_label="[Doc: Employee_Handbook.docx, Section: 2.1 Benefits, Para: 3]",
            metadata={"headings": ["2.1 Benefits"], "paragraphs": [3]},
            content_hash="h3" * 32,
        ),
        ChunkRecord(
            id="eval-hr-002",
            document_id="doc-hr-docx",
            workspace_id="default",
            chunk_index=1,
            char_start=176,
            char_end=330,
            token_count=32,
            content="Section 2.4 Retirement: 401(k) matching is provided dollar-for-dollar up to 6% of base salary, vesting immediately upon enrollment.",
            citation_label="[Doc: Employee_Handbook.docx, Section: 2.4 Retirement, Para: 1]",
            metadata={"headings": ["2.4 Retirement"], "paragraphs": [1]},
            content_hash="h4" * 32,
        ),
        # --- Document 3: Cloud Infrastructure Guide (TXT) ---
        ChunkRecord(
            id="eval-cloud-001",
            document_id="doc-cloud-txt",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=150,
            token_count=30,
            content="Network Architecture: Internal VPC peering operates on CIDR range 10.240.0.0/16 with strict security group egress filtering enabled.",
            citation_label="[Doc: Cloud_Architecture.txt, Lines: 12-25]",
            metadata={"line_start": 12, "line_end": 25},
            content_hash="h5" * 32,
        ),
        ChunkRecord(
            id="eval-cloud-002",
            document_id="doc-cloud-txt",
            workspace_id="default",
            chunk_index=1,
            char_start=151,
            char_end=310,
            token_count=33,
            content="Incident Escalation SLA: Tier-1 production outages require an incident commander assigned within 15 minutes and stakeholder briefing within 30 minutes.",
            citation_label="[Doc: Cloud_Architecture.txt, Lines: 45-60]",
            metadata={"line_start": 45, "line_end": 60},
            content_hash="h6" * 32,
        ),
    ]

    queries = [
        # 1. Exact Policy Code Matching (tests BM25 lexical precision)
        EvaluationTestCase(
            query="What are the MFA requirements under policy SEC-402?",
            target_chunk_ids=["eval-sec-001"],
            category="exact_code",
            description="Exact policy code SEC-402 lookup",
        ),
        # 2. Key Rotation Interval (tests lexical + semantic retrieval)
        EvaluationTestCase(
            query="How frequently must encryption keys be rotated under policy SEC-405?",
            target_chunk_ids=["eval-sec-002"],
            category="exact_code",
            description="Exact encryption key rotation policy",
        ),
        # 3. Semantic Query with Vocabulary Mismatch (tests dense embedding strengths)
        EvaluationTestCase(
            query="What time off is granted to new mothers or fathers welcoming a newborn?",
            target_chunk_ids=["eval-hr-001"],
            category="semantic",
            description="Parental leave query with vocabulary variation",
        ),
        # 4. Numerical / Percentage Benefits Lookup (tests reranking disambiguation)
        EvaluationTestCase(
            query="What is the maximum company 401k matching percentage?",
            target_chunk_ids=["eval-hr-002"],
            category="numerical",
            description="401k match percentage query",
        ),
        # 5. Technical Network Parameter (tests hybrid lexical/semantic precision)
        EvaluationTestCase(
            query="What is the internal VPC peering CIDR block?",
            target_chunk_ids=["eval-cloud-001"],
            category="technical_param",
            description="VPC CIDR network specification",
        ),
        # 6. Incident Response Timeframe (tests hybrid + rerank SLA retrieval)
        EvaluationTestCase(
            query="How fast must an incident commander be assigned during a tier-1 outage?",
            target_chunk_ids=["eval-cloud-002"],
            category="sla_timeframe",
            description="Incident commander assignment SLA",
        ),
        # 7. Unanswerable Query (tests retrieval absence / thresholding)
        EvaluationTestCase(
            query="What is the meal reimbursement allowance for international business trips?",
            target_chunk_ids=[],
            category="unanswerable",
            description="Out-of-domain unanswerable question",
        ),
        # 8. Another Unanswerable Query
        EvaluationTestCase(
            query="What are the guidelines for bringing emotional support pets into the office?",
            target_chunk_ids=[],
            category="unanswerable",
            description="Unindexed workplace pet policy question",
            reference_answer="The available documents do not contain sufficient evidence to answer this question.",
        ),
    ]

    return chunks, queries


def get_expanded_evaluation_dataset() -> tuple[list[ChunkRecord], list[EvaluationTestCase]]:
    """Returns an enterprise-grade 24-chunk, 6-document, 20-query evaluation collection
    spanning PDF, DOCX, TXT, and Notion documents with authoritative reference answers.
    """
    chunks = [
        # --- 1. Security & Compliance (PDF) ---
        ChunkRecord(
            id="eval-sec-001",
            document_id="doc-security-pdf",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=160,
            token_count=35,
            content="Policy SEC-402: All remote administrative access requires hardware-backed FIDO2 multi-factor authentication tokens. SMS and email codes are prohibited.",
            citation_label="[Doc: Security_Compliance.pdf, Page: 4]",
            metadata={"pages": [4]},
            content_hash="h1" * 32,
        ),
        ChunkRecord(
            id="eval-sec-002",
            document_id="doc-security-pdf",
            workspace_id="default",
            chunk_index=1,
            char_start=161,
            char_end=320,
            token_count=34,
            content="Policy SEC-405: Encryption keys must be rotated at least every 90 days. Backups must be encrypted at rest using AES-256-GCM algorithms.",
            citation_label="[Doc: Security_Compliance.pdf, Page: 8]",
            metadata={"pages": [8]},
            content_hash="h2" * 32,
        ),
        ChunkRecord(
            id="eval-sec-003",
            document_id="doc-security-pdf",
            workspace_id="default",
            chunk_index=2,
            char_start=321,
            char_end=490,
            token_count=36,
            content="Policy SEC-410: Security audit logs must be retained in immutable cold storage for a minimum of 365 days. Direct log deletion or alteration is strictly prevented.",
            citation_label="[Doc: Security_Compliance.pdf, Page: 12]",
            metadata={"pages": [12]},
            content_hash="h3" * 32,
        ),
        ChunkRecord(
            id="eval-sec-004",
            document_id="doc-security-pdf",
            workspace_id="default",
            chunk_index=3,
            char_start=491,
            char_end=650,
            token_count=33,
            content="Policy SEC-415: Critical vulnerability disclosures require remediation patching deployed within 24 hours of public CVE publication.",
            citation_label="[Doc: Security_Compliance.pdf, Page: 15]",
            metadata={"pages": [15]},
            content_hash="h4" * 32,
        ),
        ChunkRecord(
            id="eval-sec-005",
            document_id="doc-security-pdf",
            workspace_id="default",
            chunk_index=4,
            char_start=651,
            char_end=820,
            token_count=35,
            content="Policy SEC-420: Inactive user web portal sessions must automatically terminate after 15 minutes of inactivity. Concurrent logins from disparate geo-locations are blocked.",
            citation_label="[Doc: Security_Compliance.pdf, Page: 19]",
            metadata={"pages": [19]},
            content_hash="h5" * 32,
        ),
        # --- 2. Employee Handbook & Benefits (DOCX) ---
        ChunkRecord(
            id="eval-hr-001",
            document_id="doc-hr-docx",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=175,
            token_count=38,
            content="Section 2.1 Benefits: The company provides 16 weeks of fully paid parental leave for primary caregivers, applicable for birth, adoption, or foster placement.",
            citation_label="[Doc: Employee_Handbook.docx, Section: 2.1 Benefits, Para: 3]",
            metadata={"headings": ["2.1 Benefits"], "paragraphs": [3]},
            content_hash="h6" * 32,
        ),
        ChunkRecord(
            id="eval-hr-002",
            document_id="doc-hr-docx",
            workspace_id="default",
            chunk_index=1,
            char_start=176,
            char_end=330,
            token_count=32,
            content="Section 2.4 Retirement: 401(k) matching is provided dollar-for-dollar up to 6% of base salary, vesting immediately upon enrollment.",
            citation_label="[Doc: Employee_Handbook.docx, Section: 2.4 Retirement, Para: 1]",
            metadata={"headings": ["2.4 Retirement"], "paragraphs": [1]},
            content_hash="h7" * 32,
        ),
        ChunkRecord(
            id="eval-hr-003",
            document_id="doc-hr-docx",
            workspace_id="default",
            chunk_index=2,
            char_start=331,
            char_end=510,
            token_count=36,
            content="Section 2.6 Remote Stipend: Full-time remote employees receive a one-time home office setup stipend of $1,000 and a monthly internet allowance of $75.",
            citation_label="[Doc: Employee_Handbook.docx, Section: 2.6 Remote Stipend, Para: 2]",
            metadata={"headings": ["2.6 Remote Stipend"], "paragraphs": [2]},
            content_hash="h8" * 32,
        ),
        ChunkRecord(
            id="eval-hr-004",
            document_id="doc-hr-docx",
            workspace_id="default",
            chunk_index=3,
            char_start=511,
            char_end=690,
            token_count=38,
            content="Section 2.8 Tuition Assistance: Accredited degree and professional certification coursework is reimbursed up to $5,250 annually subject to manager pre-approval and grade B or higher.",
            citation_label="[Doc: Employee_Handbook.docx, Section: 2.8 Tuition Assistance, Para: 1]",
            metadata={"headings": ["2.8 Tuition Assistance"], "paragraphs": [1]},
            content_hash="h9" * 32,
        ),
        ChunkRecord(
            id="eval-hr-005",
            document_id="doc-hr-docx",
            workspace_id="default",
            chunk_index=4,
            char_start=691,
            char_end=870,
            token_count=37,
            content="Section 3.1 Code of Conduct: Outside consulting or commercial software development requires written disclosure and pre-clearance from the ethics committee to prevent IP conflicts.",
            citation_label="[Doc: Employee_Handbook.docx, Section: 3.1 Code of Conduct, Para: 4]",
            metadata={"headings": ["3.1 Code of Conduct"], "paragraphs": [4]},
            content_hash="h10" * 32,
        ),
        # --- 3. Cloud Infrastructure Guide (TXT) ---
        ChunkRecord(
            id="eval-cloud-001",
            document_id="doc-cloud-txt",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=150,
            token_count=30,
            content="Network Architecture: Internal VPC peering operates on CIDR range 10.240.0.0/16 with strict security group egress filtering enabled.",
            citation_label="[Doc: Cloud_Architecture.txt, Lines: 12-25]",
            metadata={"line_start": 12, "line_end": 25},
            content_hash="h11" * 32,
        ),
        ChunkRecord(
            id="eval-cloud-002",
            document_id="doc-cloud-txt",
            workspace_id="default",
            chunk_index=1,
            char_start=151,
            char_end=310,
            token_count=33,
            content="Incident Escalation SLA: Tier-1 production outages require an incident commander assigned within 15 minutes and stakeholder briefing within 30 minutes.",
            citation_label="[Doc: Cloud_Architecture.txt, Lines: 45-60]",
            metadata={"line_start": 45, "line_end": 60},
            content_hash="h12" * 32,
        ),
        ChunkRecord(
            id="eval-cloud-003",
            document_id="doc-cloud-txt",
            workspace_id="default",
            chunk_index=2,
            char_start=311,
            char_end=495,
            token_count=37,
            content="Kubernetes Compute Auto-scaling: Horizontal Pod Autoscaler (HPA) triggers pod scaling when average CPU utilization exceeds 70% or memory utilization exceeds 80% over a 3-minute window.",
            citation_label="[Doc: Cloud_Architecture.txt, Lines: 75-90]",
            metadata={"line_start": 75, "line_end": 90},
            content_hash="h13" * 32,
        ),
        ChunkRecord(
            id="eval-cloud-004",
            document_id="doc-cloud-txt",
            workspace_id="default",
            chunk_index=3,
            char_start=496,
            char_end=670,
            token_count=34,
            content="Disaster Recovery Objectives: Tier-0 mission critical services target a Recovery Point Objective (RPO) under 5 minutes and Recovery Time Objective (RTO) under 30 minutes.",
            citation_label="[Doc: Cloud_Architecture.txt, Lines: 110-125]",
            metadata={"line_start": 110, "line_end": 125},
            content_hash="h14" * 32,
        ),
        # --- 4. Financial Policies (PDF) ---
        ChunkRecord(
            id="eval-fin-001",
            document_id="doc-finance-pdf",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=175,
            token_count=36,
            content="Travel Expense Per Diem: Domestic business travel meal reimbursement is capped at $75 per day without alcohol, requiring itemized receipts for any transaction over $25.",
            citation_label="[Doc: Financial_Policies.pdf, Page: 2]",
            metadata={"pages": [2]},
            content_hash="h15" * 32,
        ),
        ChunkRecord(
            id="eval-fin-002",
            document_id="doc-finance-pdf",
            workspace_id="default",
            chunk_index=1,
            char_start=176,
            char_end=350,
            token_count=35,
            content="Procurement Authorization Matrix: Purchase orders between $10,000 and $50,000 require Department Director approval; commitments exceeding $50,000 require Chief Financial Officer sign-off.",
            citation_label="[Doc: Financial_Policies.pdf, Page: 6]",
            metadata={"pages": [6]},
            content_hash="h16" * 32,
        ),
        ChunkRecord(
            id="eval-fin-003",
            document_id="doc-finance-pdf",
            workspace_id="default",
            chunk_index=2,
            char_start=351,
            char_end=520,
            token_count=34,
            content="Vehicle Mileage Reimbursement: Personal vehicle business travel is reimbursed at $0.67 per mile based on documented odometer readings or standard Google Maps transit logs.",
            citation_label="[Doc: Financial_Policies.pdf, Page: 9]",
            metadata={"pages": [9]},
            content_hash="h17" * 32,
        ),
        # --- 5. Engineering Runbook (DOCX) ---
        ChunkRecord(
            id="eval-runbook-001",
            document_id="doc-runbook-docx",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=170,
            token_count=33,
            content="Database Migration Rollback: In case of migration failure during schema upgrade, execute 'alembic downgrade -1' and notify the database on-call engineer within 5 minutes.",
            citation_label="[Doc: Engineering_Runbook.docx, Section: 1.2 Migration Rollbacks, Para: 2]",
            metadata={"headings": ["1.2 Migration Rollbacks"], "paragraphs": [2]},
            content_hash="h18" * 32,
        ),
        ChunkRecord(
            id="eval-runbook-002",
            document_id="doc-runbook-docx",
            workspace_id="default",
            chunk_index=1,
            char_start=171,
            char_end=345,
            token_count=35,
            content="Redis Cache Policy: Production caches utilize volatile-lru eviction with a maximum memory threshold of 32GB; cache keys must include an explicit TTL not exceeding 24 hours.",
            citation_label="[Doc: Engineering_Runbook.docx, Section: 2.3 Cache Eviction, Para: 1]",
            metadata={"headings": ["2.3 Cache Eviction"], "paragraphs": [1]},
            content_hash="h19" * 32,
        ),
        ChunkRecord(
            id="eval-runbook-003",
            document_id="doc-runbook-docx",
            workspace_id="default",
            chunk_index=2,
            char_start=346,
            char_end=530,
            token_count=38,
            content="API Rate Limiting: External public endpoints enforce a rate limit of 100 requests per minute per API token, returning HTTP 429 with 'Retry-After' and 'X-RateLimit-Remaining' response headers.",
            citation_label="[Doc: Engineering_Runbook.docx, Section: 3.5 API Rate Limiting, Para: 3]",
            metadata={"headings": ["3.5 API Rate Limiting"], "paragraphs": [3]},
            content_hash="h20" * 32,
        ),
        ChunkRecord(
            id="eval-runbook-004",
            document_id="doc-runbook-docx",
            workspace_id="default",
            chunk_index=3,
            char_start=531,
            char_end=710,
            token_count=36,
            content="TLS Certificate Renewal: Automated certbot renewals trigger 30 days prior to expiry; manual certificates must be uploaded to AWS Certificate Manager with at least 2048-bit RSA keys.",
            citation_label="[Doc: Engineering_Runbook.docx, Section: 4.1 Certificate Rotation, Para: 2]",
            metadata={"headings": ["4.1 Certificate Rotation"], "paragraphs": [2]},
            content_hash="h21" * 32,
        ),
        # --- 6. Notion Knowledge Base (Notion) ---
        ChunkRecord(
            id="eval-notion-001",
            document_id="doc-notion-team",
            workspace_id="default",
            chunk_index=0,
            char_start=0,
            char_end=190,
            token_count=38,
            content="[Notion: Engineering Operating Cadence, Section: Sprints] Bi-weekly sprints begin with sprint planning on alternate Mondays at 10:00 AM UTC and conclude with demo retrospectives on alternate Fridays at 3:00 PM UTC.",
            citation_label="[Notion: Engineering Operating Cadence, Section: Sprints]",
            metadata={"source_type": "notion", "url": "https://notion.so/sprints"},
            content_hash="h22" * 32,
        ),
        ChunkRecord(
            id="eval-notion-002",
            document_id="doc-notion-team",
            workspace_id="default",
            chunk_index=1,
            char_start=191,
            char_end=380,
            token_count=37,
            content="[Notion: Engineering Operating Cadence, Section: Post-Mortems] All Severity-1 incidents mandate a blameless post-mortem document completed within 72 hours, published in the Engineering Wiki with actionable preventive Jira tickets.",
            citation_label="[Notion: Engineering Operating Cadence, Section: Post-Mortems]",
            metadata={"source_type": "notion", "url": "https://notion.so/postmortems"},
            content_hash="h23" * 32,
        ),
        ChunkRecord(
            id="eval-notion-003",
            document_id="doc-notion-team",
            workspace_id="default",
            chunk_index=2,
            char_start=381,
            char_end=560,
            token_count=35,
            content="[Notion: Engineering Operating Cadence, Section: Onboarding] New engineers must complete 1Password provisioning, sign GitHub CLA, and execute a verified staging deployment within their first 14 days.",
            citation_label="[Notion: Engineering Operating Cadence, Section: Onboarding]",
            metadata={"source_type": "notion", "url": "https://notion.so/onboarding"},
            content_hash="h24" * 32,
        ),
    ]

    queries = [
        # 1. Exact Policy Code Matching
        EvaluationTestCase(
            query="What are the MFA requirements under policy SEC-402?",
            target_chunk_ids=["eval-sec-001"],
            category="exact_code",
            description="Exact policy code SEC-402 lookup",
            reference_answer="Under policy SEC-402, all remote administrative access requires hardware-backed FIDO2 multi-factor authentication tokens. SMS and email codes are prohibited.",
        ),
        # 2. Key Rotation Interval
        EvaluationTestCase(
            query="How frequently must encryption keys be rotated under policy SEC-405?",
            target_chunk_ids=["eval-sec-002"],
            category="exact_code",
            description="Exact encryption key rotation policy",
            reference_answer="Under policy SEC-405, encryption keys must be rotated at least every 90 days, and backups must be encrypted at rest using AES-256-GCM.",
        ),
        # 3. Semantic Query with Vocabulary Mismatch
        EvaluationTestCase(
            query="What time off is granted to new mothers or fathers welcoming a newborn?",
            target_chunk_ids=["eval-hr-001"],
            category="semantic",
            description="Parental leave query with vocabulary variation",
            reference_answer="The company provides 16 weeks of fully paid parental leave for primary caregivers welcoming a child through birth, adoption, or foster placement.",
        ),
        # 4. Numerical Percentage Benefits Lookup
        EvaluationTestCase(
            query="What is the maximum company 401k matching percentage?",
            target_chunk_ids=["eval-hr-002"],
            category="numerical",
            description="401k match percentage query",
            reference_answer="The company provides dollar-for-dollar 401(k) matching up to 6% of base salary, vesting immediately upon enrollment.",
        ),
        # 5. Technical Network Parameter
        EvaluationTestCase(
            query="What is the internal VPC peering CIDR block?",
            target_chunk_ids=["eval-cloud-001"],
            category="technical_param",
            description="VPC CIDR network specification",
            reference_answer="Internal VPC peering operates on CIDR range 10.240.0.0/16 with strict security group egress filtering enabled.",
        ),
        # 6. Incident Response Timeframe
        EvaluationTestCase(
            query="How fast must an incident commander be assigned during a tier-1 outage?",
            target_chunk_ids=["eval-cloud-002"],
            category="sla_timeframe",
            description="Incident commander assignment SLA",
            reference_answer="Tier-1 production outages require an incident commander assigned within 15 minutes and a stakeholder briefing within 30 minutes.",
        ),
        # 7. Travel Meal Per Diem Cap
        EvaluationTestCase(
            query="What is the maximum daily meal per diem for domestic business travel?",
            target_chunk_ids=["eval-fin-001"],
            category="numerical",
            description="Domestic travel meal per diem limit",
            reference_answer="Domestic business travel meal reimbursement is capped at $75 per day without alcohol, requiring itemized receipts for transactions over $25.",
        ),
        # 8. Purchase Order Approval Threshold
        EvaluationTestCase(
            query="Who must approve purchase orders exceeding $50,000?",
            target_chunk_ids=["eval-fin-002"],
            category="authorization",
            description="Procurement sign-off thresholds",
            reference_answer="Commitments exceeding $50,000 require sign-off from the Chief Financial Officer (CFO).",
        ),
        # 9. Mileage Reimbursement Rate
        EvaluationTestCase(
            query="What is the personal vehicle reimbursement rate per mile?",
            target_chunk_ids=["eval-fin-003"],
            category="numerical",
            description="Personal vehicle mileage rate",
            reference_answer="Personal vehicle business travel is reimbursed at $0.67 per mile based on documented odometer readings or Google Maps logs.",
        ),
        # 10. Database Rollback Command
        EvaluationTestCase(
            query="What command should be run if a database schema migration fails?",
            target_chunk_ids=["eval-runbook-001"],
            category="runbook_command",
            description="Alembic rollback procedure",
            reference_answer="Execute 'alembic downgrade -1' and notify the database on-call engineer within 5 minutes.",
        ),
        # 11. Redis Cache Eviction Policy
        EvaluationTestCase(
            query="What eviction policy and memory limit are used for production Redis caches?",
            target_chunk_ids=["eval-runbook-002"],
            category="technical_param",
            description="Redis cache eviction and maxmemory configuration",
            reference_answer="Production caches utilize volatile-lru eviction with a maximum memory threshold of 32GB, and keys must have a TTL not exceeding 24 hours.",
        ),
        # 12. Rate Limiting Headers
        EvaluationTestCase(
            query="What response headers are returned when an API client is rate limited?",
            target_chunk_ids=["eval-runbook-003"],
            category="technical_param",
            description="API rate limiting headers and status code",
            reference_answer="External public endpoints enforce 100 requests per minute and return HTTP 429 with 'Retry-After' and 'X-RateLimit-Remaining' headers.",
        ),
        # 13. Distractor Disambiguation (Certificate Renewal vs Key Rotation)
        EvaluationTestCase(
            query="How many days prior to expiration are automated TLS certificates renewed?",
            target_chunk_ids=["eval-runbook-004"],
            category="distractor_disambiguation",
            description="Distinguish TLS cert renewal (30 days) from encryption key rotation (90 days)",
            reference_answer="Automated certbot renewals trigger 30 days prior to certificate expiry.",
        ),
        # 14. Distractor Disambiguation (CVE Patching vs Outage SLA)
        EvaluationTestCase(
            query="Within what timeframe must critical CVE vulnerability disclosures be patched?",
            target_chunk_ids=["eval-sec-004"],
            category="distractor_disambiguation",
            description="Distinguish CVE patching SLA (24 hours) from incident response SLAs",
            reference_answer="Policy SEC-415 requires critical CVE vulnerability disclosures to have remediation patching deployed within 24 hours.",
        ),
        # 15. Security Log Retention
        EvaluationTestCase(
            query="How many days must security audit logs be stored in immutable cold storage?",
            target_chunk_ids=["eval-sec-003"],
            category="exact_code",
            description="Security audit log retention requirement",
            reference_answer="Policy SEC-410 requires security audit logs to be retained in immutable cold storage for at least 365 days.",
        ),
        # 16. Remote Work Home Setup Stipend
        EvaluationTestCase(
            query="How much money do remote employees get for home office setup and monthly internet?",
            target_chunk_ids=["eval-hr-003"],
            category="semantic",
            description="Remote work setup stipend and internet allowance",
            reference_answer="Remote employees receive a one-time home office setup stipend of $1,000 and a monthly internet allowance of $75.",
        ),
        # 17. Notion Sprint Schedule
        EvaluationTestCase(
            query="When do bi-weekly sprints start and conclude in the engineering cadence?",
            target_chunk_ids=["eval-notion-001"],
            category="notion_workflow",
            description="Notion engineering sprint planning cadence",
            reference_answer="Sprints begin with planning on alternate Mondays at 10:00 AM UTC and conclude with demo retrospectives on alternate Fridays at 3:00 PM UTC.",
        ),
        # 18. Notion Post-Mortem Timeline
        EvaluationTestCase(
            query="Within how many hours must a blameless post-mortem be completed after a Sev-1 incident?",
            target_chunk_ids=["eval-notion-002"],
            category="notion_workflow",
            description="Notion severity-1 blameless post-mortem deadline",
            reference_answer="All Severity-1 incidents mandate a blameless post-mortem document completed within 72 hours and published in the Engineering Wiki.",
        ),
        # 19. Unanswerable Query 1
        EvaluationTestCase(
            query="What is the meal reimbursement allowance for international European business trips?",
            target_chunk_ids=[],
            category="unanswerable",
            description="Out-of-domain unanswerable question regarding international travel",
            reference_answer="The available documents do not contain sufficient evidence to answer this question.",
        ),
        # 20. Unanswerable Query 2
        EvaluationTestCase(
            query="What are the guidelines for bringing emotional support pets into the office?",
            target_chunk_ids=[],
            category="unanswerable",
            description="Unindexed workplace pet policy question",
            reference_answer="The available documents do not contain sufficient evidence to answer this question.",
        ),
    ]

    return chunks, queries


class RetrievalBenchmarkHarness:
    """Executes multi-mode retrieval benchmark and computes measured Recall@K, MRR@K, Precision@K, and Context Precision."""

    def __init__(self, search_service: SearchService):
        self.search_service = search_service

    def evaluate_mode(
        self,
        mode: SearchMode,
        test_cases: list[EvaluationTestCase],
        k: int = 5,
    ) -> BenchmarkMetricResult:
        """Evaluates a single search mode across test cases and computes exact metrics."""
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        precisions: list[float] = []
        context_precisions: list[float] = []
        latencies_ms: list[float] = []

        # Only evaluate answerable cases for standard retrieval metrics
        answerable_cases = [tc for tc in test_cases if tc.target_chunk_ids]

        for tc in answerable_cases:
            t0 = time.perf_counter()
            query_obj = SearchQuery(
                query=tc.query,
                mode=mode,
                top_k=k,
                workspace_id="default",
            )
            response = self.search_service.search(query_obj)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            retrieved_ids = [r.chunk_id for r in response.results[:k]]
            target_set = set(tc.target_chunk_ids)

            # Calculate Recall@K
            found_targets = [cid for cid in retrieved_ids if cid in target_set]
            recall = len(found_targets) / len(target_set) if target_set else 0.0
            recalls.append(recall)

            # Calculate MRR@K (Reciprocal Rank of first relevant item)
            rr = 0.0
            for rank, cid in enumerate(retrieved_ids, start=1):
                if cid in target_set:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            # Calculate Precision@K
            precision = len(found_targets) / k if k > 0 else 0.0
            precisions.append(precision)

            # Calculate Context Precision (MAP@K)
            cp = compute_context_precision(retrieved_ids, tc.target_chunk_ids, k=k)
            context_precisions.append(cp)

        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_context_precision = sum(context_precisions) / len(context_precisions) if context_precisions else 0.0
        avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

        return BenchmarkMetricResult(
            mode=mode.value,
            recall_at_5=round(avg_recall, 4),
            mrr_at_5=round(avg_mrr, 4),
            precision_at_5=round(avg_precision, 4),
            context_precision_at_5=round(avg_context_precision, 4),
            avg_latency_ms=round(avg_latency, 2),
            total_queries=len(answerable_cases),
        )

    def run_full_benchmark(
        self,
        test_cases: list[EvaluationTestCase],
        k: int = 5,
    ) -> dict[str, BenchmarkMetricResult]:
        """Runs benchmarks across all 4 modes (BM25, Semantic, Hybrid, Hybrid+Rerank)."""
        modes = [
            SearchMode.BM25,
            SearchMode.SEMANTIC,
            SearchMode.HYBRID,
            SearchMode.HYBRID_RERANK,
        ]
        results = {}
        for mode in modes:
            results[mode.value] = self.evaluate_mode(mode, test_cases, k=k)
        return results
