import argparse
import sys
import tempfile
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.models.search import SearchMode
from app.services.evaluation.benchmark import (
    RetrievalBenchmarkHarness,
    get_reproducible_evaluation_dataset,
)
from app.services.retrieval.bm25_indexer import BM25Retriever
from app.services.retrieval.reranker import FlashRankReranker
from app.services.retrieval.search_service import SearchService
from app.services.retrieval.vector_store import ChromaVectorStore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Secure Document Intelligence - Retrieval Evaluation & Quality Gates"
    )
    parser.add_argument(
        "--dataset",
        choices=["standard", "expanded"],
        default="expanded",
        help="Evaluation dataset: 'standard' (6 chunks, 8 queries) or 'expanded' (24 chunks, 20 queries).",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Cutoff rank K for Recall, MRR, Precision, and Context Precision (default: 5).",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Enforce automated CI quality gates on retrieval metrics.",
    )
    parser.add_argument(
        "--gate-mode",
        type=str,
        default="hybrid_rerank",
        help="Retrieval mode to evaluate for the quality gate (default: hybrid_rerank).",
    )
    parser.add_argument(
        "--gate-recall",
        type=float,
        default=0.85,
        help="Minimum Recall@K threshold required for CI gate to pass (default: 0.85).",
    )
    parser.add_argument(
        "--gate-mrr",
        type=float,
        default=0.80,
        help="Minimum MRR@K threshold required for CI gate to pass (default: 0.80).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 88)
    print("  SECURE DOCUMENT INTELLIGENCE: RETRIEVAL EVALUATION BENCHMARK")
    print(f"  Dataset: {args.dataset.upper()} | Rank K: {args.top_k} | Quality Gate Mode: {args.gate_mode}")
    print("=" * 88)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        chroma_path = Path(tmp_dir) / "chroma_eval"
        bm25 = BM25Retriever()
        vector_store = ChromaVectorStore(persist_dir=chroma_path)
        reranker = FlashRankReranker()
        search_service = SearchService(
            bm25_retriever=bm25,
            vector_store=vector_store,
            reranker=reranker,
        )

        chunks, queries = get_reproducible_evaluation_dataset(dataset_size=args.dataset)
        print(f"\n[1] Indexing {len(chunks)} multi-format chunks (PDF, DOCX, TXT, Notion)...")
        search_service.index_document_chunks(chunks, "Evaluation_Collection")
        print("    Indexing completed successfully.")

        print(f"\n[2] Executing Benchmark across {len(queries)} evaluation queries (k={args.top_k})...")
        harness = RetrievalBenchmarkHarness(search_service)
        results = harness.run_full_benchmark(queries, k=args.top_k)

        print("\n" + "=" * 94)
        print(f"  MEASURED EVALUATION RESULTS (K={args.top_k}, Dataset={args.dataset.capitalize()})")
        print("=" * 94)
        header = (
            f"{'Retrieval Mode':<16} | "
            f"{'Recall@' + str(args.top_k):<10} | "
            f"{'MRR@' + str(args.top_k):<10} | "
            f"{'Prec@' + str(args.top_k):<10} | "
            f"{'CtxPrec@' + str(args.top_k):<12} | "
            f"{'Latency (ms)':<12}"
        )
        print(header)
        print("-" * len(header))

        for mode_name, metric in results.items():
            print(
                f"{metric.mode:<16} | "
                f"{metric.recall_at_5:<10.4f} | "
                f"{metric.mrr_at_5:<10.4f} | "
                f"{metric.precision_at_5:<10.4f} | "
                f"{metric.context_precision_at_5:<12.4f} | "
                f"{metric.avg_latency_ms:<12.2f}"
            )
        print("=" * 94)
        print("Note: Results are directly measured on reproducible ground-truth test collection.\n")

        # CI Quality Gate Check
        if args.ci:
            target_metric = results.get(args.gate_mode)
            if not target_metric:
                print(f"[CI QUALITY GATE ERROR] Mode '{args.gate_mode}' not found in benchmark results.")
                sys.exit(1)

            recall_pass = target_metric.recall_at_5 >= args.gate_recall
            mrr_pass = target_metric.mrr_at_5 >= args.gate_mrr

            print("=" * 94)
            print("  CI QUALITY GATE VERIFICATION")
            print("=" * 94)
            print(f"  Target Mode:      {args.gate_mode}")
            print(f"  Recall@{args.top_k}:         {target_metric.recall_at_5:.4f} (Threshold: >={args.gate_recall:.4f}) -> {'PASS' if recall_pass else 'FAIL'}")
            print(f"  MRR@{args.top_k}:            {target_metric.mrr_at_5:.4f} (Threshold: >={args.gate_mrr:.4f}) -> {'PASS' if mrr_pass else 'FAIL'}")
            print(f"  ContextPrec@{args.top_k}:    {target_metric.context_precision_at_5:.4f}")
            print(f"  Avg Latency:      {target_metric.avg_latency_ms:.2f} ms")
            print("=" * 94)

            if not (recall_pass and mrr_pass):
                print(f"\n[CI QUALITY GATE FAILED] Mode '{args.gate_mode}' failed retrieval quality thresholds!")
                sys.exit(1)
            else:
                print(f"\n[CI QUALITY GATE PASSED] Mode '{args.gate_mode}' satisfied all retrieval quality thresholds.")


if __name__ == "__main__":
    main()
