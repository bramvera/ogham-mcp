#!/usr/bin/env python3
"""LoCoMo benchmark for Ogham MCP.

Evaluates retrieval quality using the LoCoMo dataset (Maharana et al.)
which tests long-term conversational memory with 1,986 QA pairs across
10 conversations.

Metrics:
- Recall@K: fraction of questions where the ground truth evidence
  appears in the top K retrieved results
- MRR: Mean Reciprocal Rank of the first relevant result

Usage:
    # Download LoCoMo dataset first:
    curl -L -o benchmarks/locomo10.json \
      https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

    # Run the benchmark:
    uv run python3 benchmarks/locomo_benchmark.py [--top-k 10] [--profile locomo]

    # Clean up after:
    uv run python3 benchmarks/locomo_benchmark.py --cleanup

Requires: Ogham MCP server configured with a working database and embedding provider.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent / "locomo10.json"
RESULTS_FILE = Path(__file__).parent / "locomo_results.json"

# LoCoMo category mapping
CATEGORIES = {
    "1": "single-hop",
    "2": "temporal",
    "3": "multi-hop",
    "4": "open-domain",
    "5": "adversarial",
}


def load_dataset() -> list[dict]:
    if not DATA_FILE.exists():
        print(f"Dataset not found at {DATA_FILE}")
        print("Download it:")
        print(
            "  curl -L -o benchmarks/locomo10.json "
            "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        )
        sys.exit(1)
    return json.loads(DATA_FILE.read_text())


def ingest_conversations(data: list[dict], profile: str) -> int:
    """Ingest LoCoMo conversations into Ogham as memories."""
    from ogham.database import store_memory as db_store
    from ogham.embeddings import generate_embedding

    total_stored = 0
    for sample in data:
        conv_id = sample["sample_id"]
        conversation = sample["conversation"]

        # Group turns into session-sized chunks (every 5 turns)
        chunk_size = 5
        for i in range(0, len(conversation), chunk_size):
            chunk = conversation[i : i + chunk_size]
            # Build a readable memory from the conversation chunk
            lines = []
            for turn in chunk:
                speaker = turn.get("speaker", turn.get("role", "unknown"))
                text = turn.get("text", turn.get("content", ""))
                lines.append(f"{speaker}: {text}")

            content = "\n".join(lines)
            if len(content.strip()) < 20:
                continue

            embedding = generate_embedding(content)
            db_store(
                content=content,
                embedding=embedding,
                profile=profile,
                source="locomo-benchmark",
                tags=[f"conv:{conv_id}", f"chunk:{i // chunk_size}"],
            )
            total_stored += 1

        logger.info("Ingested %s (%d turns -> %d chunks)", conv_id, len(conversation), total_stored)

    return total_stored


def evaluate(data: list[dict], profile: str, top_k: int = 10) -> dict:
    """Run QA evaluation against stored memories."""
    from ogham.database import hybrid_search_memories
    from ogham.embeddings import generate_embedding

    results = {
        "total_questions": 0,
        "recall_at_k": 0,
        "reciprocal_ranks": [],
        "by_category": {},
    }

    for sample in data:
        conv_id = sample["sample_id"]
        for qa in sample["qa"]:
            question = qa["question"]
            answer = qa["answer"]
            evidence = qa.get("evidence", "")
            category = str(qa.get("category", "unknown"))
            cat_name = CATEGORIES.get(category, f"cat-{category}")

            # Search Ogham
            query_embedding = generate_embedding(question)
            search_results = hybrid_search_memories(
                query_text=question,
                query_embedding=query_embedding,
                match_count=top_k,
                profile=profile,
            )

            # Check if evidence or answer appears in any of the top K results
            found = False
            rank = 0
            for i, result in enumerate(search_results):
                content = result.get("content", "")
                # Check if the answer or evidence text appears in the result
                if answer.lower() in content.lower() or (
                    evidence and evidence.lower()[:50] in content.lower()
                ):
                    found = True
                    rank = i + 1
                    break

            results["total_questions"] += 1
            if found:
                results["recall_at_k"] += 1
                results["reciprocal_ranks"].append(1.0 / rank)
            else:
                results["reciprocal_ranks"].append(0.0)

            # Per-category tracking
            if cat_name not in results["by_category"]:
                results["by_category"][cat_name] = {"total": 0, "recall": 0, "rr_sum": 0.0}
            results["by_category"][cat_name]["total"] += 1
            if found:
                results["by_category"][cat_name]["recall"] += 1
                results["by_category"][cat_name]["rr_sum"] += 1.0 / rank

        logger.info("Evaluated %s (%d questions so far)", conv_id, results["total_questions"])

    # Compute final metrics
    total = results["total_questions"]
    recall = results["recall_at_k"] / total if total > 0 else 0.0
    mrr = sum(results["reciprocal_ranks"]) / total if total > 0 else 0.0

    summary = {
        "total_questions": total,
        "top_k": top_k,
        "recall_at_k": round(recall * 100, 1),
        "mrr": round(mrr, 3),
        "by_category": {},
    }

    for cat, stats in results["by_category"].items():
        cat_total = stats["total"]
        summary["by_category"][cat] = {
            "total": cat_total,
            "recall_at_k": round(stats["recall"] / cat_total * 100, 1) if cat_total > 0 else 0.0,
            "mrr": round(stats["rr_sum"] / cat_total, 3) if cat_total > 0 else 0.0,
        }

    return summary


def cleanup(profile: str):
    """Remove all benchmark memories."""
    from ogham.database import get_backend

    backend = get_backend()
    client = backend._get_client()
    result = (
        client.table("memories")
        .delete()
        .eq("profile", profile)
        .eq("source", "locomo-benchmark")
        .execute()
    )
    count = len(result.data) if result.data else 0
    logger.info("Deleted %d benchmark memories from profile '%s'", count, profile)


def main():
    parser = argparse.ArgumentParser(description="LoCoMo benchmark for Ogham MCP")
    parser.add_argument("--top-k", type=int, default=10, help="Top K for Recall@K (default: 10)")
    parser.add_argument("--profile", default="locomo", help="Ogham profile for benchmark data")
    parser.add_argument("--cleanup", action="store_true", help="Remove benchmark data and exit")
    parser.add_argument(
        "--skip-ingest", action="store_true", help="Skip ingestion (use existing data)"
    )
    args = parser.parse_args()

    if args.cleanup:
        cleanup(args.profile)
        return

    data = load_dataset()
    total_qa = sum(len(s["qa"]) for s in data)
    logger.info("Loaded %d conversations with %d QA pairs", len(data), total_qa)

    if not args.skip_ingest:
        logger.info("Ingesting conversations into profile '%s'...", args.profile)
        start = time.time()
        stored = ingest_conversations(data, args.profile)
        elapsed = time.time() - start
        logger.info("Ingested %d chunks in %.1fs", stored, elapsed)

    logger.info("Running evaluation (Recall@%d)...", args.top_k)
    start = time.time()
    results = evaluate(data, args.profile, args.top_k)
    elapsed = time.time() - start

    results["evaluation_time_s"] = round(elapsed, 1)
    results["embedding_provider"] = os.environ.get("EMBEDDING_PROVIDER", "unknown")

    # Print results
    print("\n" + "=" * 50)
    print("LoCoMo Benchmark Results (Ogham MCP)")
    print("=" * 50)
    print(f"Questions:      {results['total_questions']}")
    print(f"Recall@{args.top_k}:      {results['recall_at_k']}%")
    print(f"MRR:            {results['mrr']}")
    print(f"Provider:       {results['embedding_provider']}")
    print(f"Eval time:      {results['evaluation_time_s']}s")
    print()
    print("By category:")
    for cat, stats in sorted(results["by_category"].items()):
        r_at_k = stats["recall_at_k"]
        mrr = stats["mrr"]
        n = stats["total"]
        print(f"  {cat:15s}  Recall@{args.top_k}: {r_at_k:5.1f}%  MRR: {mrr:.3f}  (n={n})")

    # Save results
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
