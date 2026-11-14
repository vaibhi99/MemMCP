"""
Benchmark MemMCP Context Precision.
Calculates Mean Average Precision (MAP) / Context Precision over the evaluation dataset.
Uses an offline local text embedder so it never hits Google Gemini 429 rate limits.
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

from memmcp.config import Settings
from memmcp.memory_manager import MemoryManager
from memmcp.embeddings.base import Embedder, normalize

# Path to the CSV dataset
CSV_PATH = Path(
    r"C:\Users\HP\.gemini\antigravity\brain\6607a890-30a6-4d4a-a6fd-686e8c2af032\evaluation_dataset.csv"
)


class LocalDeterministicEmbedder(Embedder):
    """
    Fast, local, offline bag-of-words / hash embedder.
    Requires no external APIs and never hits rate limits (429 RESOURCE_EXHAUSTED).
    Produces normalized 768-dimensional vectors with high semantic cosine similarity
    for matching terms and low similarity for noise.
    """
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = text.lower().replace("-", " ").replace(":", " ").replace("/", " ").split()
        for w in words:
            # Hash words deterministically across buckets
            h = hash(w) % self.dim
            vec[h] += 1.0
        return normalize(vec)


def calculate_context_precision(retrieved_contexts: list[str], ground_truth: str) -> float:
    """
    Computes standard Context Precision (Mean Average Precision @ K):
    Precision@k = (relevant items up to rank k) / k
    Context Precision = Sum(Precision@k * is_relevant(k)) / (total relevant items retrieved)
    """
    gt_clean = ground_truth.strip().lower()
    
    hits = 0
    precision_sum = 0.0

    for rank, ctx in enumerate(retrieved_contexts, start=1):
        ctx_clean = ctx.strip().lower()
        # Check if ground truth key concepts match retrieved memory
        is_hit = gt_clean in ctx_clean or ctx_clean in gt_clean
        
        # Also check significant word overlap
        if not is_hit:
            gt_words = set(w for w in gt_clean.split() if len(w) > 3)
            ctx_words = set(w for w in ctx_clean.split() if len(w) > 3)
            overlap = len(gt_words & ctx_words) / max(len(gt_words), 1)
            if overlap >= 0.6:
                is_hit = True

        if is_hit:
            hits += 1
            precision_sum += hits / rank

    if hits == 0:
        return 0.0

    return precision_sum / hits


def run_benchmark():
    if not CSV_PATH.exists():
        print(f"Error: Dataset not found at {CSV_PATH}")
        return

    # Load dataset
    df = pd.read_csv(CSV_PATH)
    # Filter out non-numeric placeholder rows
    df = df[df["query_id"].astype(str).str.isdigit()]

    print(f"\n=======================================================")
    print(f"   MEMMCP CONTEXT PRECISION EVALUATION BENCHMARK       ")
    print(f"=======================================================")
    print(f"Loaded {len(df)} test evaluation scenarios from CSV.\n")

    # Use a clean, isolated temporary directory for benchmark
    temp_dir = Path(tempfile.mkdtemp(prefix="memmcp_eval_"))
    settings = Settings(
        data_dir=temp_dir,
        pii_policy="redact",
        judge_enabled=False,
    )
    settings.ensure_dirs()
    
    manager = MemoryManager(settings=settings)
    # Override with our offline local embedder to bypass rate limits
    manager.embedder = LocalDeterministicEmbedder(dim=768)

    print("Phase 1: Seeding Memory Store (Ground Truth + Stale + Noise Facts)...")

    # 1. Seed stale facts first (simulating historical knowledge)
    for _, row in df.iterrows():
        scope = str(row["expected_scope"])
        proj = None if scope == "global" else scope.replace("project:", "")
        stale = str(row.get("stale_facts", "")).strip()
        if stale and stale.lower() != "nan":
            manager.remember(
                content=stale,
                project_name=proj,
                importance=0.4,
                key=f"fact_{row['query_id']}",  # Conflict key
            )

    # 2. Seed noise/distractor facts
    for _, row in df.iterrows():
        scope = str(row["expected_scope"])
        proj = None if scope == "global" else scope.replace("project:", "")
        noise_str = str(row.get("noise_facts", "")).strip()
        if noise_str and noise_str.lower() != "nan":
            for noise_item in noise_str.split(";"):
                if noise_item.strip():
                    manager.remember(
                        content=noise_item.strip(),
                        project_name=proj,
                        importance=0.3,
                    )

    # 3. Seed Ground Truth facts (supersedes stale ones via conflict key)
    for _, row in df.iterrows():
        scope = str(row["expected_scope"])
        proj = None if scope == "global" else scope.replace("project:", "")
        gt = str(row["ground_truth_fact"]).strip()
        manager.remember(
            content=gt,
            project_name=proj,
            importance=0.9,
            key=f"fact_{row['query_id']}",  # Same key -> Distiller supersedes stale
        )

    print("Phase 2: Executing Recall & Evaluating Retrieval Ranks...\n")

    results = []

    for _, row in df.iterrows():
        qid = row["query_id"]
        query = str(row["query"])
        scope = str(row["expected_scope"])
        proj = None if scope == "global" else scope.replace("project:", "")
        gt = str(row["ground_truth_fact"]).strip()

        # Call MemMCP recall tool
        recalled = manager.recall(
            query=query,
            project_name=proj,
            top_k=5,
        )

        retrieved_texts = [item.memory.content for item in recalled]
        
        # Calculate Context Precision score
        score = calculate_context_precision(retrieved_texts, gt)
        
        # Check rank of ground truth
        rank = -1
        for idx, text in enumerate(retrieved_texts, start=1):
            if gt.lower() in text.lower() or text.lower() in gt.lower():
                rank = idx
                break

        results.append({
            "query_id": qid,
            "query": query,
            "ground_truth": gt,
            "top_1_hit": rank == 1,
            "rank": rank if rank != -1 else "Miss",
            "context_precision": score
        })

    # Summary calculations
    res_df = pd.DataFrame(results)
    mean_precision = res_df["context_precision"].mean()
    top_1_accuracy = res_df["top_1_hit"].mean()

    print("--- Sample Retrieval Results ---")
    print(res_df[["query_id", "rank", "context_precision", "query"]].head(10).to_string(index=False))

    print("\n=======================================================")
    print("                    FINAL METRICS                      ")
    print("=======================================================")
    print(f"Total Queries Evaluated : {len(res_df)}")
    print(f"Top-1 Retrieval Accuracy: {top_1_accuracy:.1%}")
    print(f"Mean Context Precision  : {mean_precision:.1%}")
    print("=======================================================\n")

    # Clean up temp test directory
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_benchmark()
