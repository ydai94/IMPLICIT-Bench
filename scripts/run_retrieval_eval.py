"""
Retrieval Accuracy: Neutral Prompt → Knowledge Graph.

Tests whether Qwen3-Embedding-8B embeddings of neutral prompts can correctly
retrieve the matching knowledge graph (KG) triple from the benchmark.

Usage:
    python run_retrieval_eval.py --gpu 0
    python run_retrieval_eval.py --gpu 0 --batch-size 32
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = "/data/gpfs/projects/punim2888"
MERGED_CSV = os.path.join(BASE_DIR, "stereoimage", "data", "benchmark_scores.csv")
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
MODEL_CACHE = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "stereoimage")


def build_kg_repr(row):
    """KG representation: head | relation | stereo_tails -> anti_tails."""
    return (f"{row['head']} | {row['relation']} | "
            f"{row['stereotype_tails']} → {row['anti_stereotype_tails']}")


def build_kg_repr_alt(row):
    """Alternative KG representation: target | axis | stereo_tails -> anti_tails."""
    return (f"{row['target']} | {row['axis']} | "
            f"{row['stereotype_tails']} → {row['anti_stereotype_tails']}")


def add_instruction(text, instruction):
    """Add Qwen3-Embedding instruction prefix."""
    return f"Instruct: {instruction}\nQuery: {text}"


def compute_retrieval_metrics(sim_matrix, ks=(1, 5, 10, 50)):
    """Compute retrieval metrics from a similarity matrix.

    Ground truth: row i matches column i.
    Returns dict of metrics and per-query ranks.
    """
    n = sim_matrix.shape[0]
    # For each query, get rank of the correct document
    sorted_indices = sim_matrix.argsort(axis=1)[:, ::-1]
    gt = np.arange(n)
    ranks = np.array([(sorted_indices[i] == gt[i]).argmax() + 1 for i in range(n)])

    metrics = {}
    for k in ks:
        metrics[f"R@{k}"] = (ranks <= k).mean() * 100
    metrics["MRR"] = (1.0 / ranks).mean()
    metrics["Mean_Rank"] = ranks.mean()
    metrics["Median_Rank"] = np.median(ranks)

    return metrics, ranks


def print_metrics(name, metrics, log):
    """Print and log retrieval metrics."""
    log(f"\n  --- {name} ---")
    log(f"    R@1={metrics['R@1']:.1f}%  R@5={metrics['R@5']:.1f}%  "
        f"R@10={metrics['R@10']:.1f}%  R@50={metrics['R@50']:.1f}%")
    log(f"    MRR={metrics['MRR']:.4f}  Mean_Rank={metrics['Mean_Rank']:.1f}  "
        f"Median_Rank={metrics['Median_Rank']:.0f}")


def per_group_metrics(ranks, cases, group_col, log):
    """Compute and log R@1, MRR per group."""
    log(f"\n    By {group_col}:")
    log(f"    {'group':25s} {'n':>5s}  {'R@1':>6s}  {'R@5':>6s}  {'R@10':>7s}  {'MRR':>6s}  {'MedRk':>6s}")
    log(f"    {'-'*25} {'-'*5}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}")
    for g in sorted(cases[group_col].dropna().unique()):
        mask = cases[group_col] == g
        r = ranks[mask.values]
        if len(r) < 5:
            continue
        r1 = (r <= 1).mean() * 100
        r5 = (r <= 5).mean() * 100
        r10 = (r <= 10).mean() * 100
        mrr = (1.0 / r).mean()
        med = np.median(r)
        log(f"    {g:25s} {len(r):5d}  {r1:5.1f}%  {r5:5.1f}%  {r10:6.1f}%  {mrr:6.4f}  {med:6.0f}")


def main():
    parser = argparse.ArgumentParser(description="Retrieval eval: prompt → KG")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load and de-dup to case level
    df = pd.read_csv(MERGED_CSV)
    cases = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    n = len(cases)
    print(f"Loaded {n} unique cases")

    # Build text representations
    prompts_neutral = cases["prompt_neutral"].tolist()
    prompts_stereo = cases["prompt_stereotype_trigger"].tolist()
    prompts_anti = cases["prompt_anti_stereotype_trigger"].tolist()
    kg_texts = [build_kg_repr(row) for _, row in cases.iterrows()]
    kg_texts_alt = [build_kg_repr_alt(row) for _, row in cases.iterrows()]

    print(f"\nExamples:")
    for i in [0, 1]:
        print(f"  Prompt: {prompts_neutral[i]}")
        print(f"  KG:     {kg_texts[i]}")
        print(f"  KG_alt: {kg_texts_alt[i]}")
        print()

    # Add instruction prefix for queries
    instruct = "Given a description, retrieve the matching knowledge graph triple"
    instruct_p2p = "Given a sentence, retrieve the most semantically similar sentence"
    prompts_neutral_q = [add_instruction(t, instruct) for t in prompts_neutral]
    prompts_stereo_q = [add_instruction(t, instruct_p2p) for t in prompts_stereo]
    prompts_anti_q = [add_instruction(t, instruct_p2p) for t in prompts_anti]

    # Load model
    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE, trust_remote_code=True)
    print("Model loaded")

    # Encode all texts
    print("Encoding neutral prompts (with instruction)...")
    emb_neutral_q = model.encode(prompts_neutral_q, normalize_embeddings=True,
                                 batch_size=args.batch_size, show_progress_bar=True)
    print("Encoding neutral prompts (plain, as documents)...")
    emb_neutral_doc = model.encode(prompts_neutral, normalize_embeddings=True,
                                   batch_size=args.batch_size, show_progress_bar=True)
    print("Encoding KG triples (head|relation)...")
    emb_kg = model.encode(kg_texts, normalize_embeddings=True,
                          batch_size=args.batch_size, show_progress_bar=True)
    print("Encoding KG triples (target|axis)...")
    emb_kg_alt = model.encode(kg_texts_alt, normalize_embeddings=True,
                              batch_size=args.batch_size, show_progress_bar=True)
    print("Encoding stereotype prompts...")
    emb_stereo_q = model.encode(prompts_stereo_q, normalize_embeddings=True,
                                batch_size=args.batch_size, show_progress_bar=True)
    print("Encoding anti-stereotype prompts...")
    emb_anti_q = model.encode(prompts_anti_q, normalize_embeddings=True,
                              batch_size=args.batch_size, show_progress_bar=True)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # === Retrieval evaluations ===
    summary_lines = []
    def log(msg):
        print(msg)
        summary_lines.append(msg)

    log("=" * 70)
    log("RETRIEVAL ACCURACY: Neutral Prompt → Knowledge Graph")
    log(f"Benchmark: {n} cases")
    log("=" * 70)

    results = cases[["id", "dataset", "bias_type", "target"]].copy()

    # 1. Neutral prompt → KG (head|relation)
    sim1 = emb_neutral_q @ emb_kg.T
    m1, ranks1 = compute_retrieval_metrics(sim1)
    print_metrics("Neutral Prompt → KG (head|relation|tails)", m1, log)
    per_group_metrics(ranks1, cases, "bias_type", log)
    per_group_metrics(ranks1, cases, "dataset", log)
    results["rank_prompt2kg"] = ranks1
    results["sim_prompt2kg"] = sim1[np.arange(n), np.arange(n)]

    # 2. Neutral prompt → KG (target|axis) — alternative repr
    sim2 = emb_neutral_q @ emb_kg_alt.T
    m2, ranks2 = compute_retrieval_metrics(sim2)
    print_metrics("Neutral Prompt → KG (target|axis|tails)", m2, log)
    per_group_metrics(ranks2, cases, "bias_type", log)
    results["rank_prompt2kg_alt"] = ranks2
    results["sim_prompt2kg_alt"] = sim2[np.arange(n), np.arange(n)]

    # 3. Stereotype prompt → Neutral prompt
    sim3 = emb_stereo_q @ emb_neutral_doc.T
    m3, ranks3 = compute_retrieval_metrics(sim3)
    print_metrics("Stereotype Prompt → Neutral Prompt", m3, log)
    per_group_metrics(ranks3, cases, "bias_type", log)
    results["rank_stereo2neutral"] = ranks3

    # 4. Anti-stereotype prompt → Neutral prompt
    sim4 = emb_anti_q @ emb_neutral_doc.T
    m4, ranks4 = compute_retrieval_metrics(sim4)
    print_metrics("Anti-Stereotype Prompt → Neutral Prompt", m4, log)
    per_group_metrics(ranks4, cases, "bias_type", log)
    results["rank_anti2neutral"] = ranks4

    # 5. KG → Neutral prompt (reverse direction, plain KG → plain neutral)
    sim5 = emb_kg @ emb_neutral_doc.T
    m5, ranks5 = compute_retrieval_metrics(sim5)
    print_metrics("KG (head|relation|tails) → Neutral Prompt", m5, log)
    per_group_metrics(ranks5, cases, "bias_type", log)
    results["rank_kg2prompt"] = ranks5

    # === Summary comparison ===
    log(f"\n{'='*70}")
    log("SUMMARY COMPARISON")
    log(f"{'='*70}")
    log(f"  {'Task':45s} {'R@1':>6s}  {'R@5':>6s}  {'R@10':>7s}  {'MRR':>6s}")
    log(f"  {'-'*45} {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}")
    for name, m in [
        ("Neutral → KG (head|relation|tails)", m1),
        ("Neutral → KG (target|axis|tails)", m2),
        ("Stereotype → Neutral", m3),
        ("Anti-Stereotype → Neutral", m4),
        ("KG → Neutral", m5),
    ]:
        log(f"  {name:45s} {m['R@1']:5.1f}%  {m['R@5']:5.1f}%  {m['R@10']:6.1f}%  {m['MRR']:6.4f}")

    # Save
    results.to_csv(os.path.join(OUTPUT_DIR, "retrieval_results.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "retrieval_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nSaved: retrieval_results.csv, retrieval_summary.txt")
    print("Done!")


if __name__ == "__main__":
    main()
