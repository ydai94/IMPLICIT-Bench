"""
Extract best-matching KG triple for each neutral prompt via embedding retrieval.

Uses cached embeddings from cache_embeddings.py (CPU-only, no GPU needed).

Usage:
    python cache_kg_extraction.py
"""

import json
import os

import numpy as np
import pandas as pd

from config import CACHE_DIR, DATA_CSV, EMBEDDINGS_DIR, EXTRACTED_KG_CSV


def main():
    # Load dataset
    df = pd.read_csv(DATA_CSV)
    n = len(df)
    print(f"Loaded {n} cases from {DATA_CSV}")

    # Load cached embeddings
    emb_neutral = np.load(os.path.join(EMBEDDINGS_DIR, "neutral_prompts.npy"))
    emb_kg = np.load(os.path.join(EMBEDDINGS_DIR, "kg_triples.npy"))
    with open(os.path.join(EMBEDDINGS_DIR, "ids.json")) as f:
        ids = json.load(f)

    assert len(ids) == n, f"ID count mismatch: {len(ids)} vs {n}"
    print(f"Loaded embeddings: neutral={emb_neutral.shape}, kg={emb_kg.shape}")

    # Compute similarity matrix
    print("Computing similarity matrix...")
    sim = emb_neutral @ emb_kg.T  # (n, n)

    # For each row, find best match excluding self
    print("Finding best matches (excluding self)...")
    # Zero out diagonal so self-match is never chosen
    np.fill_diagonal(sim, -1.0)
    best_indices = sim.argmax(axis=1)
    best_scores = sim[np.arange(n), best_indices]

    # Build extraction results
    rows = []
    for i in range(n):
        j = best_indices[i]
        matched_row = df.iloc[j]
        rows.append({
            "id": ids[i],
            "extracted_head": matched_row["head"],
            "extracted_relation": matched_row["relation"],
            "extracted_stereotype_tails": matched_row["stereotype_tails"],
            "extracted_anti_stereotype_tails": matched_row["anti_stereotype_tails"],
            "extracted_prompt_stereotype": matched_row["prompt_stereotype"],
            "extracted_prompt_anti_stereotype": matched_row["prompt_anti_stereotype"],
            "extracted_from_id": ids[j],
            "similarity_score": float(best_scores[i]),
        })

    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(EXTRACTED_KG_CSV), exist_ok=True)
    result_df.to_csv(EXTRACTED_KG_CSV, index=False)

    # Print summary stats
    print(f"\nSaved {len(result_df)} extraction results to {EXTRACTED_KG_CSV}")
    print(f"  Similarity scores: mean={best_scores.mean():.4f}, "
          f"min={best_scores.min():.4f}, max={best_scores.max():.4f}, "
          f"median={np.median(best_scores):.4f}")
    print(f"\nTop-5 examples:")
    top5 = result_df.nlargest(5, "similarity_score")
    for _, r in top5.iterrows():
        print(f"  [{r['similarity_score']:.4f}] {r['id'][:12]}... -> "
              f"{r['extracted_from_id'][:12]}... "
              f"({r['extracted_head']} | {r['extracted_relation']})")

    print("\nDone!")


if __name__ == "__main__":
    main()
