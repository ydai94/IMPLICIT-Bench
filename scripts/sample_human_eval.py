"""Stratified sampler for the human evaluation.

Reads `data/merged_all_aggregated.csv` (1,831 prompt units across 11 bias types
and 2 source datasets) and selects 100 cases with a per-bias-type allocation
that puts a floor under small categories and a cap on large ones, so every
bias type carries enough N for per-category human-vs-VLM correlation.

Output: `data/human_eval/sampled_cases.csv` with all KG and prompt fields
copied through, plus a `human_eval_idx` 1..100 column for stable ordering.
"""

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_CSV = os.path.join(REPO_ROOT, "data", "merged_all_aggregated.csv")
OUT_DIR = os.path.join(REPO_ROOT, "data", "human_eval")
OUT_CSV = os.path.join(OUT_DIR, "sampled_cases.csv")

# Per-bias-type sample allocation. Sums to 50 (each case becomes 4 Forms
# questions: 1 KG-validity + 3 image ratings, total 200 — fits Forms' cap).
# Floor = 3 per type, cap = 6 for the three big StereoSet categories; the rest
# is distributed roughly proportionally to pool size.
ALLOCATION = {
    "profession": 6,
    "race": 6,
    "gender": 6,
    "socioeconomic": 5,
    "religion": 5,
    "race-color": 5,
    "age": 4,
    "nationality": 4,
    "sexual-orientation": 3,
    "disability": 3,
    "physical-appearance": 3,
}


def stratified_sample(df: pd.DataFrame, allocation: dict, seed: int) -> pd.DataFrame:
    chunks = []
    for bias_type, n in allocation.items():
        pool = df[df["bias_type"] == bias_type]
        if len(pool) == 0:
            raise ValueError(f"No rows in pool for bias_type={bias_type!r}")
        if len(pool) < n:
            raise ValueError(
                f"Pool for bias_type={bias_type!r} has {len(pool)} rows, "
                f"need {n}. Lower the allocation."
            )
        chunks.append(pool.sample(n=n, random_state=seed))
    sampled = pd.concat(chunks, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled.insert(0, "human_eval_idx", range(1, len(sampled) + 1))
    return sampled


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", default=SOURCE_CSV)
    p.add_argument("--out-csv", default=OUT_CSV)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.source_csv)
    needed_cols = {
        "source", "id", "target", "bias_type", "head", "relation",
        "stereotype_tails", "anti_stereotype_tails",
        "prompt_neutral", "prompt_stereotype", "prompt_anti_stereotype",
    }
    missing = needed_cols - set(df.columns)
    if missing:
        sys.exit(f"Source CSV missing columns: {sorted(missing)}")

    sampled = stratified_sample(df, ALLOCATION, args.seed)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    sampled.to_csv(args.out_csv, index=False)

    print(f"Wrote {len(sampled)} rows to {args.out_csv}")
    print("\nPer bias_type:")
    print(sampled["bias_type"].value_counts().sort_index().to_string())
    print("\nPer source:")
    print(sampled["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
