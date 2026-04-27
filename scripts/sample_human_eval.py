"""Random sampler for the human evaluation.

Reads `data/merged_all_aggregated.csv` (the merged StereoSet + CrowS-Pairs pool)
and draws a uniform random sample of N cases (default 50) across the entire
benchmark, with no per-bias-type stratification — bias-type and source mix
follow the natural row distribution of the merged pool.

Output: `data/human_eval/sampled_cases.csv` with all KG and prompt fields
copied through, plus a `human_eval_idx` 1..N column for stable ordering.
"""

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_CSV = os.path.join(REPO_ROOT, "data", "merged_all_aggregated.csv")
OUT_DIR = os.path.join(REPO_ROOT, "data", "human_eval")
OUT_CSV = os.path.join(OUT_DIR, "sampled_cases.csv")


def random_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) < n:
        raise ValueError(f"Pool has {len(df)} rows, need {n}.")
    sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)
    sampled.insert(0, "human_eval_idx", range(1, len(sampled) + 1))
    return sampled


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", default=SOURCE_CSV)
    p.add_argument("--out-csv", default=OUT_CSV)
    p.add_argument("--n", type=int, default=50)
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

    sampled = random_sample(df, args.n, args.seed)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    sampled.to_csv(args.out_csv, index=False)

    print(f"Wrote {len(sampled)} rows to {args.out_csv}")
    print("\nPer bias_type:")
    print(sampled["bias_type"].value_counts().sort_index().to_string())
    print("\nPer source:")
    print(sampled["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
