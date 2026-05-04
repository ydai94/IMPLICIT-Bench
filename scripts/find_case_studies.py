"""Find case studies for the introduction figure.

Selects cases where all four baseline T2I models (Qwen-Image, GPT-Image,
SD-3, Nano-Banana) generate the *neutral* prompt with high stereotype bias
(mean score >= --baseline-min across 3 seeds, every seed prompt-aligned),
AND our method (exp_09) drops bias to <= --ours-max with all 3 seeds aligned.

Outputs a ranked CSV plus a per-case directory of staged images for figure
assembly.
"""

import argparse
import os
import shutil
import sys
from typing import Dict

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO_ROOT, "cache", "eval_results")
SOURCE_CSV = os.path.join(REPO_ROOT, "data", "benchmark_prompts.csv")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data", "case_studies_intro_figure")

# (label, eval_csv, alignment_csv)
EXPERIMENTS = [
    ("qwen", "exp_00_eval.csv",         "exp_00_alignment.csv"),
    ("gpt",  "exp_12_eval.csv",         "exp_12_alignment_local.csv"),
    ("sd3",  "exp_13_eval.csv",         "exp_13_alignment_local.csv"),
    ("nano", "exp_14_eval_local.csv",   "exp_14_alignment_local.csv"),
    ("ours", "exp_11_eval.csv",         "exp_11_alignment.csv"),
]
BASELINES = ["qwen", "gpt", "sd3", "nano"]


def load_one(label: str, eval_csv: str, align_csv: str) -> pd.DataFrame:
    """Aggregate one experiment to one row per case_id."""
    bias = pd.read_csv(os.path.join(EVAL_DIR, eval_csv),
                       usecols=["case_id", "seed", "image_path", "score"])
    align = pd.read_csv(os.path.join(EVAL_DIR, align_csv),
                        usecols=["case_id", "seed", "image_path", "aligned"])

    # All rows in these eval CSVs are neutral-prompt evaluations by design
    # (exp_00/12/13/14 baselines and exp_11 our method are all evaluated on
    # the neutral image only). exp_00 has some rows with empty image_path
    # (backfilled scores) -- keep them, they're still neutral evals.
    bias["score"] = pd.to_numeric(bias["score"], errors="coerce")
    bias = bias.dropna(subset=["score"])
    align["aligned"] = align["aligned"].astype(bool)

    # exp_00 has rows with empty image_path (backfilled from legacy CSV).
    # Reconstruct from the canonical experiment_outputs layout so we can
    # stage every seed.
    if label == "qwen":
        canonical = bias["case_id"].astype(str) + "/seed_" + bias["seed"].astype(str) + "/neutral.png"
        canonical = (REPO_ROOT + "/experiment_outputs/exp_00_baseline/" + canonical)
        bias["image_path"] = bias["image_path"].fillna("").where(
            bias["image_path"].fillna("").str.len() > 0, canonical)

    # Per-case bias agg
    bias_agg = bias.groupby("case_id").agg(
        mean_bias=("score", "mean"),
        n_bias_seeds=("score", "count"),
    ).reset_index()

    # Per-case alignment agg (count True seeds, total seeds)
    align_agg = align.groupby("case_id").agg(
        aligned_seeds=("aligned", "sum"),
        n_align_seeds=("aligned", "count"),
    ).reset_index()
    align_agg["all_aligned"] = align_agg["aligned_seeds"] == align_agg["n_align_seeds"]

    # Per-case (case_id -> seed -> image_path) for staging
    paths = (
        bias[["case_id", "seed", "image_path"]]
        .drop_duplicates(subset=["case_id", "seed"])
        .pivot(index="case_id", columns="seed", values="image_path")
    )
    paths.columns = [f"path_seed{int(c)}" for c in paths.columns]
    paths = paths.reset_index()

    out = bias_agg.merge(align_agg, on="case_id", how="outer").merge(
        paths, on="case_id", how="outer")
    out = out.rename(columns={
        "mean_bias": f"bias_{label}",
        "n_bias_seeds": f"n_bias_{label}",
        "aligned_seeds": f"aligned_seeds_{label}",
        "n_align_seeds": f"n_align_{label}",
        "all_aligned": f"all_aligned_{label}",
        **{f"path_seed{k}": f"{label}_path_seed{k}" for k in (0, 1, 2)},
    })
    return out


def stage_images(row: pd.Series, dest_dir: str) -> Dict[str, int]:
    """Copy this case's images into dest_dir/<rank>_<case_id>/ ."""
    os.makedirs(dest_dir, exist_ok=True)
    counts = {"copied": 0, "missing": 0}
    for label, _, _ in EXPERIMENTS:
        for seed in (0, 1, 2):
            src = row.get(f"{label}_path_seed{seed}")
            if not isinstance(src, str) or not src or not os.path.exists(src):
                counts["missing"] += 1
                continue
            dst = os.path.join(dest_dir, f"{label}_seed{seed}.png")
            shutil.copy2(src, dst)
            counts["copied"] += 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-min", type=float, default=4.0,
                   help="Minimum mean bias for each baseline model.")
    p.add_argument("--ours-max", type=float, default=1.0,
                   help="Maximum mean bias for our method.")
    p.add_argument("--top-n", type=int, default=20,
                   help="Stage images for at most this many top cases.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--no-copy", action="store_true",
                   help="Skip copying images; only write the CSV.")
    p.add_argument("--ours-min-aligned-seeds", type=int, default=3,
                   help="Minimum aligned seeds for our method (1-3). "
                        "Lower this to 2 if the strict 3/3 filter is too "
                        "tight (exp_09 over-debiases on some cases).")
    p.add_argument("--baselines-min-aligned-seeds", type=int, default=3,
                   help="Minimum aligned seeds for each baseline (1-3).")
    args = p.parse_args()

    # Load each experiment and merge by case_id
    merged = None
    for label, eval_csv, align_csv in EXPERIMENTS:
        df = load_one(label, eval_csv, align_csv)
        merged = df if merged is None else merged.merge(df, on="case_id", how="outer")
        print(f"[load] {label}: {len(df)} cases")

    print(f"[merge] {len(merged)} unique case_ids across all experiments")

    # Strict filter: each baseline mean >= baseline_min, fully aligned, 3 seeds;
    # ours mean <= ours_max, fully aligned, 3 seeds.
    cond = pd.Series(True, index=merged.index)
    for lbl in BASELINES:
        cond &= (merged[f"bias_{lbl}"] >= args.baseline_min)
        cond &= (merged[f"aligned_seeds_{lbl}"] >= args.baselines_min_aligned_seeds)
        cond &= (merged[f"n_bias_{lbl}"] == 3)
    cond &= (merged["bias_ours"] <= args.ours_max)
    cond &= (merged["aligned_seeds_ours"] >= args.ours_min_aligned_seeds)
    cond &= (merged["n_bias_ours"] == 3)

    survivors = merged[cond].copy()
    print(f"[filter] {len(survivors)} cases pass "
          f"(baselines >= {args.baseline_min}, ours <= {args.ours_max}, "
          f"all 3 seeds aligned everywhere)")

    if len(survivors) == 0:
        print("\nNo cases pass the strict filter. Try loosening --baseline-min "
              "(e.g. 3.5) or --ours-max (e.g. 1.5).")

    # Rank by min-baseline minus ours (cleanest, biggest gap first)
    survivors["min_baseline"] = survivors[[f"bias_{l}" for l in BASELINES]].min(axis=1)
    survivors["gap"] = survivors["min_baseline"] - survivors["bias_ours"]
    survivors = survivors.sort_values(
        ["gap", "bias_ours"], ascending=[False, True]).reset_index(drop=True)
    survivors.insert(0, "rank", range(1, len(survivors) + 1))

    # Join prompts/KG fields
    src = pd.read_csv(SOURCE_CSV)
    src = src.rename(columns={"id": "case_id"})
    keep = ["case_id", "source", "bias_type", "target", "head", "relation",
            "stereotype_tails", "anti_stereotype_tails", "prompt_neutral"]
    survivors = survivors.merge(src[keep], on="case_id", how="left")

    # Write CSV
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "candidates.csv")
    out_cols = (
        ["rank", "case_id", "source", "bias_type", "target", "head",
         "relation", "stereotype_tails", "anti_stereotype_tails",
         "prompt_neutral",
         "bias_qwen", "bias_gpt", "bias_sd3", "bias_nano", "bias_ours",
         "min_baseline", "gap"]
    )
    survivors[out_cols].to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")

    # Stage images for the top-N
    if not args.no_copy and len(survivors) > 0:
        img_root = os.path.join(args.out_dir, "images")
        os.makedirs(img_root, exist_ok=True)
        n = min(args.top_n, len(survivors))
        print(f"[stage] copying images for top {n} cases -> {img_root}/")
        for _, row in survivors.head(n).iterrows():
            sub = os.path.join(img_root, f"{int(row['rank']):02d}_{row['case_id']}")
            counts = stage_images(row, sub)
            print(f"  rank {int(row['rank']):02d} {row['case_id']}: "
                  f"{counts['copied']} copied, {counts['missing']} missing")

    # Brief on-screen summary
    if len(survivors) > 0:
        print("\nTop 10 candidates:")
        top = survivors.head(10)
        for _, row in top.iterrows():
            print(f"  #{int(row['rank']):02d} [{row['bias_type']}] {row['case_id']}: "
                  f"qwen={row['bias_qwen']:.2f} gpt={row['bias_gpt']:.2f} "
                  f"sd3={row['bias_sd3']:.2f} nano={row['bias_nano']:.2f} "
                  f"ours={row['bias_ours']:.2f} (gap={row['gap']:.2f})")
            print(f"      prompt: {row['prompt_neutral']}")


if __name__ == "__main__":
    main()
