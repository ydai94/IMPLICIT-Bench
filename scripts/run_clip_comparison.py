"""
Lean Stereotype Benchmark: CLIP + Qwen3-VL + Gemma4 Analysis.

Computes CLIP image embedding similarities and merges with VLM evaluation
scores for the curated lean_stereotype cases from StereoSet and CrowS-Pairs.
Only processes cases present in the benchmark CSVs.

Usage:
    python run_clip_comparison.py                    # full run
    python run_clip_comparison.py --resume           # resume from partial
    python run_clip_comparison.py --skip-plots       # compute only
    python run_clip_comparison.py --gpu 0 --batch-size 64
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from scipy import stats
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# --- Constants ---
BASE_DIR = "/data/gpfs/projects/punim2888"
STEREOSET_IMAGE_BASE = os.path.join(BASE_DIR, "stereoset", "generated_images", "qwen_v5")
CROWSPAIRS_IMAGE_BASE = os.path.join(BASE_DIR, "crows-pairs", "generated_images", "qwen")
STEREOSET_CSV = os.path.join(BASE_DIR, "stereoimage", "data", "lean_stereotype_union.csv")
CROWSPAIRS_CSV = os.path.join(BASE_DIR, "stereoimage", "data", "neutral_lean_stereotype.csv")
MODEL_NAME = "openai/clip-vit-large-patch14"
MODEL_CACHE = os.path.join(BASE_DIR, "models", "clip-vit-large-patch14")
SEEDS = range(3)
IMAGE_TYPES = ["neutral", "stereotype_trigger", "anti_stereotype_trigger"]
DIR_PATTERN = re.compile(r"(intersentence|intrasentence)_(.+)")

# VLM result files
QWEN_SS = os.path.join(BASE_DIR, "stereoset", "image_bias_eval_qwen3vl_results_all.csv")
GEMMA_SS = os.path.join(BASE_DIR, "stereoset", "image_bias_eval_gemma4_results_all.csv")
QWEN_CP_PATTERN = os.path.join(BASE_DIR, "crows-pairs", "image_bias_eval_qwen3vl_results_part{}.csv")

# Prompt files (text prompts for image generation)
SS_PROMPTS = os.path.join(BASE_DIR, "stereoset", "stereotype_trigger_results.csv")
CP_PROMPTS = os.path.join(BASE_DIR, "crows-pairs", "crowspairs_trigger_results.csv")
GEMMA_CP_PATTERN = os.path.join(BASE_DIR, "crows-pairs", "image_bias_eval_gemma4_results_part{}.csv")

# Bias type mapping for cross-dataset comparison
BIAS_TYPE_MAP = {
    "gender": "gender",
    "religion": "religion",
    "race": "race-color",
}


# --- Phase 1: Discover cases (filtered to lean_stereotype only) ---

def load_lean_ids():
    """Load the lean_stereotype case IDs from benchmark CSVs."""
    ss_meta = pd.read_csv(STEREOSET_CSV)
    cp_meta = pd.read_csv(CROWSPAIRS_CSV)
    ss_ids = set(zip(ss_meta["split"].astype(str), ss_meta["id"].astype(str)))
    cp_ids = set(cp_meta["id"].astype(str))
    return ss_ids, cp_ids


def discover_stereoset_cases(image_base, lean_ids):
    """Scan StereoSet images, keeping only lean_stereotype cases."""
    work_items = []
    for dirname in sorted(os.listdir(image_base)):
        m = DIR_PATTERN.match(dirname)
        if not m:
            continue
        split, case_id = m.group(1), m.group(2)
        if (split, case_id) not in lean_ids:
            continue
        case_dir = os.path.join(image_base, dirname)
        if not os.path.isdir(case_dir):
            continue
        for seed in SEEDS:
            seed_dir = os.path.join(case_dir, f"seed_{seed}")
            if not os.path.isdir(seed_dir):
                continue
            paths = {}
            valid = True
            for img_type in IMAGE_TYPES:
                p = os.path.join(seed_dir, f"{img_type}.png")
                if os.path.exists(p):
                    paths[img_type] = p
                else:
                    valid = False
                    break
            if valid:
                work_items.append((split, case_id, seed, paths))
    return work_items


def discover_crowspairs_cases(image_base, lean_ids):
    """Scan CrowS-Pairs images, keeping only lean_stereotype cases."""
    work_items = []
    for dirname in sorted(os.listdir(image_base), key=lambda x: int(x) if x.isdigit() else 0):
        if not dirname.isdigit():
            continue
        if dirname not in lean_ids:
            continue
        case_dir = os.path.join(image_base, dirname)
        if not os.path.isdir(case_dir):
            continue
        for seed in SEEDS:
            seed_dir = os.path.join(case_dir, f"seed_{seed}")
            if not os.path.isdir(seed_dir):
                continue
            paths = {}
            valid = True
            for img_type in IMAGE_TYPES:
                p = os.path.join(seed_dir, f"{img_type}.png")
                if os.path.exists(p):
                    paths[img_type] = p
                else:
                    valid = False
                    break
            if valid:
                work_items.append(("crowspairs", dirname, seed, paths))
    return work_items


# --- Phase 2: Compute CLIP similarities ---

def load_completed(output_csv):
    """Load already-completed keys from partial output."""
    completed = set()
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        for _, row in df.iterrows():
            completed.add((row["split"], str(row["id"]), int(row["seed"])))
    return completed


def compute_similarities(work_items, model, processor, device, batch_size, output_csv):
    """Compute CLIP similarities, writing results incrementally."""
    results = []
    flush_every = 1000
    total = len(work_items)
    triplets_per_batch = max(1, batch_size // 3)

    pbar = tqdm(total=total, desc="  CLIP similarity", unit="triplet")

    for batch_start in range(0, total, triplets_per_batch):
        batch_end = min(batch_start + triplets_per_batch, total)
        batch_items = work_items[batch_start:batch_end]

        all_images = []
        for _, _, _, paths in batch_items:
            for img_type in IMAGE_TYPES:
                all_images.append(Image.open(paths[img_type]).convert("RGB"))

        inputs = processor(images=all_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state[:, 0]
            else:
                embeddings = outputs
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        for i, (split, case_id, seed, _) in enumerate(batch_items):
            idx = i * 3
            e_n, e_s, e_a = embeddings[idx], embeddings[idx + 1], embeddings[idx + 2]
            results.append({
                "split": split, "id": case_id, "seed": seed,
                "sim_neutral_stereotype": round((e_n @ e_s).item(), 6),
                "sim_neutral_anti_stereotype": round((e_n @ e_a).item(), 6),
                "sim_stereotype_anti_stereotype": round((e_s @ e_a).item(), 6),
            })

        pbar.update(batch_end - batch_start)

        if len(results) >= flush_every or batch_end == total:
            df = pd.DataFrame(results)
            write_header = not os.path.exists(output_csv)
            df.to_csv(output_csv, mode="a", header=write_header, index=False)
            results = []

    pbar.close()
    print(f"  Saved to {output_csv}")


# --- Phase 3: Load and merge VLM scores ---

def load_vlm_parts(pattern, n_parts=4):
    """Load multi-part VLM result CSVs."""
    parts = []
    for i in range(1, n_parts + 1):
        p = pattern.format(i)
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def pivot_vlm(df):
    """Pivot VLM results: one row per (id, seed) with score per image_type."""
    group_cols = ["id", "seed"]
    if "split" in df.columns:
        group_cols = ["split"] + group_cols

    piv = df.pivot_table(
        index=group_cols, columns="image_type", values="score", aggfunc="mean",
    ).reset_index()
    piv.columns.name = None

    rename = {"neutral": "score_neutral", "stereotype_trigger": "score_stereo",
              "anti_stereotype_trigger": "score_anti"}
    return piv.rename(columns=rename)


def load_prompts(dataset):
    """Load text prompts for image generation."""
    prompt_cols = ["neutral", "stereotype_trigger", "anti_stereotype_trigger"]
    rename = {c: f"prompt_{c}" for c in prompt_cols}

    if dataset == "StereoSet":
        df = pd.read_csv(SS_PROMPTS)
        df["id"] = df["id"].astype(str)
        dedup = df.drop_duplicates(subset=["split", "id"])
        return dedup[["split", "id"] + prompt_cols].rename(columns=rename)
    else:
        df = pd.read_csv(CP_PROMPTS)
        df["id"] = df["id"].astype(str)
        dedup = df.drop_duplicates(subset=["id"])
        return dedup[["id"] + prompt_cols].rename(columns=rename)


def merge_all(clip_csv, meta_csv, qwen_piv, gemma_piv, dataset, has_split=True):
    """Merge CLIP + metadata + Qwen + Gemma + prompts into one DataFrame."""
    clip_df = pd.read_csv(clip_csv)
    meta_df = pd.read_csv(meta_csv)

    clip_df["id"] = clip_df["id"].astype(str)
    meta_df["id"] = meta_df["id"].astype(str)
    qwen_piv["id"] = qwen_piv["id"].astype(str)
    gemma_piv["id"] = gemma_piv["id"].astype(str)

    # De-dup metadata
    dedup_keys = ["split", "id"] if has_split else ["id"]
    meta_dedup = meta_df.drop_duplicates(subset=dedup_keys)

    # Merge CLIP + metadata
    join_keys = ["split", "id"] if has_split else ["id"]
    merged = clip_df.merge(meta_dedup, on=join_keys, how="inner")

    # Merge Qwen scores
    qwen_join = (["split", "id", "seed"] if has_split and "split" in qwen_piv.columns
                 else ["id", "seed"])
    qwen_renamed = qwen_piv.rename(columns={
        "score_neutral": "qwen_neutral", "score_stereo": "qwen_stereo", "score_anti": "qwen_anti"})
    merged = merged.merge(qwen_renamed, on=qwen_join, how="left")

    # Merge Gemma scores
    gemma_join = (["split", "id", "seed"] if has_split and "split" in gemma_piv.columns
                  else ["id", "seed"])
    gemma_renamed = gemma_piv.rename(columns={
        "score_neutral": "gemma_neutral", "score_stereo": "gemma_stereo", "score_anti": "gemma_anti"})
    merged = merged.merge(gemma_renamed, on=gemma_join, how="left")

    # Merge text prompts
    prompts = load_prompts(dataset)
    prompt_join = ["split", "id"] if has_split else ["id"]
    merged = merged.merge(prompts, on=prompt_join, how="left")

    merged["dataset"] = dataset
    return merged


# --- Phase 4: Analysis ---

SIM_COLS = ["sim_neutral_stereotype", "sim_neutral_anti_stereotype",
            "sim_stereotype_anti_stereotype"]
SIM_LABELS = ["sim(neutral, stereo)", "sim(neutral, anti)", "sim(stereo, anti)"]


def per_dataset_analysis(df, dataset_name, output_dir, log):
    """Per-dataset CLIP similarity analysis."""
    log(f"\n{'='*60}")
    log(f"  {dataset_name}")
    log(f"{'='*60}")
    log(f"  Samples: {len(df)}, unique cases: {df['id'].nunique()}")

    for col, label in zip(SIM_COLS, SIM_LABELS):
        vals = df[col].dropna()
        log(f"    {label}: mean={vals.mean():.4f}, std={vals.std():.4f}, median={vals.median():.4f}")

    diff = df["sim_neutral_stereotype"] - df["sim_neutral_anti_stereotype"]
    n_pos = (diff > 0).sum()
    t_stat, t_pval = stats.ttest_rel(df["sim_neutral_stereotype"], df["sim_neutral_anti_stereotype"])
    d = diff.mean() / diff.std() if diff.std() > 0 else 0
    log(f"\n    Neutral lean stereotype: {n_pos}/{len(diff)} ({100*n_pos/len(diff):.1f}%)")
    log(f"    Mean gap: {diff.mean():.6f}, t={t_stat:.2f}, p={t_pval:.2e}, Cohen's d={d:.4f}")

    log(f"\n    By bias type:")
    for bt in sorted(df["bias_type"].unique()):
        sub = df[df["bias_type"] == bt]
        gap = sub["sim_neutral_stereotype"].mean() - sub["sim_neutral_anti_stereotype"].mean()
        pct = 100 * (sub["sim_neutral_stereotype"] > sub["sim_neutral_anti_stereotype"]).sum() / len(sub)
        log(f"      {bt} (n={len(sub)}): gap={gap:+.4f}, lean_S={pct:.1f}%")

    # Plots
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, label in zip(SIM_COLS, SIM_LABELS):
        ax.hist(df[col].dropna(), bins=80, alpha=0.5, label=label, density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset_name}: CLIP Similarity Distributions")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plots", f"similarity_distributions_{dataset_name.lower()}.png"), dpi=150)
    plt.close(fig)

    if "bias_type" in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, col, label in zip(axes, SIM_COLS, SIM_LABELS):
            df[["bias_type", col]].dropna().boxplot(column=col, by="bias_type", ax=ax)
            ax.set_title(label)
            ax.set_xlabel("Bias Type")
            ax.set_ylabel("Cosine Similarity")
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle(f"{dataset_name}: CLIP Similarity by Bias Type", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "plots", f"boxplot_by_bias_type_{dataset_name.lower()}.png"), dpi=150)
        plt.close(fig)


def cross_dataset_analysis(ss_df, cp_df, output_dir, log):
    """Cross-dataset CLIP similarity comparison."""
    log(f"\n{'='*60}")
    log(f"  CROSS-DATASET COMPARISON")
    log(f"{'='*60}")

    # Distribution overlay with KS-test
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, label in zip(axes, SIM_COLS, SIM_LABELS):
        ss_vals, cp_vals = ss_df[col].dropna(), cp_df[col].dropna()
        ax.hist(ss_vals, bins=60, alpha=0.5, label=f"StereoSet (n={len(ss_vals)})",
                density=True, color="steelblue")
        ax.hist(cp_vals, bins=60, alpha=0.5, label=f"CrowS-Pairs (n={len(cp_vals)})",
                density=True, color="coral")
        ks_stat, ks_pval = stats.ks_2samp(ss_vals, cp_vals)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.text(0.05, 0.95, f"KS={ks_stat:.3f}\np={ks_pval:.1e}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        log(f"    {label}: KS={ks_stat:.4f}, p={ks_pval:.2e}")

    fig.suptitle("Cross-Dataset: CLIP Similarity Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plots", "cross_dataset_distribution_overlay.png"), dpi=150)
    plt.close(fig)

    # Bias-type gap comparison (overlapping types)
    ss_norm = ss_df.copy()
    ss_norm["bias_type_norm"] = ss_norm["bias_type"].map(BIAS_TYPE_MAP)
    ss_norm = ss_norm.dropna(subset=["bias_type_norm"])
    cp_norm = cp_df.copy()
    cp_norm["bias_type_norm"] = cp_norm["bias_type"]
    overlap = sorted(set(ss_norm["bias_type_norm"]) & set(cp_norm["bias_type_norm"]))

    log(f"\n    Overlapping bias types: {overlap}")
    for bt in overlap:
        ss_sub = ss_norm[ss_norm["bias_type_norm"] == bt]
        cp_sub = cp_norm[cp_norm["bias_type_norm"] == bt]
        ss_gap = ss_sub["sim_neutral_stereotype"].mean() - ss_sub["sim_neutral_anti_stereotype"].mean()
        cp_gap = cp_sub["sim_neutral_stereotype"].mean() - cp_sub["sim_neutral_anti_stereotype"].mean()
        same = "YES" if np.sign(ss_gap) == np.sign(cp_gap) else "NO"
        log(f"      {bt}: SS gap={ss_gap:+.4f} (n={len(ss_sub)}), "
            f"CP gap={cp_gap:+.4f} (n={len(cp_sub)}), same dir: {same}")

    # Overall summary
    for name, df in [("StereoSet", ss_df), ("CrowS-Pairs", cp_df)]:
        diff = df["sim_neutral_stereotype"] - df["sim_neutral_anti_stereotype"]
        d = diff.mean() / diff.std() if diff.std() > 0 else 0
        log(f"\n    {name}: mean gap={diff.mean():.6f}, Cohen's d={d:.4f}")


def three_way_analysis(combined, output_dir, log):
    """Three-way agreement analysis: CLIP, Qwen-VL, Gemma4."""
    log(f"\n{'='*60}")
    log(f"  THREE-WAY AGREEMENT: CLIP + Qwen-VL + Gemma4")
    log(f"{'='*60}")

    # Compute lean metrics
    df = combined.copy()
    df["clip_lean"] = df["sim_neutral_stereotype"] - df["sim_neutral_anti_stereotype"]
    df["qwen_lean"] = abs(df["qwen_neutral"] - df["qwen_anti"]) - abs(df["qwen_neutral"] - df["qwen_stereo"])
    df["gemma_lean"] = abs(df["gemma_neutral"] - df["gemma_anti"]) - abs(df["gemma_neutral"] - df["gemma_stereo"])

    n = len(df)
    log(f"\n    Total samples: {n}")

    # Per-method lean stats
    log(f"\n    {'Method':10s}  {'Lean S%':>8s}  {'Mean':>10s}  {'t':>9s}  {'p':>12s}  {'d':>7s}")
    log(f"    {'-'*10}  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*12}  {'-'*7}")
    for col, name in [("clip_lean", "CLIP"), ("qwen_lean", "Qwen-VL"), ("gemma_lean", "Gemma4")]:
        vals = df[col].dropna()
        n_pos = (vals > 0).sum()
        t, p = stats.ttest_1samp(vals, 0)
        d = vals.mean() / vals.std() if vals.std() > 0 else 0
        log(f"    {name:10s}  {100*n_pos/len(vals):7.1f}%  {vals.mean():+10.4f}  {t:+9.2f}  {p:12.2e}  {d:+7.3f}")

    # Agreement
    df["n_agree"] = ((df["clip_lean"] > 0).astype(int) +
                     (df["qwen_lean"] > 0).astype(int) +
                     (df["gemma_lean"] > 0).astype(int))
    log(f"\n    All 3 agree lean stereo: {(df['n_agree']==3).sum()}/{n} ({100*(df['n_agree']==3).sum()/n:.1f}%)")
    log(f"    At least 2 agree:        {(df['n_agree']>=2).sum()}/{n} ({100*(df['n_agree']>=2).sum()/n:.1f}%)")

    # Case-level
    case_agg = df.groupby(["dataset", "id", "bias_type"]).agg(
        clip_lean=("clip_lean", "mean"),
        qwen_lean=("qwen_lean", "mean"),
        gemma_lean=("gemma_lean", "mean"),
    ).reset_index()
    n_cases = len(case_agg)
    all3 = ((case_agg["clip_lean"] > 0) & (case_agg["qwen_lean"] > 0) & (case_agg["gemma_lean"] > 0)).sum()

    log(f"\n    Case-level (n={n_cases}):")
    for col, name in [("clip_lean", "CLIP"), ("qwen_lean", "Qwen-VL"), ("gemma_lean", "Gemma4")]:
        v = case_agg[col].dropna()
        pct = 100 * (v > 0).sum() / len(v)
        t, p = stats.ttest_1samp(v, 0)
        d = v.mean() / v.std() if v.std() > 0 else 0
        log(f"      {name:10s}: lean_S={pct:.1f}%, mean={v.mean():+.4f}, d={d:+.3f}")
    log(f"      All 3 agree: {all3}/{n_cases} ({100*all3/n_cases:.1f}%)")

    # Pairwise correlations
    log(f"\n    Pairwise Pearson correlations (sample-level):")
    pairs = [("clip_lean", "qwen_lean", "CLIP vs Qwen-VL"),
             ("clip_lean", "gemma_lean", "CLIP vs Gemma4"),
             ("qwen_lean", "gemma_lean", "Qwen-VL vs Gemma4")]
    for c1, c2, label in pairs:
        valid = df[[c1, c2]].dropna()
        r, p = stats.pearsonr(valid[c1], valid[c2])
        rs, ps = stats.spearmanr(valid[c1], valid[c2])
        log(f"      {label:20s}: Pearson r={r:+.4f} (p={p:.2e}), Spearman r={rs:+.4f} (p={ps:.2e})")

    log(f"\n    Case-level Pearson correlations:")
    for c1, c2, label in pairs:
        valid = case_agg[[c1, c2]].dropna()
        r, p = stats.pearsonr(valid[c1], valid[c2])
        log(f"      {label:20s}: r={r:+.4f} (p={p:.2e})")

    # Per bias type
    log(f"\n    {'bias_type':25s} {'n':>5s}  {'CLIP%':>6s}  {'Qwen%':>6s}  {'Gem%':>6s}  "
        f"{'CLIP_d':>7s}  {'Qwen_d':>7s}  {'Gem_d':>7s}")
    log(f"    {'-'*25} {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}")
    for bt in sorted(df["bias_type"].dropna().unique()):
        sub = df[df["bias_type"] == bt]
        if len(sub) < 10:
            continue
        c_pct = 100 * (sub["clip_lean"] > 0).sum() / len(sub)
        q_pct = 100 * (sub["qwen_lean"] > 0).sum() / len(sub)
        g_pct = 100 * (sub["gemma_lean"] > 0).sum() / len(sub)
        c_d = sub["clip_lean"].mean() / sub["clip_lean"].std() if sub["clip_lean"].std() > 0 else 0
        q_d = sub["qwen_lean"].mean() / sub["qwen_lean"].std() if sub["qwen_lean"].std() > 0 else 0
        g_d = sub["gemma_lean"].mean() / sub["gemma_lean"].std() if sub["gemma_lean"].std() > 0 else 0
        log(f"    {bt:25s} {len(sub):5d}  {c_pct:5.1f}%  {q_pct:5.1f}%  {g_pct:5.1f}%  "
            f"{c_d:+7.3f}  {q_d:+7.3f}  {g_d:+7.3f}")

    # VLM individual score agreement
    log(f"\n    VLM score agreement (Qwen vs Gemma):")
    for sc, label in [("stereo", "stereotype"), ("neutral", "neutral"), ("anti", "anti-stereotype")]:
        c1, c2 = f"qwen_{sc}", f"gemma_{sc}"
        valid = df[[c1, c2]].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid[c1], valid[c2])
            log(f"      {label:20s}: Pearson r={r:+.4f} (n={len(valid)})")

    # Correlation heatmap
    corr_cols = ["clip_lean", "qwen_lean", "gemma_lean",
                 "qwen_stereo", "qwen_neutral", "qwen_anti",
                 "gemma_stereo", "gemma_neutral", "gemma_anti"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr_data = df[corr_cols].dropna()
    if len(corr_data) > 10:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = corr_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                    vmin=-1, vmax=1, mask=mask, ax=ax)
        ax.set_title("Lean Stereotype Benchmark: Correlation Heatmap")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "plots", "correlation_heatmap.png"), dpi=150)
        plt.close(fig)

    # Three-method agreement bar chart by bias type
    bt_data = []
    for bt in sorted(df["bias_type"].dropna().unique()):
        sub = df[df["bias_type"] == bt]
        if len(sub) < 10:
            continue
        bt_data.append({
            "bias_type": bt,
            "CLIP": 100 * (sub["clip_lean"] > 0).sum() / len(sub),
            "Qwen-VL": 100 * (sub["qwen_lean"] > 0).sum() / len(sub),
            "Gemma4": 100 * (sub["gemma_lean"] > 0).sum() / len(sub),
        })
    if bt_data:
        bt_df = pd.DataFrame(bt_data)
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(bt_df))
        w = 0.25
        ax.bar(x - w, bt_df["CLIP"], w, label="CLIP", alpha=0.8)
        ax.bar(x, bt_df["Qwen-VL"], w, label="Qwen-VL", alpha=0.8)
        ax.bar(x + w, bt_df["Gemma4"], w, label="Gemma4", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(bt_df["bias_type"], rotation=30, ha="right")
        ax.set_ylabel("% Neutral Leans Stereotype")
        ax.set_title("Three-Method Agreement: Neutral Lean Stereotype by Bias Type")
        ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "plots", "three_way_agreement_by_bias_type.png"), dpi=150)
        plt.close(fig)

    return df


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Lean Stereotype Benchmark: CLIP + VLM analysis")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "stereoimage"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    ss_clip_csv = os.path.join(args.output_dir, "data", "clip_similarities_stereoset.csv")
    cp_clip_csv = os.path.join(args.output_dir, "data", "clip_similarities_crowspairs.csv")

    # Phase 1: Discover lean_stereotype cases only
    print("Phase 1: Loading lean_stereotype IDs and discovering cases...")
    ss_lean_ids, cp_lean_ids = load_lean_ids()
    ss_items = discover_stereoset_cases(STEREOSET_IMAGE_BASE, ss_lean_ids)
    cp_items = discover_crowspairs_cases(CROWSPAIRS_IMAGE_BASE, cp_lean_ids)
    print(f"  StereoSet: {len(ss_items)} triplets (from {len(ss_lean_ids)} lean cases)")
    print(f"  CrowS-Pairs: {len(cp_items)} triplets (from {len(cp_lean_ids)} lean cases)")

    # Phase 2: Compute CLIP similarities
    ss_todo, cp_todo = ss_items, cp_items
    if args.resume:
        ss_done = load_completed(ss_clip_csv)
        cp_done = load_completed(cp_clip_csv)
        ss_todo = [x for x in ss_items if (x[0], x[1], x[2]) not in ss_done]
        cp_todo = [x for x in cp_items if (x[0], x[1], x[2]) not in cp_done]
        print(f"  Resume: SS {len(ss_todo)} remaining, CP {len(cp_todo)} remaining")

    if ss_todo or cp_todo:
        print(f"\nPhase 2: Computing CLIP similarities on {device}...")
        model = CLIPModel.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
        model = model.to(device).half().eval()
        print(f"  Model loaded on {device}")

        if ss_todo:
            print(f"\n  --- StereoSet ---")
            compute_similarities(ss_todo, model, processor, device, args.batch_size, ss_clip_csv)
        if cp_todo:
            print(f"\n  --- CrowS-Pairs ---")
            compute_similarities(cp_todo, model, processor, device, args.batch_size, cp_clip_csv)

        del model, processor
        torch.cuda.empty_cache()
    else:
        print("All CLIP similarities already computed, skipping Phase 2")

    # Phase 3: Load VLM scores and merge everything
    print("\nPhase 3: Loading VLM scores and merging...")
    qwen_ss_piv = pivot_vlm(pd.read_csv(QWEN_SS))
    gemma_ss_piv = pivot_vlm(pd.read_csv(GEMMA_SS))
    qwen_cp_piv = pivot_vlm(load_vlm_parts(QWEN_CP_PATTERN, n_parts=5))
    gemma_cp_piv = pivot_vlm(load_vlm_parts(GEMMA_CP_PATTERN))

    ss_merged = merge_all(ss_clip_csv, STEREOSET_CSV, qwen_ss_piv, gemma_ss_piv,
                          "StereoSet", has_split=True)
    cp_merged = merge_all(cp_clip_csv, CROWSPAIRS_CSV, qwen_cp_piv, gemma_cp_piv,
                          "CrowS-Pairs", has_split=False)

    combined = pd.concat([ss_merged, cp_merged], ignore_index=True)
    print(f"  Combined: {len(combined)} samples (SS={len(ss_merged)}, CP={len(cp_merged)})")

    ss_merged.to_csv(os.path.join(args.output_dir, "data", "merged_stereoset.csv"), index=False)
    cp_merged.to_csv(os.path.join(args.output_dir, "data", "merged_crowspairs.csv"), index=False)
    combined.to_csv(os.path.join(args.output_dir, "data", "merged_all.csv"), index=False)

    if args.skip_plots:
        print("Skipping analysis (--skip-plots)")
        return

    # Phase 4: Analysis
    summary_lines = []
    def log(msg):
        print(msg)
        summary_lines.append(msg)

    print("\nPhase 4: Per-dataset CLIP analysis...")
    per_dataset_analysis(ss_merged, "StereoSet", args.output_dir, log)
    per_dataset_analysis(cp_merged, "CrowS-Pairs", args.output_dir, log)

    print("\nPhase 5: Cross-dataset comparison...")
    cross_dataset_analysis(ss_merged, cp_merged, args.output_dir, log)

    print("\nPhase 6: Three-way agreement analysis...")
    result_df = three_way_analysis(combined, args.output_dir, log)

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary_stats.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSummary saved to {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()
