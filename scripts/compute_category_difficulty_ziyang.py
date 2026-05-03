#!/usr/bin/env python3
"""Per-category difficulty / coverage analysis.

For each bias category, compute Hedges' g between stereotype-arm and
anti-stereotype-arm VLM scores, with cluster-bootstrap 95% CI at the case_id
level. Joins LLM judge-agreement rate and writes a ranked plot.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_BASE = "/data/gpfs/projects/punim2888/stereoimage"
MERGED_CSV = f"{_BASE}/data/merged_all.csv"
AGREEMENT_CSV = f"{_BASE}/reports/agreement_by_bias_type.csv"
OUT_PLOT = f"{_BASE}/plots/category_difficulty_ranking_ziyang.png"

EVALUATORS = ["qwen", "gemma"]
ARMS = ["neutral", "stereo", "anti"]


def hedges_g(s_vals, a_vals):
    """Hedges' g (Cohen's d with small-sample correction) for S vs A."""
    s = np.asarray(s_vals, dtype=float)
    a = np.asarray(a_vals, dtype=float)
    n_s, n_a = len(s), len(a)
    if n_s < 2 or n_a < 2:
        return np.nan
    var_s = s.var(ddof=1)
    var_a = a.var(ddof=1)
    pooled_sd = np.sqrt(((n_s - 1) * var_s + (n_a - 1) * var_a) / (n_s + n_a - 2))
    if pooled_sd == 0:
        return np.nan
    d = (s.mean() - a.mean()) / pooled_sd
    df = n_s + n_a - 2
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return d * j


def cluster_bootstrap_g(df_cat, evaluators, n_boot=1000, seed=0):
    """Cluster bootstrap on case_id. Returns (g_point, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    case_ids = df_cat["id"].unique()

    def _g_from_df(d):
        s_vals, a_vals = [], []
        for ev in evaluators:
            s_col, a_col = f"{ev}_stereo", f"{ev}_anti"
            mask = d[s_col].notna() & d[a_col].notna()
            s_vals.append(d.loc[mask, s_col].to_numpy())
            a_vals.append(d.loc[mask, a_col].to_numpy())
        s = np.concatenate(s_vals)
        a = np.concatenate(a_vals)
        return hedges_g(s, a)

    g_point = _g_from_df(df_cat)

    by_id = {cid: g for cid, g in df_cat.groupby("id")}
    boot_gs = np.empty(n_boot)
    n_ids = len(case_ids)
    for b in range(n_boot):
        sample_ids = rng.choice(case_ids, size=n_ids, replace=True)
        sample_df = pd.concat([by_id[cid] for cid in sample_ids], ignore_index=True)
        boot_gs[b] = _g_from_df(sample_df)
    boot_gs = boot_gs[~np.isnan(boot_gs)]
    ci_low, ci_high = np.percentile(boot_gs, [2.5, 97.5])
    return g_point, ci_low, ci_high


def assign_tier(g):
    """Cohen's d conventions: ≥2.0 'huge', ≥1.5 'very large', <1.5 'large/medium'."""
    if g >= 2.0:
        return "Strong"
    if g >= 1.5:
        return "Medium"
    return "Weak"


TIER_COLOR = {"Strong": "#2e7d32", "Medium": "#f9a825", "Weak": "#c62828"}
BAR_HEIGHT = 0.55
TICK_LABEL_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 15
LEGEND_FONTSIZE = 11
BAR_VALUE_FONTSIZE = 12


def main(n_boot: int):
    df = pd.read_csv(MERGED_CSV)
    needed = [f"{ev}_{arm}" for ev in EVALUATORS for arm in ARMS]
    print(f"Loaded {len(df)} rows from merged_all.csv")
    for c in needed:
        print(f"  {c}: nonnull={df[c].notna().sum()}")

    agree_df = pd.read_csv(AGREEMENT_CSV)
    agree_df = agree_df[agree_df["approach"] == "matching"]
    judge_agreement = (
        agree_df.groupby("bias_type")["accuracy"].mean().rename("judge_agreement")
    )

    rows = []
    for bias_type, df_cat in df.groupby("bias_type"):
        n_units = df_cat["id"].nunique()
        n_obs = (
            df_cat[["qwen_stereo", "gemma_stereo"]].notna().sum().sum()
            + df_cat[["qwen_anti", "gemma_anti"]].notna().sum().sum()
        )

        g_pooled, ci_low, ci_high = cluster_bootstrap_g(
            df_cat, EVALUATORS, n_boot=n_boot, seed=0
        )

        per_eval = {}
        for ev in EVALUATORS:
            mask = df_cat[f"{ev}_stereo"].notna() & df_cat[f"{ev}_anti"].notna()
            per_eval[f"hedges_g_{ev}"] = hedges_g(
                df_cat.loc[mask, f"{ev}_stereo"], df_cat.loc[mask, f"{ev}_anti"]
            )

        means = {}
        for arm in ARMS:
            vals = pd.concat(
                [df_cat[f"{ev}_{arm}"].dropna() for ev in EVALUATORS],
                ignore_index=True,
            )
            means[f"avg_{arm}"] = vals.mean()

        rows.append(
            dict(
                bias_type=bias_type,
                n_units=n_units,
                n_obs=int(n_obs),
                hedges_g_pooled=g_pooled,
                ci_low=ci_low,
                ci_high=ci_high,
                **per_eval,
                **means,
                bias_amp=means["avg_stereo"] - means["avg_neutral"],
                total_sep=means["avg_stereo"] - means["avg_anti"],
                judge_agreement=judge_agreement.get(bias_type, np.nan),
                tier=assign_tier(g_pooled),
            )
        )

    out = pd.DataFrame(rows).sort_values(
        "hedges_g_pooled", ascending=False
    ).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6.5))
    plot_df = out.iloc[::-1].reset_index(drop=True)  # ascending for top-down bars
    colors = [TIER_COLOR[t] for t in plot_df["tier"]]
    y = np.arange(len(plot_df))
    err_low = plot_df["hedges_g_pooled"] - plot_df["ci_low"]
    err_high = plot_df["ci_high"] - plot_df["hedges_g_pooled"]
    bars = ax.barh(
        y,
        plot_df["hedges_g_pooled"],
        xerr=[err_low, err_high],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
        height=BAR_HEIGHT,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{bt} (n={n})" for bt, n in zip(plot_df["bias_type"], plot_df["n_units"])],
        fontsize=TICK_LABEL_FONTSIZE,
    )
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel(
        "Hedges' g  (stereotype vs anti-stereotype, pooled evaluators)\n"
        "← harder to evaluate          easier to evaluate →",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.set_title(
        "Per-Category Benchmark Difficulty\n"
        "Lower g = bias is harder to measure (signal visually compressed)",
        fontsize=TITLE_FONTSIZE,
    )

    x_max = plot_df["ci_high"].max()
    label_offset = max(0.03 * x_max, 0.05)
    ax.set_xlim(right=x_max + label_offset * 3.0)
    for bar, g, ci_high in zip(bars, plot_df["hedges_g_pooled"], plot_df["ci_high"]):
        ax.text(
            ci_high + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{g:.2f}",
            va="center",
            ha="left",
            fontsize=BAR_VALUE_FONTSIZE,
        )

    ax.axvline(1.5, color="grey", linestyle="--", linewidth=0.7)
    ax.axvline(2.0, color="grey", linestyle="--", linewidth=0.7)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=TIER_COLOR[t]) for t in ["Strong", "Medium", "Weak"]
    ]
    ax.legend(
        legend_handles,
        [
            "Strong (g≥2.0) — easy to evaluate",
            "Medium (1.5≤g<2.0)",
            "Weak (g<1.5) — difficult to evaluate",
        ],
        loc="lower right",
        fontsize=LEGEND_FONTSIZE,
    )
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    print(f"Wrote {OUT_PLOT}")

    # Markdown table to stdout
    print("\n## Ranked table\n")
    cols = ["rank", "bias_type", "n_units", "hedges_g_pooled", "ci_low", "ci_high",
            "bias_amp", "total_sep", "judge_agreement", "tier"]
    headers = ["Rank", "Bias Type", "N", "Hedges' g", "CI low", "CI high",
               "S−N", "S−A", "Judge Agr.", "Tier"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for _, r in out.iterrows():
        print(
            f"| {r['rank']} | {r['bias_type']} | {r['n_units']} | "
            f"{r['hedges_g_pooled']:.3f} | {r['ci_low']:.3f} | {r['ci_high']:.3f} | "
            f"{r['bias_amp']:+.3f} | {r['total_sep']:.3f} | "
            f"{r['judge_agreement']*100:.1f}% | {r['tier']} |"
        )

    # Cross-evaluator Spearman sanity
    rho = out[["hedges_g_qwen", "hedges_g_gemma"]].corr(method="spearman").iloc[0, 1]
    print(f"\nCross-evaluator Spearman ρ (qwen rank vs gemma rank): {rho:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()
    main(n_boot=args.n_boot)
