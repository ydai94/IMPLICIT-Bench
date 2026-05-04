"""Beyond-correlation human-vs-VLM analysis: where do they actually disagree?

Reads:
  data/human_eval/manifest.csv              + Round-1 form responses CSV
  data/human_eval_round2/manifest.csv       + Round-2 form responses CSV

Writes:
  reports/human_vlm_divergence.md
  data/human_vlm_divergence_examples.csv

Reuses the loader from analyze_human_eval.py, so this report uses the
same 12 raters per round.

Reports:
  - Per-image table: human mean, Qwen3-VL, Gemma-4, signed deltas.
  - Top-N divergence cases per VLM (default |delta| > 1.5, N = 10).
  - Cross-tab of mean |delta| by bias type and condition.
  - Direction-flip count: cases where human consensus picks one direction
    on stereotype vs anti-stereotype but the VLM picks the other.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = "/data/gpfs/projects/punim2888/stereoimage"
sys.path.insert(0, f"{ROOT}/scripts")

from analyze_human_eval import (  # noqa: E402
    ROUNDS,
    DROPPED_RATERS,
    long_form,
)
from scipy.stats import pearsonr  # noqa: E402

OUT_MD = f"{ROOT}/reports/human_vlm_divergence.md"
OUT_CSV = f"{ROOT}/data/human_vlm_divergence_examples.csv"

DELTA_THRESHOLD = 1.5      # Cases with |human_mean - VLM| > this go in top-N.
TOP_N = 10                  # Per-VLM divergence cases listed.


def md_table(df: pd.DataFrame, fmt=None) -> str:
    fmt = fmt or {}

    def cell(c, v):
        if pd.isna(v):
            return ""
        f = fmt.get(c)
        if f is not None and isinstance(v, (int, float, np.floating)):
            return f.format(v)
        return str(v)

    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(cell(c, r[c]) for c in cols) + " |"
            for _, r in df.iterrows()]
    return "\n".join([head, sep] + body)


def main():
    long = pd.concat([long_form(r["name"], r["manifest"], r["responses"])
                      for r in ROUNDS], ignore_index=True)
    long = long[~long["rater"].isin(DROPPED_RATERS)].reset_index(drop=True)
    long["rating"] = pd.to_numeric(long["rating"], errors="coerce")
    long["vlm_qwen"] = pd.to_numeric(long["vlm_qwen"], errors="coerce")
    long["vlm_gemma"] = pd.to_numeric(long["vlm_gemma"], errors="coerce")

    ir = long[long["kind"] == "image_rating"].copy()

    # Pull bias_type + condition from the manifest via the long DataFrame.
    # Each (round, case_id, condition) cell already carries vlm_qwen and
    # vlm_gemma (manifest-side, identical across raters). Keep one row per
    # image and aggregate human ratings.
    per_image = (ir.groupby(["round", "case_id", "condition", "bias_type"])
                 .agg(human_mean=("rating", "mean"),
                      human_std=("rating", "std"),
                      n_raters=("rating", "count"),
                      vlm_qwen=("vlm_qwen", "first"),
                      vlm_gemma=("vlm_gemma", "first"))
                 .reset_index())
    per_image = per_image.dropna(subset=["human_mean",
                                         "vlm_qwen", "vlm_gemma"])
    per_image["delta_qwen"] = per_image["human_mean"] - per_image["vlm_qwen"]
    per_image["delta_gemma"] = per_image["human_mean"] - per_image["vlm_gemma"]
    per_image["abs_delta_qwen"] = per_image["delta_qwen"].abs()
    per_image["abs_delta_gemma"] = per_image["delta_gemma"].abs()

    # Pull prompts from the round manifests so the example tables can carry
    # the actual prompt text alongside scores. Manifests have prompt_neutral
    # but not the trigger texts; for trigger arms we need to join through
    # benchmark_prompts.csv on (case_id == id).
    bench = pd.read_csv(f"{ROOT}/data/benchmark_prompts.csv")
    bench = bench[["id", "source", "bias_type",
                   "prompt_neutral", "prompt_stereotype",
                   "prompt_anti_stereotype"]].rename(columns={"id": "case_id"})
    per_image = per_image.merge(
        bench.drop(columns=["bias_type"]),
        on="case_id", how="left")

    def prompt_for_condition(row):
        c = row["condition"]
        if c == "neutral":
            return row["prompt_neutral"]
        if c == "stereotype_trigger":
            return row["prompt_stereotype"]
        if c == "anti_stereotype_trigger":
            return row["prompt_anti_stereotype"]
        return None
    per_image["prompt"] = per_image.apply(prompt_for_condition, axis=1)

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    out: list[str] = []
    out.append("# Human vs VLM Divergence")
    out.append("")
    n_rounds = per_image["round"].nunique()
    n_imgs = len(per_image)
    n_cases = per_image["case_id"].nunique()
    out.append(
        f"Constructed from `{ROOT.split('/')[-1]}/data/human_eval{{,_round2}}/` "
        f"using the same loader as `analyze_human_eval.py` (12 raters per "
        f"round). Rows below: one per (round, case_id, condition) image, "
        f"with `human_mean` averaged across the round's raters.")
    out.append("")
    out.append(
        f"- Rounds: {n_rounds}; cases: {n_cases}; images with both VLM "
        f"scores: **{n_imgs}**.")
    out.append(
        f"- Pearson r (per-image): "
        f"human vs Qwen3-VL = "
        f"{pearsonr(per_image.human_mean, per_image.vlm_qwen).statistic:.3f}, "
        f"human vs Gemma-4  = "
        f"{pearsonr(per_image.human_mean, per_image.vlm_gemma).statistic:.3f}.")
    out.append(
        f"- Mean |human - Qwen3-VL| = "
        f"{per_image.abs_delta_qwen.mean():.3f}; "
        f"mean |human - Gemma-4| = "
        f"{per_image.abs_delta_gemma.mean():.3f}.")
    out.append("")

    # ------------------------------------------------------------------
    # 1. Divergence histogram (binned)
    # ------------------------------------------------------------------
    out.append("## 1. Divergence distribution")
    out.append("")
    bins = [-5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5]
    labels = ["[-5,-3)", "[-3,-2)", "[-2,-1)", "[-1,-0.5)",
              "[-0.5,0.5)", "[0.5,1)", "[1,2)", "[2,3)", "[3,5]"]
    rows = []
    for vlm, col in [("Qwen3-VL", "delta_qwen"), ("Gemma-4", "delta_gemma")]:
        cuts = pd.cut(per_image[col], bins=bins, labels=labels,
                      include_lowest=True)
        counts = cuts.value_counts().reindex(labels, fill_value=0)
        row = {"VLM": vlm}
        for lab in labels:
            row[lab] = int(counts[lab])
        rows.append(row)
    cols = ["VLM"] + labels
    out.append(md_table(pd.DataFrame(rows), fmt={}))
    out.append("")
    out.append(
        "Bins are signed (`human - VLM`). Negative = humans rate the image as "
        "less stereotypical than the VLM does; positive = humans rate higher. "
        "The mass concentrated in `[-0.5, 0.5)` matches the headline Pearson "
        "correlations (>= 0.76) reported in `human_eval_summary.md`.")
    out.append("")

    # ------------------------------------------------------------------
    # 2. Per-bias-type and per-condition disagreement
    # ------------------------------------------------------------------
    out.append("## 2. Mean |delta| by bias type and condition")
    out.append("")
    grp = (per_image
           .groupby(["bias_type", "condition"])
           .agg(n=("case_id", "size"),
                mean_abs_delta_qwen=("abs_delta_qwen", "mean"),
                mean_abs_delta_gemma=("abs_delta_gemma", "mean"))
           .reset_index())
    out.append(md_table(
        grp, fmt={"mean_abs_delta_qwen": "{:.2f}",
                  "mean_abs_delta_gemma": "{:.2f}"}))
    out.append("")
    out.append(
        "High-|delta| cells are the cells where a future user should "
        "interpret VLM scores cautiously. The neutral arm tends to have "
        "larger gaps than the stereotype-trigger arm, consistent with "
        "neutral images being visually ambiguous: humans may infer "
        "stereotype cues from background context that the VLM does not "
        "encode at the same granularity.")
    out.append("")

    out.append("## 3. Mean |delta| per bias type (all conditions pooled)")
    out.append("")
    bt = (per_image
          .groupby("bias_type")
          .agg(n=("case_id", "size"),
               mean_abs_delta_qwen=("abs_delta_qwen", "mean"),
               mean_abs_delta_gemma=("abs_delta_gemma", "mean"),
               pearson_qwen=("human_mean",
                             lambda s: pearsonr(s, per_image.loc[s.index,
                                                                  "vlm_qwen"]).statistic
                             if len(s) >= 3 else np.nan),
               pearson_gemma=("human_mean",
                              lambda s: pearsonr(s, per_image.loc[s.index,
                                                                   "vlm_gemma"]).statistic
                              if len(s) >= 3 else np.nan))
          .reset_index()
          .sort_values("mean_abs_delta_qwen", ascending=False))
    out.append(md_table(
        bt, fmt={"mean_abs_delta_qwen": "{:.2f}",
                 "mean_abs_delta_gemma": "{:.2f}",
                 "pearson_qwen": "{:.3f}",
                 "pearson_gemma": "{:.3f}"}))
    out.append("")

    # ------------------------------------------------------------------
    # 4. Top divergence cases per VLM
    # ------------------------------------------------------------------
    def shorten(s, n=80):
        if not isinstance(s, str):
            return ""
        s = s.replace("\n", " ").replace("|", "/").strip()
        return (s[:n - 3] + "...") if len(s) > n else s

    for vlm, col, abs_col in [("Qwen3-VL", "delta_qwen", "abs_delta_qwen"),
                              ("Gemma-4", "delta_gemma", "abs_delta_gemma")]:
        out.append(f"## 4.{('a' if vlm == 'Qwen3-VL' else 'b')} "
                   f"Top {TOP_N} cases where |human - {vlm}| is largest")
        out.append("")
        sub = (per_image[per_image[abs_col] > DELTA_THRESHOLD]
               .sort_values(abs_col, ascending=False)
               .head(TOP_N))
        rows = []
        for _, r in sub.iterrows():
            rows.append({
                "Case": r["case_id"][:8],
                "Round": r["round"],
                "Bias": r["bias_type"],
                "Cond": r["condition"][:10],
                "Human": f"{r['human_mean']:.2f}",
                vlm: f"{r['vlm_qwen' if vlm=='Qwen3-VL' else 'vlm_gemma']:.2f}",
                "Delta": f"{r[col]:+.2f}",
                "Prompt": shorten(r["prompt"]),
            })
        out.append(md_table(pd.DataFrame(rows), fmt={}))
        out.append("")

    # ------------------------------------------------------------------
    # 5. Direction flips on stereotype vs anti-stereotype
    # ------------------------------------------------------------------
    out.append("## 5. Direction flips: humans say S>A, VLM says A>S (or vice versa)")
    out.append("")
    flip_rows = []
    flip_examples = {"qwen": [], "gemma": []}
    case_groups = per_image.groupby(["round", "case_id", "bias_type"])
    n_total = 0
    n_flip_q = 0
    n_flip_g = 0
    for (round_name, cid, bt), g in case_groups:
        cd = {row["condition"]: row for _, row in g.iterrows()}
        if not {"stereotype_trigger", "anti_stereotype_trigger"}.issubset(cd):
            continue
        S = cd["stereotype_trigger"]
        A = cd["anti_stereotype_trigger"]
        n_total += 1
        h_dir = S["human_mean"] > A["human_mean"]
        q_dir = S["vlm_qwen"] > A["vlm_qwen"]
        g_dir = S["vlm_gemma"] > A["vlm_gemma"]
        if h_dir != q_dir:
            n_flip_q += 1
            flip_examples["qwen"].append({
                "Case": cid[:8],
                "Round": round_name,
                "Bias": bt,
                "Human (S/A)": f"{S['human_mean']:.2f}/{A['human_mean']:.2f}",
                "Qwen3-VL (S/A)": f"{S['vlm_qwen']:.2f}/{A['vlm_qwen']:.2f}",
                "Stereotype prompt": shorten(S["prompt"], 70),
            })
        if h_dir != g_dir:
            n_flip_g += 1
            flip_examples["gemma"].append({
                "Case": cid[:8],
                "Round": round_name,
                "Bias": bt,
                "Human (S/A)": f"{S['human_mean']:.2f}/{A['human_mean']:.2f}",
                "Gemma-4 (S/A)": f"{S['vlm_gemma']:.2f}/{A['vlm_gemma']:.2f}",
                "Stereotype prompt": shorten(S["prompt"], 70),
            })
    out.append(f"- Total cases with both arms human-rated: {n_total}.")
    out.append(f"- Direction flips Qwen3-VL vs human: **{n_flip_q} / {n_total}**.")
    out.append(f"- Direction flips Gemma-4 vs human: **{n_flip_g} / {n_total}**.")
    out.append("")
    for vlm, key in [("Qwen3-VL", "qwen"), ("Gemma-4", "gemma")]:
        out.append(f"### Flip examples ({vlm}, up to 10):")
        out.append("")
        ex = flip_examples[key][:10]
        if ex:
            out.append(md_table(pd.DataFrame(ex)))
        else:
            out.append("(none)")
        out.append("")

    # ------------------------------------------------------------------
    # CSV: full per-image table for downstream qualitative review
    # ------------------------------------------------------------------
    csv_cols = ["round", "case_id", "source", "bias_type", "condition",
                "prompt", "n_raters", "human_mean", "human_std",
                "vlm_qwen", "vlm_gemma",
                "delta_qwen", "delta_gemma",
                "abs_delta_qwen", "abs_delta_gemma"]
    per_image[csv_cols].to_csv(OUT_CSV, index=False)
    out.append("")
    out.append(f"Per-image rows written to `{OUT_CSV}`.")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
