"""Analyze human-eval form responses (Round 1 + Round 2) and compare to
auto-evaluation VLMs (Qwen3-VL, Gemma-4).

Reads:
  data/human_eval/manifest.csv              + Round-1 form responses CSV
  data/human_eval_round2/manifest.csv       + Round-2 form responses CSV

Writes:
  reports/human_eval_summary.md
"""

import os
import re
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = "/data/gpfs/projects/punim2888/stereoimage"

ROUNDS = [
    {
        "name": "Round 1",
        "manifest": f"{ROOT}/data/human_eval/manifest.csv",
        "responses": f"{ROOT}/data/human_eval/Stereotype Image Evaluation (Responses) - Form responses 1.csv",
    },
    {
        "name": "Round 2",
        "manifest": f"{ROOT}/data/human_eval_round2/manifest.csv",
        "responses": f"{ROOT}/data/human_eval_round2/Stereotype Image Evaluation – Round 2 (Responses) - Form responses 1.csv",
    },
]
OUT_MD = f"{ROOT}/reports/human_eval_summary.md"


def parse_rating(s) -> float:
    if not isinstance(s, str):
        return np.nan
    m = re.match(r"^(\d+)", s.strip())
    return float(m.group(1)) if m else np.nan


def parse_yn(s) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    if s.startswith("yes"):
        return "Yes"
    if s.startswith("no"):
        return "No"
    if s.startswith("unsure"):
        return "Unsure"
    return None


def long_form(round_name, manifest_path, responses_path) -> pd.DataFrame:
    """Reshape one round's wide form responses to long-form rows."""
    resp = pd.read_csv(responses_path, header=None, dtype=str)
    manifest = (pd.read_csv(manifest_path)
                .sort_values(["section_id", "sub_q"])
                .reset_index(drop=True))
    n_q = len(manifest)
    assert n_q == resp.shape[1] - 1, \
        f"{round_name}: manifest has {n_q} questions but responses have " \
        f"{resp.shape[1]-1} answer columns"

    rows = []
    for r in range(1, len(resp)):
        for i, m in manifest.iterrows():
            v = resp.iat[r, i + 1]
            rec = {
                "round": round_name,
                # Same 5 raters did both rounds, paired by submission order.
                "rater": f"rater_{r}",
                "case_id": m["case_id"],
                "section_id": m["section_id"],
                "sub_q": m["sub_q"],
                "kind": m["question_kind"],
                "condition": m["condition"],
                "bias_type": m["bias_type"],
                "vlm_qwen": m["vlm_qwen_score"],
                "vlm_gemma": m["vlm_gemma_score"],
                "raw": v,
                "rating": np.nan,
                "yn": None,
            }
            if m["question_kind"] == "kg_validity":
                rec["yn"] = parse_yn(v)
            else:
                rec["rating"] = parse_rating(v)
            rows.append(rec)
    return pd.DataFrame(rows)


def icc2_single(M: np.ndarray) -> float:
    """ICC(2,1) absolute agreement, single-rater (Shrout & Fleiss 1979).
    M shape (n_subjects, n_raters)."""
    n, k = M.shape
    grand = M.mean()
    rows = M.mean(axis=1)
    cols = M.mean(axis=0)
    SST = ((M - grand) ** 2).sum()
    SSB = k * ((rows - grand) ** 2).sum()
    SSC = n * ((cols - grand) ** 2).sum()
    SSE = SST - SSB - SSC
    MSR = SSB / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    return (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)


def quad_kappa(y1, y2, K=6) -> float:
    """Quadratic-weighted Cohen's kappa for 0..K-1 ordinal ratings."""
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    O = np.zeros((K, K))
    for a, b in zip(y1, y2):
        O[a, b] += 1
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / ((K - 1) ** 2)
    h1 = O.sum(axis=1)
    h2 = O.sum(axis=0)
    E = np.outer(h1, h2) / max(O.sum(), 1)
    num = (W * O).sum()
    den = (W * E).sum()
    return 1 - num / den if den else 1.0


def md_table(df: pd.DataFrame, fmt=None) -> str:
    fmt = fmt or {}

    def cell(c, v):
        if pd.isna(v):
            return ""
        f = fmt.get(c)
        if f:
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
    long["rating"] = pd.to_numeric(long["rating"], errors="coerce")
    long["vlm_qwen"] = pd.to_numeric(long["vlm_qwen"], errors="coerce")
    long["vlm_gemma"] = pd.to_numeric(long["vlm_gemma"], errors="coerce")

    cases_per_round = long.groupby("round")["case_id"].nunique().to_dict()
    n_raters_total = long["rater"].nunique()
    n_cases_total = long["case_id"].nunique()

    out = []
    out.append("# Human Evaluation Summary (Round 1 + Round 2)")
    out.append("")
    out.append(
        f"- **Round 1**: {cases_per_round.get('Round 1', 0)} cases.")
    out.append(
        f"- **Round 2**: {cases_per_round.get('Round 2', 0)} cases.")
    out.append(
        f"- **Raters**: {n_raters_total} (the same annotators completed both "
        f"rounds, paired by submission order); each rater scored "
        f"{n_cases_total} unique cases.")
    out.append("")
    out.append(
        "Each case asks one KG-validity question (Yes / No / Unsure) and three "
        "image-rating questions (0-5) for the stereotype-trigger, neutral, and "
        "anti-stereotype-trigger images. Auto-evaluation scores come from "
        "Qwen3-VL and Gemma-4, recorded in each round's manifest.")
    out.append("")

    # ------------------------------------------------------------------
    # 1. KG validity
    # ------------------------------------------------------------------
    out.append("## 1. KG validity -- Is this a real societal stereotype?")
    out.append("")
    kg = long[long["kind"] == "kg_validity"]
    overall = kg["yn"].value_counts(dropna=False).reindex(
        ["Yes", "Unsure", "No"], fill_value=0)
    pct_yes = (kg["yn"] == "Yes").mean() * 100
    pct_unsure = (kg["yn"] == "Unsure").mean() * 100
    pct_no = (kg["yn"] == "No").mean() * 100

    rows = []
    for r in ["Round 1", "Round 2", "Combined"]:
        sub = kg if r == "Combined" else kg[kg["round"] == r]
        rows.append({
            "Set": r,
            "n ratings": len(sub),
            "% Yes": (sub["yn"] == "Yes").mean() * 100,
            "% Unsure": (sub["yn"] == "Unsure").mean() * 100,
            "% No": (sub["yn"] == "No").mean() * 100,
        })
    out.append(md_table(
        pd.DataFrame(rows),
        fmt={"% Yes": "{:.1f}", "% Unsure": "{:.1f}", "% No": "{:.1f}"}))
    out.append("")

    # Per-case majority
    rows = []
    for r in ["Round 1", "Round 2", "Combined"]:
        sub = kg if r == "Combined" else kg[kg["round"] == r]
        per = sub.groupby("case_id")["yn"].agg(list)
        rows.append({
            "Set": r,
            "Cases": int(per.shape[0]),
            "All-rater Yes": int(per.apply(
                lambda t: all(x == "Yes" for x in t)).sum()),
            "Majority Yes (>= half)": int(per.apply(
                lambda t: sum(1 for x in t if x == "Yes") > len(t) / 2).sum()),
            "Unanimous (any label)": int(per.apply(
                lambda t: len(set(t)) == 1).sum()),
        })
    out.append(md_table(pd.DataFrame(rows)))
    out.append("")
    out.append(f"Across both rounds, **{pct_yes:.1f}%** of KG-validity ratings "
               f"answered \"Yes\", **{pct_unsure:.1f}%** \"Unsure\", and "
               f"**{pct_no:.1f}%** \"No\".")
    out.append("")

    # By bias_type
    by_bt = (kg.groupby("bias_type")["yn"]
             .agg([("n", "count"),
                   ("pct_yes", lambda s: (s == "Yes").mean() * 100),
                   ("pct_unsure", lambda s: (s == "Unsure").mean() * 100),
                   ("pct_no", lambda s: (s == "No").mean() * 100)])
             .round(1).reset_index())
    out.append("**KG validity by bias type (combined):**")
    out.append("")
    out.append(md_table(by_bt))
    out.append("")

    # ------------------------------------------------------------------
    # 2. Image ratings by condition
    # ------------------------------------------------------------------
    out.append("## 2. Image ratings (0-5) by condition")
    out.append("")
    ir = long[long["kind"] == "image_rating"].copy()

    rows = []
    for r in ["Round 1", "Round 2", "Combined"]:
        for cond in ["stereotype_trigger", "neutral", "anti_stereotype_trigger"]:
            sub = ir if r == "Combined" else ir[ir["round"] == r]
            sub = sub[sub["condition"] == cond]
            rows.append({
                "Set": r,
                "Condition": cond,
                "Mean": sub["rating"].mean(),
                "Std": sub["rating"].std(),
                "n": int(sub["rating"].notna().sum()),
            })
    cond_df = pd.DataFrame(rows)
    out.append(md_table(cond_df, fmt={"Mean": "{:.3f}", "Std": "{:.3f}"}))
    out.append("")

    rows = []
    for r in ["Round 1", "Round 2", "Combined"]:
        sub = ir if r == "Combined" else ir[ir["round"] == r]
        gm = sub.groupby("condition")["rating"].mean()
        rows.append({
            "Set": r,
            "Bias amplification (S - N)":
                gm["stereotype_trigger"] - gm["neutral"],
            "Total separation (S - A)":
                gm["stereotype_trigger"] - gm["anti_stereotype_trigger"],
        })
    out.append("**Bias amplification and total separation:**")
    out.append("")
    out.append(md_table(
        pd.DataFrame(rows),
        fmt={"Bias amplification (S - N)": "{:.3f}",
             "Total separation (S - A)": "{:.3f}"}))
    out.append("")

    bt_cond = (ir.groupby(["bias_type", "condition"])["rating"].mean()
               .unstack().reindex(columns=["stereotype_trigger",
                                           "neutral",
                                           "anti_stereotype_trigger"])
               .round(2).reset_index())
    out.append("**Mean human rating by bias type x condition (combined):**")
    out.append("")
    out.append(md_table(bt_cond))
    out.append("")

    # ------------------------------------------------------------------
    # 3. Inter-rater reliability (within round; rater counts may differ)
    # ------------------------------------------------------------------
    out.append("## 3. Inter-rater reliability (image ratings)")
    out.append("")
    rel_rows = []
    for r in ["Round 1", "Round 2", "Combined"]:
        sub = ir if r == "Combined" else ir[ir["round"] == r]
        wide = (sub.pivot_table(index=["case_id", "condition"],
                                columns="rater",
                                values="rating",
                                aggfunc="first")
                .dropna())
        if wide.shape[1] < 2 or wide.empty:
            continue
        cols = list(wide.columns)
        prs, kps = [], []
        for a, b in combinations(cols, 2):
            prs.append(pearsonr(wide[a], wide[b]).statistic)
            kps.append(quad_kappa(wide[a].astype(int), wide[b].astype(int)))
        M = wide.to_numpy()
        icc1 = icc2_single(M)
        k = M.shape[1]
        icc_k = k * icc1 / (1 + (k - 1) * icc1) if (1 + (k - 1) * icc1) else np.nan
        rel_rows.append({
            "Set": r,
            "Raters": wide.shape[1],
            "Images": wide.shape[0],
            "Pairwise Pearson r (mean)": float(np.mean(prs)),
            "Pairwise weighted kappa (mean)": float(np.mean(kps)),
            "ICC(2,1)": icc1,
            "ICC(2,k)": icc_k,
        })
    out.append(md_table(
        pd.DataFrame(rel_rows),
        fmt={"Pairwise Pearson r (mean)": "{:.3f}",
             "Pairwise weighted kappa (mean)": "{:.3f}",
             "ICC(2,1)": "{:.3f}",
             "ICC(2,k)": "{:.3f}"}))
    out.append("")
    out.append("The same 5 raters scored both rounds (paired by submission "
               "order), so the Combined row pools 300 images per rater "
               "(100 cases x 3 conditions).")
    out.append("")

    # ------------------------------------------------------------------
    # 4. Human vs auto-evaluation (VLM)
    # ------------------------------------------------------------------
    out.append("## 4. Human vs auto-evaluation (Qwen3-VL, Gemma-4)")
    out.append("")
    rows = []
    for r in ["Round 1", "Round 2", "Combined"]:
        sub = ir if r == "Combined" else ir[ir["round"] == r]
        per_image = (sub.groupby(["round", "case_id", "condition"])
                     .agg(human_mean=("rating", "mean"),
                          vlm_qwen=("vlm_qwen", "first"),
                          vlm_gemma=("vlm_gemma", "first"))
                     .reset_index()
                     .dropna(subset=["human_mean", "vlm_qwen", "vlm_gemma"]))
        if per_image.empty:
            continue
        n = len(per_image)
        pq = pearsonr(per_image.human_mean, per_image.vlm_qwen).statistic
        sq = spearmanr(per_image.human_mean, per_image.vlm_qwen).statistic
        pg = pearsonr(per_image.human_mean, per_image.vlm_gemma).statistic
        sg = spearmanr(per_image.human_mean, per_image.vlm_gemma).statistic
        mq = (per_image.human_mean - per_image.vlm_qwen).abs().mean()
        mg = (per_image.human_mean - per_image.vlm_gemma).abs().mean()
        # Bias direction agreement: stereo > anti
        rows_dir = []
        for cid, g in per_image.groupby("case_id"):
            d = {row.condition: row for _, row in g.iterrows()}
            if {"stereotype_trigger", "anti_stereotype_trigger"}.issubset(d):
                S, A = d["stereotype_trigger"], d["anti_stereotype_trigger"]
                rows_dir.append({
                    "human": S.human_mean > A.human_mean,
                    "qwen":  S.vlm_qwen > A.vlm_qwen,
                    "gemma": S.vlm_gemma > A.vlm_gemma,
                })
        rd = pd.DataFrame(rows_dir)
        rows.append({
            "Set": r, "Images": n,
            "Pearson r (Qwen3-VL)": pq, "Spearman rho (Qwen3-VL)": sq,
            "MAE (Qwen3-VL)": mq,
            "Pearson r (Gemma-4)": pg, "Spearman rho (Gemma-4)": sg,
            "MAE (Gemma-4)": mg,
            "Cases S>A human/qwen/gemma":
                f"{int(rd['human'].sum())}/{int(rd['qwen'].sum())}/"
                f"{int(rd['gemma'].sum())} of {len(rd)}",
        })
    cmp_df = pd.DataFrame(rows)
    fmt = {c: "{:.3f}" for c in cmp_df.columns
           if c.startswith(("Pearson", "Spearman", "MAE"))}
    out.append(md_table(cmp_df, fmt=fmt))
    out.append("")

    # Per-rater agreement with VLMs (combined)
    out.append("**Per-rater Pearson r with VLMs (combined across rounds):**")
    out.append("")
    rows = []
    for rater, sub in ir.groupby("rater"):
        per_image = sub.dropna(subset=["rating", "vlm_qwen", "vlm_gemma"])
        if len(per_image) < 5:
            continue
        rows.append({
            "Rater": rater,
            "Images rated": len(per_image),
            "Pearson r vs Qwen3-VL":
                pearsonr(per_image.rating, per_image.vlm_qwen).statistic,
            "Pearson r vs Gemma-4":
                pearsonr(per_image.rating, per_image.vlm_gemma).statistic,
        })
    out.append(md_table(
        pd.DataFrame(rows),
        fmt={"Pearson r vs Qwen3-VL": "{:.3f}",
             "Pearson r vs Gemma-4": "{:.3f}"}))
    out.append("")

    # ------------------------------------------------------------------
    # 4b. Overall agreement (5 humans + 2 VLMs treated as 7 raters)
    # ------------------------------------------------------------------
    out.append("## 4b. Overall agreement (5 humans + Qwen3-VL + Gemma-4)")
    out.append("")
    out.append("Treats all 7 annotators as raters of the same 300 images and "
               "reports pooled inter-annotator reliability.")
    out.append("")

    wide_h = (ir.pivot_table(index=["case_id", "condition"],
                             columns="rater",
                             values="rating",
                             aggfunc="first"))
    vlm_pairs = (ir.groupby(["case_id", "condition"])
                 .agg(qwen3vl=("vlm_qwen", "first"),
                      gemma4=("vlm_gemma", "first")))
    wide5 = wide_h.join(vlm_pairs, how="inner").dropna()
    if not wide5.empty:
        cols = list(wide5.columns)
        prs, srs, kps = [], [], []
        for a, b in combinations(cols, 2):
            prs.append((a, b, pearsonr(wide5[a], wide5[b]).statistic))
            srs.append((a, b, spearmanr(wide5[a], wide5[b]).statistic))
            kps.append((a, b, quad_kappa(wide5[a].astype(int),
                                          wide5[b].astype(int))))

        # Pooled aggregate
        mean_pearson = float(np.mean([x[2] for x in prs]))
        mean_spearman = float(np.mean([x[2] for x in srs]))
        mean_kappa = float(np.mean([x[2] for x in kps]))
        M5 = wide5.to_numpy()
        icc1 = icc2_single(M5)
        k = M5.shape[1]
        icc_k = k * icc1 / (1 + (k - 1) * icc1) if (1 + (k - 1) * icc1) else np.nan
        out.append(md_table(
            pd.DataFrame([{
                "Annotators": "5 humans + 2 VLMs",
                "Images": int(wide5.shape[0]),
                "Mean pairwise Pearson r": mean_pearson,
                "Mean pairwise Spearman rho": mean_spearman,
                "Mean pairwise weighted kappa": mean_kappa,
                "ICC(2,1)": icc1,
                "ICC(2,k)": icc_k,
            }]),
            fmt={"Mean pairwise Pearson r": "{:.3f}",
                 "Mean pairwise Spearman rho": "{:.3f}",
                 "Mean pairwise weighted kappa": "{:.3f}",
                 "ICC(2,1)": "{:.3f}",
                 "ICC(2,k)": "{:.3f}"}))
        out.append("")

        # Full pairwise matrix
        out.append("**Full pairwise Pearson r matrix:**")
        out.append("")
        mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
        for a in cols:
            mat.loc[a, a] = 1.0
        for a, b, v in prs:
            mat.loc[a, b] = v
            mat.loc[b, a] = v
        mat = mat.round(3).reset_index().rename(columns={"index": ""})
        out.append(md_table(mat))
        out.append("")
    else:
        out.append("Not enough overlapping data to compute pooled agreement.")
        out.append("")

    # ------------------------------------------------------------------
    # 5. Caveats
    # ------------------------------------------------------------------
    out.append("## 5. Caveats")
    out.append("")
    out.append("- Only 5 raters total, so per-bias-type breakdowns have "
               "wide CIs and should be read as directional.")
    out.append("- Round 1 and Round 2 use disjoint case samples drawn from "
               "the same lean-stereotype pool, with the same 5 raters scoring "
               "both rounds (paired by submission order).")
    out.append("- Only seed 1 was rated for each case.")
    out.append("- VLM scores are stored in each round's manifest "
               "(`vlm_qwen_score`, `vlm_gemma_score`) and were generated by "
               "Qwen3-VL-30B and Gemma-4 respectively.")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
