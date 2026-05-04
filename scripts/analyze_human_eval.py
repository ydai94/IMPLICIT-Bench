"""Analyze human-eval form responses (Round 1 + Round 2) and compare to
auto-evaluation VLMs (Qwen3-VL, Gemma-4).

Reads:
  data/human_eval/manifest.csv              + Round-1 form responses CSV
  data/human_eval_round2/manifest.csv       + Round-2 form responses CSV

Writes:
  reports/human_eval_summary.md

Round 1 has 13 form responses but Round 2 has 10. To produce a symmetric
10-vs-10 analysis we randomly subsample 10 of the 13 R1 responses with a
fixed seed, then pair the selected R1 raters to R2 raters by submission
order. The pairing is nominal -- the form collects no per-rater identifier.
"""

import os
import random
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

# Random seed for selecting which R1 form rows to keep so both rounds have
# the same number of raters. Change only if you intend the report numbers
# to shift.
RATER_SAMPLE_SEED = 42
TARGET_RATERS = 10

# Raters whose mean pairwise Pearson r with the other raters in their round
# is below this threshold are flagged as outliers and re-reported in a
# "drop outliers" row alongside the headline numbers. 0.40 cleanly
# separates one rater (mean pairwise r ≈ 0.27 in Round 1 and ≈ 0.30 in
# Round 2) from the rest of the pool, all of whom sit above 0.50.
OUTLIER_PEARSON_THRESHOLD = 0.40

# Externally confirmed: certain raters across rounds are the same person.
# This is information the form does NOT capture (it only records a
# Timestamp), but the experimenter has confirmed identity for these IDs
# from off-form knowledge (e.g. team roster). Used only to phrase the
# outlier callout correctly -- it does not unlock cross-round pooling for
# any other rater.
KNOWN_SAME_RATER = {
    frozenset({"r1_rater_8", "r2_rater_9"}),
}


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


def select_rater_rows(responses_path: str, target: int,
                      seed: int) -> list[int]:
    """Return submission-ordered row indices to keep from a responses CSV.

    Drops the header (row 0). If there are at most `target` data rows, all
    are kept; otherwise `target` are randomly sampled (seeded) and returned
    sorted ascending so submission order is preserved.
    """
    resp = pd.read_csv(responses_path, header=None, dtype=str)
    data_rows = list(range(1, len(resp)))
    if len(data_rows) <= target:
        return data_rows
    rng = random.Random(seed)
    return sorted(rng.sample(data_rows, target))


def long_form(round_name, manifest_path, responses_path,
              keep_rows: list[int]) -> pd.DataFrame:
    """Reshape one round's wide form responses to long-form rows.

    `keep_rows` is a submission-ordered list of CSV row indices to retain.
    Raters are namespaced by round (`r1_rater_*` / `r2_rater_*`) because the
    form collects no per-rater identifier, so cross-round identity cannot
    be assumed.
    """
    resp = pd.read_csv(responses_path, header=None, dtype=str)
    manifest = (pd.read_csv(manifest_path)
                .sort_values(["section_id", "sub_q"])
                .reset_index(drop=True))
    n_q = len(manifest)
    assert n_q == resp.shape[1] - 1, \
        f"{round_name}: manifest has {n_q} questions but responses have " \
        f"{resp.shape[1]-1} answer columns"

    round_tag = "r1" if round_name.endswith("1") else "r2"
    rows = []
    for slot, r in enumerate(keep_rows, start=1):
        for i, m in manifest.iterrows():
            v = resp.iat[r, i + 1]
            rec = {
                "round": round_name,
                "rater": f"{round_tag}_rater_{slot}",
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
    selections = {
        r["name"]: select_rater_rows(r["responses"], TARGET_RATERS,
                                     RATER_SAMPLE_SEED)
        for r in ROUNDS
    }
    long = pd.concat([long_form(r["name"], r["manifest"], r["responses"],
                                selections[r["name"]])
                      for r in ROUNDS], ignore_index=True)
    long["rating"] = pd.to_numeric(long["rating"], errors="coerce")
    long["vlm_qwen"] = pd.to_numeric(long["vlm_qwen"], errors="coerce")
    long["vlm_gemma"] = pd.to_numeric(long["vlm_gemma"], errors="coerce")

    cases_per_round = long.groupby("round")["case_id"].nunique().to_dict()
    raters_per_round = long.groupby("round")["rater"].nunique().to_dict()
    n_r1 = raters_per_round.get("Round 1", 0)
    n_r2 = raters_per_round.get("Round 2", 0)
    n_raters = long["rater"].nunique()
    n_cases_total = long["case_id"].nunique()
    n_kg_ratings = int((long["kind"] == "kg_validity").sum())
    n_img_ratings = int((long["kind"] == "image_rating").sum())

    # Precompute outlier raters per round using the same Pearson-threshold
    # rule as Section 3. We need this up-front so Sections 1 (KG validity)
    # and 2 (image-rating means) can also report excl-outlier rows, not
    # just Section 3.
    outlier_raters: set[str] = set()
    _ir_pre = long[long["kind"] == "image_rating"]
    for _r in ["Round 1", "Round 2"]:
        _sub = _ir_pre[_ir_pre["round"] == _r]
        _wide = (_sub.pivot_table(index=["case_id", "condition"],
                                  columns="rater",
                                  values="rating",
                                  aggfunc="first")
                 .dropna())
        if _wide.shape[1] < 2 or _wide.empty:
            continue
        _cols = list(_wide.columns)
        _per = {c: [] for c in _cols}
        for _a, _b in combinations(_cols, 2):
            _r_ab = pearsonr(_wide[_a], _wide[_b]).statistic
            _per[_a].append(_r_ab)
            _per[_b].append(_r_ab)
        for _c, _vals in _per.items():
            if np.mean(_vals) < OUTLIER_PEARSON_THRESHOLD:
                outlier_raters.add(_c)

    # Document which raw form rows were retained (1-indexed within each
    # responses CSV; row 0 is the header).
    r1_kept = selections["Round 1"]
    r2_kept = selections["Round 2"]

    out = []
    out.append("# Human Evaluation Summary (Round 1 + Round 2)")
    out.append("")
    out.append(
        f"- **Round 1**: {cases_per_round.get('Round 1', 0)} cases, "
        f"{n_r1} raters.")
    out.append(
        f"- **Round 2**: {cases_per_round.get('Round 2', 0)} cases, "
        f"{n_r2} raters.")
    out.append(
        f"- **Total ratings**: {n_kg_ratings} KG-validity + "
        f"{n_img_ratings} image ratings. Raters are anonymous form "
        f"respondents and are not paired across rounds (the form collects "
        f"only a Timestamp), so per-rater pooling across rounds is omitted.")
    out.append(
        f"- **Round 1 subsample**: 10 of 13 form responses kept "
        f"(rows {r1_kept}, 1-indexed; seed {RATER_SAMPLE_SEED}). "
        f"Round 2 kept all {len(r2_kept)} responses.")
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
        if outlier_raters:
            sub_clean = sub[~sub["rater"].isin(outlier_raters)]
            if len(sub_clean):
                rows.append({
                    "Set": f"{r} (excl. outliers)",
                    "n ratings": len(sub_clean),
                    "% Yes": (sub_clean["yn"] == "Yes").mean() * 100,
                    "% Unsure": (sub_clean["yn"] == "Unsure").mean() * 100,
                    "% No": (sub_clean["yn"] == "No").mean() * 100,
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
    set_order = []
    for r in ["Round 1", "Round 2", "Combined"]:
        set_order.append((r, False))
        if outlier_raters:
            set_order.append((r, True))
    for r, excl in set_order:
        for cond in ["stereotype_trigger", "neutral", "anti_stereotype_trigger"]:
            sub = ir if r == "Combined" else ir[ir["round"] == r]
            if excl:
                sub = sub[~sub["rater"].isin(outlier_raters)]
            sub = sub[sub["condition"] == cond]
            rows.append({
                "Set": f"{r} (excl. outliers)" if excl else r,
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

    def _stats_for(wide):
        cols = list(wide.columns)
        prs, kps = [], []
        per_rater_mean_pr = {c: [] for c in cols}
        for a, b in combinations(cols, 2):
            r_ab = pearsonr(wide[a], wide[b]).statistic
            prs.append(r_ab)
            kps.append(quad_kappa(wide[a].astype(int), wide[b].astype(int)))
            per_rater_mean_pr[a].append(r_ab)
            per_rater_mean_pr[b].append(r_ab)
        M = wide.to_numpy()
        icc1 = icc2_single(M)
        k = M.shape[1]
        icc_k = (k * icc1 / (1 + (k - 1) * icc1)
                 if (1 + (k - 1) * icc1) else np.nan)
        return {
            "n_raters": wide.shape[1],
            "n_images": wide.shape[0],
            "mean_pearson": float(np.mean(prs)),
            "mean_kappa": float(np.mean(kps)),
            "icc1": icc1,
            "icc_k": icc_k,
            "per_rater_mean_pearson": {
                c: float(np.mean(v)) for c, v in per_rater_mean_pr.items()},
        }

    rel_rows = []
    outlier_notes = []
    round_stats: dict[str, dict] = {}
    round_outliers: dict[str, list[str]] = {}
    for r in ["Round 1", "Round 2"]:
        sub = ir[ir["round"] == r]
        wide = (sub.pivot_table(index=["case_id", "condition"],
                                columns="rater",
                                values="rating",
                                aggfunc="first")
                .dropna())
        if wide.shape[1] < 2 or wide.empty:
            continue
        s = _stats_for(wide)
        round_stats[r] = {"all": s}
        rel_rows.append({
            "Set": r,
            "Raters": s["n_raters"],
            "Images": s["n_images"],
            "Pairwise Pearson r (mean)": s["mean_pearson"],
            "Pairwise weighted kappa (mean)": s["mean_kappa"],
            "ICC(2,1)": s["icc1"],
            "ICC(2,k)": s["icc_k"],
        })

        outliers = [c for c, v in s["per_rater_mean_pearson"].items()
                    if v < OUTLIER_PEARSON_THRESHOLD]
        round_outliers[r] = outliers
        if outliers:
            kept = [c for c in wide.columns if c not in outliers]
            wide_clean = wide[kept]
            sc = _stats_for(wide_clean)
            round_stats[r]["clean"] = sc
            rel_rows.append({
                "Set": f"{r} (excl. outliers)",
                "Raters": sc["n_raters"],
                "Images": sc["n_images"],
                "Pairwise Pearson r (mean)": sc["mean_pearson"],
                "Pairwise weighted kappa (mean)": sc["mean_kappa"],
                "ICC(2,1)": sc["icc1"],
                "ICC(2,k)": sc["icc_k"],
            })
            note_parts = []
            for c in outliers:
                note_parts.append(
                    f"`{c}` (mean pairwise r = "
                    f"{s['per_rater_mean_pearson'][c]:.3f})")
            outlier_notes.append(f"{r}: dropped " + ", ".join(note_parts))

    out.append(md_table(
        pd.DataFrame(rel_rows),
        fmt={"Pairwise Pearson r (mean)": "{:.3f}",
             "Pairwise weighted kappa (mean)": "{:.3f}",
             "ICC(2,1)": "{:.3f}",
             "ICC(2,k)": "{:.3f}"}))
    out.append("")
    flat_outliers = {oid for ids in round_outliers.values() for oid in ids}
    grouped_outlier_ids = []
    for group in KNOWN_SAME_RATER:
        if group <= flat_outliers:
            grouped_outlier_ids.append(sorted(group))
            flat_outliers -= group
    for oid in sorted(flat_outliers):
        grouped_outlier_ids.append([oid])
    n_unique_outliers = len(grouped_outlier_ids)
    if grouped_outlier_ids:
        rater_word = "rater" if n_unique_outliers == 1 else "raters"
        parts = []
        for ids in grouped_outlier_ids:
            if len(ids) == 1:
                parts.append(f"`{ids[0]}`")
            else:
                parts.append(" / ".join(f"`{i}`" for i in ids)
                             + " (same person across rounds)")
        out.append(
            f"After flagging at threshold "
            f"{OUTLIER_PEARSON_THRESHOLD:.2f} (mean pairwise Pearson r with "
            f"the other raters in the same round), {n_unique_outliers} "
            f"{rater_word} dropped: " + ", ".join(parts) + ".")
        out.append("")
    out.append("Each round's raters are independent. R1 and R2 are reported "
               "separately; per-rater pooling across rounds is omitted "
               "because the two forms collect no per-rater identifier.")
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

    # Per-rater agreement with VLMs (per round, raters not paired)
    out.append("**Per-rater Pearson r with VLMs (per round, raters not paired "
               "across rounds):**")
    out.append("")
    rows = []
    for (round_name, rater), sub in ir.groupby(["round", "rater"]):
        per_image = sub.dropna(subset=["rating", "vlm_qwen", "vlm_gemma"])
        if len(per_image) < 5:
            continue
        rows.append({
            "Round": round_name,
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
    # 4b. Overall agreement (per round: humans + 2 VLMs as one rater pool)
    # ------------------------------------------------------------------
    out.append("## 4b. Overall agreement (humans + Qwen3-VL + Gemma-4, "
               "per round)")
    out.append("")
    out.append("Treats each round's human raters together with Qwen3-VL and "
               "Gemma-4 as a single rater pool over that round's 150 images "
               "(50 cases x 3 conditions). Rounds are reported separately "
               "because human raters are not paired across rounds.")
    out.append("")

    summary_rows = []
    per_round_matrices = []
    for round_name in ["Round 1", "Round 2"]:
        sub = ir[ir["round"] == round_name]
        if sub.empty:
            continue
        wide_h = sub.pivot_table(index=["case_id", "condition"],
                                 columns="rater",
                                 values="rating",
                                 aggfunc="first")
        vlm_pairs = (sub.groupby(["case_id", "condition"])
                     .agg(qwen3vl=("vlm_qwen", "first"),
                          gemma4=("vlm_gemma", "first")))
        wide_all = wide_h.join(vlm_pairs, how="inner").dropna()
        if wide_all.empty:
            continue
        cols = list(wide_all.columns)
        n_humans = wide_h.shape[1]
        prs, srs, kps = [], [], []
        for a, b in combinations(cols, 2):
            prs.append((a, b, pearsonr(wide_all[a], wide_all[b]).statistic))
            srs.append((a, b, spearmanr(wide_all[a], wide_all[b]).statistic))
            kps.append((a, b, quad_kappa(wide_all[a].astype(int),
                                          wide_all[b].astype(int))))
        M = wide_all.to_numpy()
        icc1 = icc2_single(M)
        k = M.shape[1]
        icc_k = k * icc1 / (1 + (k - 1) * icc1) if (1 + (k - 1) * icc1) else np.nan
        summary_rows.append({
            "Set": round_name,
            "Annotators": f"{n_humans} humans + 2 VLMs",
            "Images": int(wide_all.shape[0]),
            "Mean pairwise Pearson r": float(np.mean([x[2] for x in prs])),
            "Mean pairwise Spearman rho": float(np.mean([x[2] for x in srs])),
            "Mean pairwise weighted kappa": float(np.mean([x[2] for x in kps])),
            "ICC(2,1)": icc1,
            "ICC(2,k)": icc_k,
        })
        per_round_matrices.append((round_name, cols, prs))

    if summary_rows:
        out.append(md_table(
            pd.DataFrame(summary_rows),
            fmt={"Mean pairwise Pearson r": "{:.3f}",
                 "Mean pairwise Spearman rho": "{:.3f}",
                 "Mean pairwise weighted kappa": "{:.3f}",
                 "ICC(2,1)": "{:.3f}",
                 "ICC(2,k)": "{:.3f}"}))
        out.append("")

        for round_name, cols, prs in per_round_matrices:
            out.append(f"**Full pairwise Pearson r matrix -- {round_name}:**")
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
    # 4c. Overall Pearson summary (averaged across rounds, with vs without
    #     outliers)
    # ------------------------------------------------------------------
    out.append("## 4c. Overall Pearson summary (averaged across rounds)")
    out.append("")
    n_dropped_unique = len(grouped_outlier_ids)
    rater_noun = "rater" if n_dropped_unique == 1 else "raters"
    out.append("Single-row view of every Pearson-related metric, averaged "
               "across Round 1 and Round 2. Inter-rater stats are the mean "
               "of each round's within-round pairwise mean (raters are not "
               "paired across rounds, so they cannot be pooled directly). "
               "Human-vs-VLM Pearson is computed on per-image human means "
               "and pooled across both rounds (300 images). The "
               f"\"excl. outliers\" row drops {n_dropped_unique} unique "
               f"{rater_noun} flagged in Section 3 -- the same person "
               f"submitted as `r1_rater_8` in Round 1 and `r2_rater_9` in "
               f"Round 2 (confirmed off-form), so dropping them removes one "
               f"row from each round.")
    out.append("")

    def _vlm_pearson(round_to_keep_raters):
        """Compute pooled human-mean vs VLM Pearson across both rounds.

        `round_to_keep_raters` maps round name -> list of rater IDs to
        include when computing per-image human means.
        """
        frames = []
        for round_name, raters in round_to_keep_raters.items():
            sub = ir[(ir["round"] == round_name) & ir["rater"].isin(raters)]
            per_image = (sub.groupby(["round", "case_id", "condition"])
                         .agg(human_mean=("rating", "mean"),
                              vlm_qwen=("vlm_qwen", "first"),
                              vlm_gemma=("vlm_gemma", "first"))
                         .reset_index()
                         .dropna(subset=["human_mean",
                                         "vlm_qwen", "vlm_gemma"]))
            frames.append(per_image)
        per_image_all = pd.concat(frames, ignore_index=True)
        return {
            "n_images": len(per_image_all),
            "pearson_qwen": pearsonr(per_image_all.human_mean,
                                     per_image_all.vlm_qwen).statistic,
            "pearson_gemma": pearsonr(per_image_all.human_mean,
                                      per_image_all.vlm_gemma).statistic,
        }

    def _row(label, stats_key):
        if not all(stats_key in round_stats[r] for r in round_stats):
            return None
        per_round = [round_stats[r][stats_key] for r in ["Round 1", "Round 2"]
                     if r in round_stats]
        n_raters = ", ".join(str(s["n_raters"]) for s in per_round)
        keep = {r: list(round_stats[r][stats_key]
                        ["per_rater_mean_pearson"].keys())
                for r in round_stats}
        vlm = _vlm_pearson(keep)
        return {
            "Set": label,
            "Raters (R1, R2)": n_raters,
            "Inter-rater Pearson r": float(np.mean(
                [s["mean_pearson"] for s in per_round])),
            "Inter-rater kappa": float(np.mean(
                [s["mean_kappa"] for s in per_round])),
            "Inter-rater ICC(2,1)": float(np.mean(
                [s["icc1"] for s in per_round])),
            "Inter-rater ICC(2,k)": float(np.mean(
                [s["icc_k"] for s in per_round])),
            "Human vs Qwen3-VL (Pearson r, 300 imgs)": vlm["pearson_qwen"],
            "Human vs Gemma-4 (Pearson r, 300 imgs)": vlm["pearson_gemma"],
        }

    summary_rows = [r for r in [_row("All raters", "all"),
                                _row("Excl. outliers", "clean")] if r]
    if summary_rows:
        out.append(md_table(
            pd.DataFrame(summary_rows),
            fmt={"Inter-rater Pearson r": "{:.3f}",
                 "Inter-rater kappa": "{:.3f}",
                 "Inter-rater ICC(2,1)": "{:.3f}",
                 "Inter-rater ICC(2,k)": "{:.3f}",
                 "Human vs Qwen3-VL (Pearson r, 300 imgs)": "{:.3f}",
                 "Human vs Gemma-4 (Pearson r, 300 imgs)": "{:.3f}"}))
        out.append("")

    # ------------------------------------------------------------------
    # 5. Caveats
    # ------------------------------------------------------------------
    out.append("## 5. Caveats")
    out.append("")
    out.append(f"- {n_r1} raters in Round 1 and {n_r2} in Round 2; per-bias-"
               f"type breakdowns still have wide CIs and should be read as "
               f"directional.")
    out.append(f"- Round 1 had 13 form responses; 10 were randomly retained "
               f"(seed {RATER_SAMPLE_SEED}, kept rows {r1_kept}) so both "
               f"rounds contribute the same number of raters.")
    out.append("- Round 1 and Round 2 use disjoint case samples drawn from "
               "the same lean-stereotype pool. The form collects no "
               "per-rater identifier, so cross-round identity is generally "
               "unknown and per-rater analyses are reported per round. "
               "One exception: the rater submitting as `r1_rater_8` and "
               "`r2_rater_9` has been confirmed off-form to be the same "
               "person; this rater is the sole outlier flagged in "
               "Section 3 / 4c.")
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
