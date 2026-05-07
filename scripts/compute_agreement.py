#!/usr/bin/env python3
"""Compute inter-annotator agreement and accuracy metrics for LLM prompt labeling.
Reads results from both matching and independent approaches."""

import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations

# ── Config ────────────────────────────────────────────────────────────────────
_BASE = "/path/to/stereoimage"
MATCHING_CSV = f"{_BASE}/data/labeled_matching.csv"
INDEPENDENT_CSV = f"{_BASE}/data/labeled_independent.csv"
REPORT_FILE = f"{_BASE}/reports/agreement_report.md"
BIAS_TYPE_CSV = f"{_BASE}/reports/agreement_by_bias_type.csv"

MODEL_NAMES = ["claude_sonnet", "qwen3", "gemma4", "llama4"]
MODEL_DISPLAY = {"claude_sonnet": "Claude Sonnet 4.6", "qwen3": "Qwen3-30B",
                 "gemma4": "Gemma4-26B", "llama4": "Llama4-Maverick"}

PROMPT_COLS = ["prompt_neutral", "prompt_stereotype", "prompt_anti_stereotype"]
GROUND_TRUTH = {"prompt_neutral": "neutral",
                "prompt_stereotype": "stereotype",
                "prompt_anti_stereotype": "anti-stereotype"}
LABEL_SET = ["neutral", "stereotype", "anti-stereotype"]


# ── Agreement metrics (manual implementations) ───────────────────────────────
def cohens_kappa(y1, y2):
    """Cohen's kappa for two raters."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y1, y2)


def fleiss_kappa(rating_matrix):
    """Fleiss' kappa for multiple raters.
    rating_matrix: (n_subjects, n_categories) -- count of raters per category per subject."""
    N, k = rating_matrix.shape
    n = rating_matrix.sum(axis=1)[0]  # raters per subject (assumed constant)
    if n <= 1:
        return 0.0

    # Proportion per category
    p_j = rating_matrix.sum(axis=0) / (N * n)

    # Per-subject agreement
    P_i = (np.sum(rating_matrix ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = np.mean(P_i)
    P_e = np.sum(p_j ** 2)

    if abs(1 - P_e) < 1e-10:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def krippendorff_alpha_nominal(data_matrix):
    """Krippendorff's alpha for nominal data.
    data_matrix: (n_coders, n_units) with values as category indices. -1 = missing."""
    n_coders, n_units = data_matrix.shape
    categories = sorted(set(data_matrix.flatten()) - {-1})
    if len(categories) < 2:
        return 1.0

    # Build coincidence matrix
    n_cats = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    coincidence = np.zeros((n_cats, n_cats))

    for u in range(n_units):
        values = [data_matrix[c, u] for c in range(n_coders) if data_matrix[c, u] != -1]
        m_u = len(values)
        if m_u < 2:
            continue
        for i in range(len(values)):
            for j in range(len(values)):
                if i != j:
                    ci, cj = cat_to_idx[values[i]], cat_to_idx[values[j]]
                    coincidence[ci, cj] += 1.0 / (m_u - 1)

    n_total = coincidence.sum()
    if n_total == 0:
        return 0.0

    # Observed disagreement
    D_o = 1.0 - np.trace(coincidence) / n_total

    # Expected disagreement
    marginals = coincidence.sum(axis=1)
    D_e = 1.0 - np.sum(marginals ** 2) / (n_total ** 2)

    if abs(D_e) < 1e-10:
        return 1.0
    return 1.0 - D_o / D_e


def confusion_matrix_str(y_true, y_pred, labels):
    """Format a confusion matrix as a markdown table string."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = "| Pred \\ True | " + " | ".join(labels) + " |"
    sep = "|" + "|".join(["---"] * (len(labels) + 1)) + "|"
    rows = [header, sep]
    for i, label in enumerate(labels):
        row = f"| {label} | " + " | ".join(str(cm[i, j]) for j in range(len(labels))) + " |"
        rows.append(row)
    return "\n".join(rows)


# ── Matching approach analysis ────────────────────────────────────────────────
def analyze_matching(df):
    report = []
    report.append("## Matching Approach Results\n")
    report.append("Each LLM saw all 3 prompts together (shuffled) and assigned labels.\n")

    # Per-model accuracy
    report.append("### Per-Model Accuracy\n")
    report.append("| Model | Row Accuracy | Prompt Accuracy (N) | Prompt Accuracy (S) | Prompt Accuracy (A) | Overall Prompt Acc |")
    report.append("|---|---|---|---|---|---|")

    model_prompt_labels = {}  # model -> list of (true, pred) for all prompts

    for mname in MODEL_NAMES:
        correct_col = f"{mname}_correct"
        if correct_col not in df.columns:
            continue
        valid = df[correct_col].apply(lambda x: x in (0, 1, "0", "1", 0.0, 1.0))
        if not valid.any():
            report.append(f"| {MODEL_DISPLAY.get(mname, mname)} | N/A | N/A | N/A | N/A | N/A |")
            continue

        row_acc = df.loc[valid, correct_col].astype(float).mean()

        # Per-prompt-type accuracy
        col_accs = {}
        all_true, all_pred = [], []
        for col, gt_label in GROUND_TRUTH.items():
            short = {"prompt_neutral": "neutral", "prompt_stereotype": "stereo",
                     "prompt_anti_stereotype": "anti"}[col]
            label_col = f"{mname}_label_{short}"
            if label_col in df.columns:
                mask = df[label_col].isin(LABEL_SET)
                if mask.any():
                    acc = (df.loc[mask, label_col] == gt_label).mean()
                    col_accs[col] = acc
                    all_true.extend([gt_label] * mask.sum())
                    all_pred.extend(df.loc[mask, label_col].tolist())

        n_acc = col_accs.get("prompt_neutral", float("nan"))
        s_acc = col_accs.get("prompt_stereotype", float("nan"))
        a_acc = col_accs.get("prompt_anti_stereotype", float("nan"))
        overall = sum(1 for t, p in zip(all_true, all_pred) if t == p) / max(len(all_true), 1)

        model_prompt_labels[mname] = list(zip(all_true, all_pred))

        report.append(
            f"| {MODEL_DISPLAY.get(mname, mname)} | {row_acc:.4f} | {n_acc:.4f} | {s_acc:.4f} | {a_acc:.4f} | {overall:.4f} |")

    # Confusion matrices
    report.append("\n### Confusion Matrices\n")
    for mname in MODEL_NAMES:
        all_true, all_pred = [], []
        for col, gt_label in GROUND_TRUTH.items():
            short = {"prompt_neutral": "neutral", "prompt_stereotype": "stereo",
                     "prompt_anti_stereotype": "anti"}[col]
            label_col = f"{mname}_label_{short}"
            if label_col in df.columns:
                mask = df[label_col].isin(LABEL_SET)
                all_true.extend([gt_label] * mask.sum())
                all_pred.extend(df.loc[mask, label_col].tolist())
        if all_true:
            report.append(f"#### {MODEL_DISPLAY.get(mname, mname)}\n")
            report.append(confusion_matrix_str(all_true, all_pred, LABEL_SET))
            report.append("")

    # Inter-model agreement
    report.append("\n### Inter-Model Agreement\n")
    report.extend(compute_agreement_section(df, "matching"))

    return "\n".join(report)


# ── Independent approach analysis ────────────────────────────────────────────
def analyze_independent(df):
    report = []
    report.append("## Independent Approach Results\n")
    report.append("Each LLM classified each prompt individually without seeing siblings.\n")

    report.append("### Per-Model Accuracy\n")
    report.append("| Model | Neutral Acc | Stereotype Acc | Anti-Stereo Acc | Overall Prompt Acc |")
    report.append("|---|---|---|---|---|")

    for mname in MODEL_NAMES:
        col_accs = {}
        total_correct = 0
        total_count = 0
        for col, gt_label in GROUND_TRUTH.items():
            short = col.replace("prompt_", "").replace("_trigger", "")
            label_col = f"{mname}_ind_{short}"
            if label_col in df.columns:
                mask = df[label_col].isin(LABEL_SET)
                if mask.any():
                    correct = (df.loc[mask, label_col] == gt_label).sum()
                    count = mask.sum()
                    col_accs[col] = correct / count
                    total_correct += correct
                    total_count += count

        n_acc = col_accs.get("prompt_neutral", float("nan"))
        s_acc = col_accs.get("prompt_stereotype", float("nan"))
        a_acc = col_accs.get("prompt_anti_stereotype", float("nan"))
        overall = total_correct / max(total_count, 1)

        report.append(
            f"| {MODEL_DISPLAY.get(mname, mname)} | {n_acc:.4f} | {s_acc:.4f} | {a_acc:.4f} | {overall:.4f} |")

    # Confusion matrices
    report.append("\n### Confusion Matrices\n")
    for mname in MODEL_NAMES:
        all_true, all_pred = [], []
        for col, gt_label in GROUND_TRUTH.items():
            short = col.replace("prompt_", "").replace("_trigger", "")
            label_col = f"{mname}_ind_{short}"
            if label_col in df.columns:
                mask = df[label_col].isin(LABEL_SET)
                all_true.extend([gt_label] * mask.sum())
                all_pred.extend(df.loc[mask, label_col].tolist())
        if all_true:
            report.append(f"#### {MODEL_DISPLAY.get(mname, mname)}\n")
            report.append(confusion_matrix_str(all_true, all_pred, LABEL_SET))
            report.append("")

    # Inter-model agreement
    report.append("\n### Inter-Model Agreement\n")
    report.extend(compute_agreement_section(df, "independent"))

    return "\n".join(report)


# ── Shared agreement computation ─────────────────────────────────────────────
def compute_agreement_section(df, approach):
    """Compute Fleiss' kappa, Krippendorff's alpha, pairwise Cohen's kappa."""
    lines = []

    # Build coding matrix: (n_units, n_models) with label indices
    label_to_idx = {l: i for i, l in enumerate(LABEL_SET)}
    n_models = len(MODEL_NAMES)

    # Collect per-prompt labels from each model
    all_model_labels = {mname: [] for mname in MODEL_NAMES}

    for col in PROMPT_COLS:
        for mname in MODEL_NAMES:
            if approach == "matching":
                short = {"prompt_neutral": "neutral", "prompt_stereotype": "stereo",
                         "prompt_anti_stereotype": "anti"}[col]
                label_col = f"{mname}_label_{short}"
            else:
                short = col.replace("prompt_", "").replace("_trigger", "")
                label_col = f"{mname}_ind_{short}"

            if label_col in df.columns:
                all_model_labels[mname].extend(df[label_col].tolist())
            else:
                all_model_labels[mname].extend([""] * len(df))

    n_units = len(all_model_labels[MODEL_NAMES[0]])

    # Filter to units where all models provided a valid label
    valid_mask = np.ones(n_units, dtype=bool)
    for mname in MODEL_NAMES:
        for i in range(n_units):
            if all_model_labels[mname][i] not in LABEL_SET:
                valid_mask[i] = False

    valid_indices = np.where(valid_mask)[0]
    lines.append(f"Units with all 4 models responding: {len(valid_indices)} / {n_units}\n")

    if len(valid_indices) < 10:
        lines.append("Not enough valid units for agreement metrics.\n")
        return lines

    # Build coding matrix for valid units
    coding = np.zeros((n_models, len(valid_indices)), dtype=int)
    for mi, mname in enumerate(MODEL_NAMES):
        for vi, ui in enumerate(valid_indices):
            coding[mi, vi] = label_to_idx.get(all_model_labels[mname][ui], -1)

    # Fleiss' kappa
    n_cats = len(LABEL_SET)
    rating_matrix = np.zeros((len(valid_indices), n_cats), dtype=int)
    for vi in range(len(valid_indices)):
        for mi in range(n_models):
            rating_matrix[vi, coding[mi, vi]] += 1
    fk = fleiss_kappa(rating_matrix)
    lines.append(f"**Fleiss' Kappa**: {fk:.4f}\n")

    # Krippendorff's alpha
    ka = krippendorff_alpha_nominal(coding)
    lines.append(f"**Krippendorff's Alpha**: {ka:.4f}\n")

    # Pairwise Cohen's kappa
    lines.append("**Pairwise Cohen's Kappa**:\n")
    lines.append("| Model A | Model B | Cohen's Kappa |")
    lines.append("|---|---|---|")
    for m1, m2 in combinations(range(n_models), 2):
        y1 = coding[m1]
        y2 = coding[m2]
        ck = cohens_kappa(y1, y2)
        lines.append(f"| {MODEL_DISPLAY.get(MODEL_NAMES[m1], MODEL_NAMES[m1])} | "
                     f"{MODEL_DISPLAY.get(MODEL_NAMES[m2], MODEL_NAMES[m2])} | {ck:.4f} |")
    lines.append("")

    return lines


# ── Bias type breakdown ──────────────────────────────────────────────────────
def bias_type_breakdown(df_matching, df_independent):
    rows = []
    for approach, df, prefix_fn in [
        ("matching", df_matching, lambda mname, col: f"{mname}_label_" + {"prompt_neutral": "neutral", "prompt_stereotype": "stereo", "prompt_anti_stereotype": "anti"}[col]),
        ("independent", df_independent, lambda mname, col: f"{mname}_ind_" + col.replace("prompt_", "").replace("_trigger", "")),
    ]:
        if df is None:
            continue
        for bias_type in sorted(df["bias_type"].dropna().unique()):
            mask_bt = df["bias_type"] == bias_type
            for mname in MODEL_NAMES:
                correct = 0
                total = 0
                for col, gt_label in GROUND_TRUTH.items():
                    label_col = prefix_fn(mname, col)
                    if label_col in df.columns:
                        sub = df.loc[mask_bt, label_col]
                        valid = sub.isin(LABEL_SET)
                        total += valid.sum()
                        correct += (sub[valid] == gt_label).sum()
                acc = correct / total if total > 0 else float("nan")
                rows.append({
                    "approach": approach,
                    "bias_type": bias_type,
                    "model": mname,
                    "accuracy": round(acc, 4),
                    "n_prompts": total,
                })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching-csv", default=MATCHING_CSV)
    parser.add_argument("--independent-csv", default=INDEPENDENT_CSV)
    args = parser.parse_args()

    report_parts = ["# LLM Prompt Labeling: Agreement Report\n"]

    df_matching = None
    df_independent = None

    try:
        df_matching = pd.read_csv(args.matching_csv)
        print(f"Loaded matching results: {len(df_matching)} rows")
        report_parts.append(analyze_matching(df_matching))
    except FileNotFoundError:
        print(f"Matching CSV not found: {args.matching_csv}")
        report_parts.append("## Matching Approach\n\nNo results found.\n")

    report_parts.append("\n---\n")

    try:
        df_independent = pd.read_csv(args.independent_csv)
        print(f"Loaded independent results: {len(df_independent)} rows")
        report_parts.append(analyze_independent(df_independent))
    except FileNotFoundError:
        print(f"Independent CSV not found: {args.independent_csv}")
        report_parts.append("## Independent Approach\n\nNo results found.\n")

    # Comparison summary
    report_parts.append("\n---\n")
    report_parts.append("## Approach Comparison\n")
    report_parts.append("| Model | Matching Acc | Independent Acc |")
    report_parts.append("|---|---|---|")
    for mname in MODEL_NAMES:
        m_acc = i_acc = "N/A"
        if df_matching is not None:
            correct_col = f"{mname}_correct"
            if correct_col in df_matching.columns:
                valid = df_matching[correct_col].apply(lambda x: x in (0, 1, "0", "1", 0.0, 1.0))
                if valid.any():
                    m_acc = f"{df_matching.loc[valid, correct_col].astype(float).mean():.4f}"
        if df_independent is not None:
            tc = tt = 0
            for col, gt in GROUND_TRUTH.items():
                short = col.replace("prompt_", "").replace("_trigger", "")
                lc = f"{mname}_ind_{short}"
                if lc in df_independent.columns:
                    mask = df_independent[lc].isin(LABEL_SET)
                    tt += mask.sum()
                    tc += (df_independent.loc[mask, lc] == gt).sum()
            if tt > 0:
                i_acc = f"{tc/tt:.4f}"
        report_parts.append(f"| {MODEL_DISPLAY.get(mname, mname)} | {m_acc} | {i_acc} |")

    # Bias type breakdown
    bt_df = bias_type_breakdown(df_matching, df_independent)
    if not bt_df.empty:
        bt_df.to_csv(BIAS_TYPE_CSV, index=False)
        print(f"Saved bias-type breakdown to {BIAS_TYPE_CSV}")

    report = "\n".join(report_parts)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved report to {REPORT_FILE}")
    print("\n" + report)


if __name__ == "__main__":
    main()
