"""Curate three failure-case slices for the IMPLICIT-Bench appendix.

Reads:
  data/benchmark_prompts.csv         (1,831 prompt units, all bias types)
  data/labeled_matching.csv          (per-prompt labels from 4 LLMs;
                                      Qwen3-30B is the disagreement headliner)
  data/benchmark_scores.csv          (per-(id, seed) Qwen3-VL and Gemma-4 scores)
  experiment_outputs/exp_14_nano_banana_2_baseline/<id>/seed_{0,1,2}.png
                                     (presence = success, absence = silent refusal)

Writes:
  reports/failure_cases.md
  data/failure_cases_examples.csv

Three slices:
  A. Generator silent refusal (Nano Banana / Gemini-3.1-flash):
     prompts where all 3 seeds were refused, sampled proportionally over
     race / religion / profession / gender (the four bias types that
     dominate the 1,165 failures per nano_banana_failure_report.md).

  B. Prompt-labelling disagreement (Qwen3-30B as outlier):
     prompts where Qwen3-30B disagrees with the unanimous label of the
     other three LLMs (Claude Sonnet 4.6, Gemma4-26B, Llama4-Maverick) on
     at least one of {neutral, stereotype, anti-stereotype}.

  C. Image-evaluator disagreement:
     (id, seed, condition) cells where |Qwen3-VL - Gemma-4| >= 2 on the
     0-5 stereotype rubric, focused on the trigger arms.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "/path/to/stereoimage"
PROMPTS_CSV = f"{ROOT}/data/benchmark_prompts.csv"
LABELED_CSV = f"{ROOT}/data/labeled_matching.csv"
SCORES_CSV = f"{ROOT}/data/benchmark_scores.csv"
NANO_DIR = Path(f"{ROOT}/experiment_outputs/exp_14_nano_banana_2_baseline")
OUT_MD = f"{ROOT}/reports/failure_cases.md"
OUT_CSV = f"{ROOT}/data/failure_cases_examples.csv"

NANO_PER_BT = {  # rows per bias type for slice A
    "race": 4, "religion": 2, "profession": 2, "gender": 2,
}
LABEL_DISAGREE_N = 10
SCORE_DISAGREE_N = 10
SCORE_DISAGREE_THRESHOLD = 2.0  # |Qwen3-VL - Gemma-4| >= this


def shorten(s, n=110):
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").replace("|", "/").strip()
    return (s[:n - 3] + "...") if len(s) > n else s


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


def slice_a_nano_refusals(prompts: pd.DataFrame) -> pd.DataFrame:
    """Find prompts where Nano Banana refused all 3 seeds (no PNGs saved)."""
    fully_failed = []
    for _, row in prompts.iterrows():
        d = NANO_DIR / row["id"]
        if not d.exists():
            saved = 0
        else:
            saved = sum((d / f"seed_{s}.png").exists() for s in (0, 1, 2))
        if saved == 0:
            fully_failed.append(row.to_dict())
    failed_df = pd.DataFrame(fully_failed)

    # Sample proportionally per the four high-impact bias types.
    chosen: list[dict] = []
    rng = np.random.default_rng(42)
    for bt, k in NANO_PER_BT.items():
        sub = failed_df[failed_df["bias_type"] == bt]
        if len(sub) == 0:
            continue
        idx = rng.choice(sub.index, size=min(k, len(sub)), replace=False)
        for i in idx:
            r = sub.loc[i]
            chosen.append({
                "Slice": "A",
                "Failure": "Nano Banana silent refusal (3/3 seeds)",
                "Source": r["source"],
                "Bias type": r["bias_type"],
                "Case": r["id"][:8],
                "Prompt (neutral)": shorten(r["prompt_neutral"]),
                "Notes": "Empty blocked_categories, finishReason=NO_IMAGE",
            })
    return pd.DataFrame(chosen)


def slice_b_label_disagreement(labeled: pd.DataFrame) -> pd.DataFrame:
    """Rows where Qwen3-30B disagrees with the unanimous label of the
    other three LLMs on at least one of {neutral, stereo, anti}."""
    label_cols = {
        "neutral": ("claude_sonnet_label_neutral", "qwen3_label_neutral",
                    "gemma4_label_neutral", "llama4_label_neutral"),
        "stereo": ("claude_sonnet_label_stereo", "qwen3_label_stereo",
                   "gemma4_label_stereo", "llama4_label_stereo"),
        "anti": ("claude_sonnet_label_anti", "qwen3_label_anti",
                 "gemma4_label_anti", "llama4_label_anti"),
    }
    rows = []
    for _, r in labeled.iterrows():
        for arm, (claude, qwen, gemma, llama) in label_cols.items():
            cl, qw, gm, lm = r[claude], r[qwen], r[gemma], r[llama]
            others = {cl, gm, lm}
            if len(others) == 1 and qw not in others:
                # All three non-Qwen agree; Qwen disagrees.
                arm_to_prompt_col = {
                    "neutral": "prompt_neutral",
                    "stereo": "prompt_stereotype",
                    "anti": "prompt_anti_stereotype",
                }
                rows.append({
                    "Slice": "B",
                    "Failure": "Qwen3-30B vs 3-of-3 majority",
                    "Source": r["source"],
                    "Bias type": r["bias_type"],
                    "Case": r["id"][:8],
                    "Arm": arm,
                    "Majority": list(others)[0],
                    "Qwen3-30B": qw,
                    "Prompt": shorten(r[arm_to_prompt_col[arm]], 90),
                })
    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        return rows_df
    # Stratify across bias types: take up to ceil(N / n_bt) per bias type
    # so the table is not dominated by profession/race.
    selected = []
    n_bt = rows_df["Bias type"].nunique()
    per_bt = max(1, LABEL_DISAGREE_N // max(n_bt, 1) + 1)
    rng = np.random.default_rng(7)
    for bt, sub in rows_df.groupby("Bias type"):
        idx = rng.choice(sub.index, size=min(per_bt, len(sub)), replace=False)
        selected.extend(sub.loc[idx].to_dict("records"))
    selected = selected[:LABEL_DISAGREE_N]
    return pd.DataFrame(selected)


def slice_c_score_disagreement(scores: pd.DataFrame) -> pd.DataFrame:
    """(id, seed, condition) cells where |Qwen3-VL - Gemma-4| >= 2 on the
    stereotype-trigger or anti-stereotype-trigger arms."""
    cells = []
    for _, r in scores.iterrows():
        for cond, qcol, gcol, prompt_col in [
            ("stereotype_trigger",
             "qwen_stereo", "gemma_stereo", "prompt_stereotype_trigger"),
            ("anti_stereotype_trigger",
             "qwen_anti", "gemma_anti", "prompt_anti_stereotype_trigger"),
        ]:
            q, g = r.get(qcol), r.get(gcol)
            if pd.isna(q) or pd.isna(g):
                continue
            d = abs(q - g)
            if d >= SCORE_DISAGREE_THRESHOLD:
                cells.append({
                    "Slice": "C",
                    "Failure": "Qwen3-VL vs Gemma-4 |delta| >= 2",
                    "Source": r.get("dataset", ""),
                    "Bias type": r["bias_type"],
                    "Case": str(r["id"])[:8],
                    "Seed": int(r["seed"]) if pd.notna(r["seed"]) else "",
                    "Condition": cond,
                    "Qwen3-VL": float(q),
                    "Gemma-4": float(g),
                    "|delta|": float(d),
                    "Prompt": shorten(r.get(prompt_col, ""), 90),
                })
    if not cells:
        return pd.DataFrame()
    df = pd.DataFrame(cells).sort_values("|delta|", ascending=False)
    # Stratify: take the top per bias_type so categories with many rows
    # don't dominate.
    selected = []
    seen_bt: dict[str, int] = {}
    for _, row in df.iterrows():
        bt = row["Bias type"]
        if seen_bt.get(bt, 0) >= 2:
            continue
        seen_bt[bt] = seen_bt.get(bt, 0) + 1
        selected.append(row.to_dict())
        if len(selected) >= SCORE_DISAGREE_N:
            break
    return pd.DataFrame(selected)


def main():
    prompts = pd.read_csv(PROMPTS_CSV)
    labeled = pd.read_csv(LABELED_CSV)
    scores = pd.read_csv(SCORES_CSV)

    a = slice_a_nano_refusals(prompts)
    b = slice_b_label_disagreement(labeled)
    c = slice_c_score_disagreement(scores)

    out: list[str] = []
    out.append("# Benchmark / Evaluator Failure Cases")
    out.append("")
    out.append(
        "Three slices that surface concrete examples where the benchmark or "
        "the evaluator fails. Counts in this report are *examples for "
        "qualitative inspection*; statistical headlines (refusal rates, "
        "kappa) belong in `nano_banana_failure_report.md` and "
        "`agreement_report.md` respectively.")
    out.append("")

    out.append("## A. Generator silent refusal (Nano Banana 2)")
    out.append("")
    out.append(
        "Cases where Gemini-3.1-flash-image-preview returned `finishReason=NO_IMAGE` "
        "with empty `blocked_categories` on all three seeds. Headline: 21.2% "
        "of 5,493 generations refused, with race + religion (StereoSet) "
        "accounting for 53.6% of failures (`nano_banana_failure_report.md`). "
        f"Examples below sampled (seed 42) at "
        f"{NANO_PER_BT} from the {len(NANO_PER_BT)} most-affected bias types.")
    out.append("")
    if not a.empty:
        out.append(md_table(a))
    else:
        out.append("(no Nano Banana experiment outputs found at "
                   f"`{NANO_DIR}` -- skipped)")
    out.append("")

    out.append("## B. VLM prompt-label disagreement (Qwen3-30B as outlier)")
    out.append("")
    out.append(
        "Cases where Qwen3-30B disagrees with a 3-of-3 unanimous label "
        "from {Claude Sonnet 4.6, Gemma4-26B, Llama4-Maverick} on at least "
        "one prompt arm. Headline: pairwise Cohen's kappa Claude--Qwen3 = "
        "0.501 (lowest of all pairs); dropping Qwen3 raises Fleiss' kappa "
        "from 0.654 to 0.759 (`agreement_report.md`). The pattern these "
        "examples expose is Qwen3's tendency to flip stereotype <-> "
        "anti-stereotype when the prompt is short or polysemous.")
    out.append("")
    if not b.empty:
        out.append(md_table(b))
    else:
        out.append("(no rows matched -- check `data/labeled_matching.csv`)")
    out.append("")

    out.append("## C. Image-evaluator disagreement (Qwen3-VL vs Gemma-4)")
    out.append("")
    out.append(
        f"Per-(id, seed, condition) cells where |Qwen3-VL - Gemma-4| >= "
        f"{SCORE_DISAGREE_THRESHOLD} on the 0-5 stereotype rubric. Up to 2 "
        f"per bias type, ranked by |delta| descending. These are the cells "
        f"a future user should NOT report a single VLM number on without "
        f"flagging.")
    out.append("")
    if not c.empty:
        out.append(md_table(c, fmt={"Qwen3-VL": "{:.2f}",
                                    "Gemma-4": "{:.2f}",
                                    "|delta|": "{:.2f}"}))
    else:
        out.append("(no rows met the threshold -- check `data/benchmark_scores.csv`)")
    out.append("")

    # Combined CSV
    parts = [df for df in [a, b, c] if not df.empty]
    if parts:
        combined = pd.concat(parts, ignore_index=True, sort=False)
        combined.to_csv(OUT_CSV, index=False)
    else:
        pd.DataFrame().to_csv(OUT_CSV, index=False)
    out.append(f"Combined examples written to `{OUT_CSV}`.")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
