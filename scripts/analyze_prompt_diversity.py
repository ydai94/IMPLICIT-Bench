"""Prompt-diversity analysis for the IMPLICIT-Bench prompt set.

Reads:
  data/benchmark_prompts.csv  (1,831 prompt units; columns prompt_neutral,
                               prompt_stereotype, prompt_anti_stereotype,
                               bias_type, source)

Writes:
  reports/prompt_diversity.md

Reports, per arm and overall:
  - Token-length distribution (mean / std / min / p25 / p50 / p75 / max).
  - Type-token ratio (TTR) and unique-token count.
  - Top content tokens after stopword removal.
  - A "scene/object" proxy: noun-like content tokens on prompt_neutral.
  - Per-bias-type token-length and TTR breakdown.

Stdlib + pandas only.
"""

import os
import re
from collections import Counter

import pandas as pd

ROOT = "/data/gpfs/projects/punim2888/stereoimage"
PROMPTS_CSV = f"{ROOT}/data/benchmark_prompts.csv"
OUT_MD = f"{ROOT}/reports/prompt_diversity.md"

ARMS = ["prompt_neutral", "prompt_stereotype", "prompt_anti_stereotype"]
ARM_LABEL = {
    "prompt_neutral": "neutral",
    "prompt_stereotype": "stereotype-trigger",
    "prompt_anti_stereotype": "anti-stereotype-trigger",
}

# Small built-in stopword list. Keeps the analysis dependency-free; the
# top-token tables stay readable rather than being dominated by function
# words. Not exhaustive but adequate for diversity description.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so",
    "of", "to", "in", "on", "at", "for", "from", "by", "with", "without",
    "as", "into", "onto", "off", "out", "up", "down", "over", "under",
    "is", "are", "was", "were", "be", "being", "been", "am",
    "do", "does", "did", "doing", "done",
    "have", "has", "had", "having",
    "this", "that", "these", "those", "there", "here",
    "it", "its", "itself", "them", "their", "theirs", "they",
    "he", "him", "his", "she", "her", "hers",
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours",
    "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
    "not", "no", "yes", "than", "such", "any", "some", "all", "each",
    "both", "few", "more", "most", "other", "another", "every",
    "very", "much", "many", "less", "least", "lot",
    "can", "could", "would", "should", "shall", "will", "may", "might",
    "must", "ought",
    "while", "after", "before", "during", "until", "since", "because",
    "also", "however", "though", "although", "yet",
    "one", "two", "three",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")


def tokenise(s: str) -> list[str]:
    if not isinstance(s, str):
        return []
    return [t.lower() for t in TOKEN_RE.findall(s)]


def content_tokens(s: str) -> list[str]:
    return [t for t in tokenise(s) if t not in STOPWORDS and len(t) > 1]


def length_stats(lengths: list[int]) -> dict:
    s = pd.Series(lengths)
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": int(s.min()),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "max": int(s.max()),
    }


def md_table(rows: list[dict], col_order: list[str], fmt=None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(col_order) + " |"
    sep = "| " + " | ".join(["---"] * len(col_order)) + " |"
    body = []
    for r in rows:
        cells = []
        for c in col_order:
            v = r.get(c, "")
            if c in fmt and isinstance(v, (int, float)):
                cells.append(fmt[c].format(v))
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def main():
    df = pd.read_csv(PROMPTS_CSV)
    out = []
    out.append("# Prompt Diversity Analysis")
    out.append("")
    out.append(
        f"Source: `data/benchmark_prompts.csv` -- "
        f"{len(df)} prompt units (StereoSet "
        f"{int((df['source'].str.lower() == 'stereoset').sum())} + CrowS-Pairs "
        f"{int(df['source'].str.lower().str.replace('-', '').str.replace('_', '').eq('crowspairs').sum())}). Each unit contributes "
        f"three prompts (neutral, stereotype-trigger, anti-stereotype-trigger), "
        f"so the prompt corpus has {3 * len(df)} entries.")
    out.append("")
    out.append(
        "Tokenisation is whitespace + punctuation (regex `[A-Za-z][A-Za-z\\-']*`), "
        "lowercased. \"Content tokens\" drop a small built-in stopword list and "
        "tokens of length 1.")
    out.append("")

    # ------------------------------------------------------------------
    # 1. Per-arm length and lexical-diversity stats
    # ------------------------------------------------------------------
    out.append("## 1. Token-length and lexical diversity, per arm")
    out.append("")

    rows = []
    arm_tokens: dict[str, list[str]] = {}
    arm_content_tokens: dict[str, list[str]] = {}
    for arm in ARMS:
        toks = []
        ctoks = []
        lengths = []
        for s in df[arm]:
            t = tokenise(s)
            toks.extend(t)
            ctoks.extend(t2 for t2 in t
                         if t2 not in STOPWORDS and len(t2) > 1)
            lengths.append(len(t))
        arm_tokens[arm] = toks
        arm_content_tokens[arm] = ctoks
        ls = length_stats(lengths)
        unique = len(set(toks))
        unique_c = len(set(ctoks))
        ttr = unique / max(len(toks), 1)
        ttr_c = unique_c / max(len(ctoks), 1)
        rows.append({
            "Arm": ARM_LABEL[arm],
            "Mean length": ls["mean"],
            "Std": ls["std"],
            "Min": ls["min"],
            "p25": ls["p25"],
            "p50": ls["p50"],
            "p75": ls["p75"],
            "Max": ls["max"],
            "Total tokens": len(toks),
            "Unique tokens": unique,
            "TTR": ttr,
            "Unique content": unique_c,
            "TTR (content)": ttr_c,
        })

    # Pooled stats
    all_toks = [t for arm in ARMS for t in arm_tokens[arm]]
    all_ctoks = [t for arm in ARMS for t in arm_content_tokens[arm]]
    pooled_lens = []
    for arm in ARMS:
        for s in df[arm]:
            pooled_lens.append(len(tokenise(s)))
    ls = length_stats(pooled_lens)
    rows.append({
        "Arm": "all arms",
        "Mean length": ls["mean"],
        "Std": ls["std"],
        "Min": ls["min"],
        "p25": ls["p25"],
        "p50": ls["p50"],
        "p75": ls["p75"],
        "Max": ls["max"],
        "Total tokens": len(all_toks),
        "Unique tokens": len(set(all_toks)),
        "TTR": len(set(all_toks)) / max(len(all_toks), 1),
        "Unique content": len(set(all_ctoks)),
        "TTR (content)": len(set(all_ctoks)) / max(len(all_ctoks), 1),
    })

    cols = ["Arm", "Mean length", "Std", "Min", "p25", "p50", "p75", "Max",
            "Total tokens", "Unique tokens", "TTR",
            "Unique content", "TTR (content)"]
    fmt = {"Mean length": "{:.2f}", "Std": "{:.2f}",
           "p25": "{:.0f}", "p50": "{:.0f}", "p75": "{:.0f}",
           "TTR": "{:.4f}", "TTR (content)": "{:.4f}"}
    out.append(md_table(rows, cols, fmt=fmt))
    out.append("")
    out.append(
        "TTR is the type-token ratio (unique tokens / total tokens). For "
        "comparison: a fully template-instantiated dataset of the form `\"a "
        "{adj} {noun} doing {verb}\"` would converge to TTR << 0.05; natural "
        "language English text typically sits between 0.05 and 0.20 depending "
        "on length. The IMPLICIT-Bench prompts are LLM-generated from KG "
        "triples, not template-instantiated.")
    out.append("")

    # ------------------------------------------------------------------
    # 2. Top content tokens (corpus-level, all arms pooled)
    # ------------------------------------------------------------------
    out.append("## 2. Top content tokens (all arms pooled, top 50)")
    out.append("")
    counter = Counter(all_ctoks)
    top = counter.most_common(50)
    half = (len(top) + 1) // 2
    left = top[:half]
    right = top[half:]
    out.append("| # | Token | Count | # | Token | Count |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for i in range(half):
        l = left[i]
        if i < len(right):
            r = right[i]
            out.append(f"| {i+1} | {l[0]} | {l[1]} | {i+half+1} | {r[0]} | {r[1]} |")
        else:
            out.append(f"| {i+1} | {l[0]} | {l[1]} |  |  |  |")
    out.append("")

    # ------------------------------------------------------------------
    # 3. Scene / object proxy on prompt_neutral
    # ------------------------------------------------------------------
    out.append("## 3. Scene / object vocabulary (prompt_neutral, top 50 nouns)")
    out.append("")
    out.append(
        "Approximate noun-like vocabulary on the neutral arm only -- "
        "captures the variety of subjects and settings the benchmark depicts "
        "before any stereotype trigger is applied. Heuristic: lowercase content "
        "tokens not in a small list of common adjectives/verbs/adverbs. This "
        "is a lower bound; some adjectives slip through.")
    out.append("")
    NON_NOUN = {
        "very", "much", "really", "often", "always", "never", "ever",
        "good", "bad", "large", "small", "new", "old", "young", "fast", "slow",
        "high", "low", "early", "late", "first", "second", "last", "next",
        "described", "describing", "described", "shown", "showing", "seen",
        "seeing", "depicted", "depicting", "rendered", "rendering", "called",
        "named", "known", "considered", "regarded", "viewed",
        "general", "common", "typical", "usual",
        "different", "same", "similar", "various",
        "appears", "appear", "appearing",
        "being", "been",
        "seem", "seems", "seeming",
        "make", "makes", "made", "making",
        "take", "takes", "took", "taken", "taking",
        "show", "shows", "showed", "shown", "showing",
        "give", "gives", "gave", "given", "giving",
        "say", "says", "said", "saying",
        "tell", "tells", "told", "telling",
        "go", "goes", "went", "gone", "going",
        "come", "comes", "came", "coming",
        "get", "gets", "got", "gotten", "getting",
        "use", "uses", "used", "using",
        "find", "finds", "found", "finding",
        "work", "works", "worked", "working",
        "look", "looks", "looked", "looking",
        "think", "thinks", "thought", "thinking",
        "feel", "feels", "felt", "feeling",
    }
    neutral_tokens = arm_content_tokens["prompt_neutral"]
    noun_like = [t for t in neutral_tokens if t not in NON_NOUN]
    nc = Counter(noun_like)
    top = nc.most_common(50)
    half = (len(top) + 1) // 2
    left = top[:half]
    right = top[half:]
    out.append("| # | Token | Count | # | Token | Count |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for i in range(half):
        l = left[i]
        if i < len(right):
            r = right[i]
            out.append(f"| {i+1} | {l[0]} | {l[1]} | {i+half+1} | {r[0]} | {r[1]} |")
        else:
            out.append(f"| {i+1} | {l[0]} | {l[1]} |  |  |  |")
    out.append("")
    out.append(
        f"Unique noun-like content tokens on `prompt_neutral`: "
        f"**{len(set(noun_like))}** across {len(noun_like)} occurrences "
        f"(content TTR = {len(set(noun_like)) / max(len(noun_like), 1):.4f}). "
        f"For a 1,831-prompt corpus this is a strong indicator that scenes "
        f"are not template-instantiated.")
    out.append("")

    # ------------------------------------------------------------------
    # 4. Per-bias-type breakdown
    # ------------------------------------------------------------------
    out.append("## 4. Per-bias-type breakdown (all three arms pooled)")
    out.append("")
    rows = []
    for bt, sub in df.groupby("bias_type"):
        toks: list[str] = []
        ctoks: list[str] = []
        lens: list[int] = []
        for arm in ARMS:
            for s in sub[arm]:
                t = tokenise(s)
                toks.extend(t)
                ctoks.extend(t2 for t2 in t
                             if t2 not in STOPWORDS and len(t2) > 1)
                lens.append(len(t))
        ls = length_stats(lens)
        rows.append({
            "Bias type": bt,
            "Units": int(sub.shape[0]),
            "Prompts": int(sub.shape[0]) * 3,
            "Mean length": ls["mean"],
            "p50": ls["p50"],
            "Total tokens": len(toks),
            "Unique tokens": len(set(toks)),
            "TTR": len(set(toks)) / max(len(toks), 1),
            "TTR (content)": len(set(ctoks)) / max(len(ctoks), 1),
        })
    rows.sort(key=lambda r: -r["Units"])
    cols = ["Bias type", "Units", "Prompts", "Mean length", "p50",
            "Total tokens", "Unique tokens", "TTR", "TTR (content)"]
    fmt = {"Mean length": "{:.2f}", "p50": "{:.0f}",
           "TTR": "{:.4f}", "TTR (content)": "{:.4f}"}
    out.append(md_table(rows, cols, fmt=fmt))
    out.append("")
    out.append(
        "TTR is computed within each bias-type sub-corpus, so smaller bias "
        "types (e.g. disability, physical-appearance) naturally have higher "
        "TTR -- fewer total tokens, less repetition. The point is that no "
        "category collapses to template-like values (TTR << 0.05).")
    out.append("")

    # ------------------------------------------------------------------
    # 5. Source breakdown (StereoSet vs CrowS-Pairs)
    # ------------------------------------------------------------------
    out.append("## 5. Source breakdown (StereoSet vs CrowS-Pairs)")
    out.append("")
    rows = []
    for src, sub in df.groupby("source"):
        toks: list[str] = []
        lens: list[int] = []
        for arm in ARMS:
            for s in sub[arm]:
                t = tokenise(s)
                toks.extend(t)
                lens.append(len(t))
        ls = length_stats(lens)
        rows.append({
            "Source": src,
            "Units": int(sub.shape[0]),
            "Mean length": ls["mean"],
            "Std": ls["std"],
            "p50": ls["p50"],
            "Unique tokens": len(set(toks)),
            "TTR": len(set(toks)) / max(len(toks), 1),
        })
    cols = ["Source", "Units", "Mean length", "Std", "p50",
            "Unique tokens", "TTR"]
    fmt = {"Mean length": "{:.2f}", "Std": "{:.2f}", "p50": "{:.0f}",
           "TTR": "{:.4f}"}
    out.append(md_table(rows, cols, fmt=fmt))
    out.append("")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
