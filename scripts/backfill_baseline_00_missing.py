"""
Backfill baseline_00 Qwen scores in data/benchmark_scores.csv for the 441 cases listed
in experiment_outputs/exp_00_baseline_missing/manifest.csv.

Pipeline (no GPU required; OpenRouter API only):

  1. build-manifest: Build a 876-row stereo+anti manifest for the 438 CP cases.
                     image_path points to stereotype_trigger.png or
                     anti_stereotype_trigger.png; image_type column tracks which.

  2. score:          Run OpenRouter Qwen3-VL on the stereo+anti manifest,
                     writing scored results to stereo_anti_eval.{jsonl,csv}.
                     Reuses score_image_api() from experiments/evaluate_all.py.
                     Resume key is (case_id, seed, image_type) so the two image
                     types per case don't collide (vs evaluate_all.py's default
                     which keys only on (case_id, alpha, seed)).

  3. backfill:       Patch data/benchmark_scores.csv in place (with .bak):
                       - qwen_neutral   <- cache/eval_results/exp_00_eval.csv
                                          (all 441 manifest cases)
                       - qwen_stereo    <- stereo_anti_eval.csv (438 CP cases)
                       - qwen_anti      <- stereo_anti_eval.csv (438 CP cases)
                     Idempotent: only fills NaN cells.

Usage:
    # one shot, all three steps in order
    python scripts/backfill_baseline_00_missing.py

    # individual steps
    python scripts/backfill_baseline_00_missing.py --step build-manifest
    python scripts/backfill_baseline_00_missing.py --step score --workers 16
    python scripts/backfill_baseline_00_missing.py --step backfill

Requires: OPENROUTER_API_KEY (read by experiments/evaluate_all.py).
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path("/data/gpfs/projects/punim2888/stereoimage")
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from evaluate_all import score_image_api  # noqa: E402

MANIFEST_DIR = REPO_ROOT / "experiment_outputs" / "exp_00_baseline_missing"
SOURCE_MANIFEST = MANIFEST_DIR / "manifest.csv"
STEREO_ANTI_MANIFEST = MANIFEST_DIR / "stereo_anti_manifest.csv"
STEREO_ANTI_JSONL = MANIFEST_DIR / "stereo_anti_eval.jsonl"
STEREO_ANTI_CSV = MANIFEST_DIR / "stereo_anti_eval.csv"

BENCHMARK_PROMPTS = REPO_ROOT / "data" / "benchmark_prompts.csv"
BENCHMARK_SCORES = REPO_ROOT / "data" / "benchmark_scores.csv"
BENCHMARK_SCORES_BAK = REPO_ROOT / "data" / "benchmark_scores.csv.bak"
EXP00_EVAL_CSV = REPO_ROOT / "cache" / "eval_results" / "exp_00_eval.csv"

IMAGE_TYPES = ["stereotype_trigger", "anti_stereotype_trigger"]


# ---------- Step 1: build manifest ----------

def build_manifest():
    """Build the 876-row stereo+anti manifest for the 438 CP cases."""
    src = pd.read_csv(SOURCE_MANIFEST)
    cp = src[src["source"] == "CrowS-Pairs"].copy()
    print(f"Source manifest: {len(src)} rows, CP subset: {len(cp)}")

    case_meta = pd.read_csv(BENCHMARK_PROMPTS)
    case_meta["id"] = case_meta["id"].astype(str)
    prompt_lookup = case_meta.set_index("id")[
        ["prompt_stereotype", "prompt_anti_stereotype"]
    ].to_dict("index")

    rows = []
    missing_files = []
    for _, r in cp.iterrows():
        case_id = str(r["case_id"])
        for image_type in IMAGE_TYPES:
            img_path = r["image_path"].replace("neutral.png", f"{image_type}.png")
            if not os.path.exists(img_path):
                missing_files.append(img_path)
                continue
            prompts = prompt_lookup.get(case_id, {})
            prompt_used = prompts.get(
                "prompt_stereotype" if image_type == "stereotype_trigger"
                else "prompt_anti_stereotype",
                "",
            )
            rows.append({
                "case_id": case_id,
                "source": "CrowS-Pairs",
                "target": r["target"],
                "bias_type": r["bias_type"],
                "experiment_id": 0,
                "alpha": r.get("alpha", ""),
                "seed": int(r["seed"]),
                "image_path": img_path,
                "prompt_used": prompt_used,
                "image_type": image_type,
            })

    if missing_files:
        raise RuntimeError(
            f"{len(missing_files)} expected images missing. First few: "
            f"{missing_files[:3]}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(STEREO_ANTI_MANIFEST, index=False)
    print(f"Wrote {len(df)} rows -> {STEREO_ANTI_MANIFEST}")
    return df


# ---------- Step 2: score via OpenRouter ----------

def _resume_key(row):
    return (str(row["case_id"]), int(row["seed"]), row["image_type"])


def _process_row(row, case_lookup):
    case_id = str(row["case_id"])
    info = case_lookup.get(case_id, {})
    result = score_image_api(
        row["image_path"],
        target=info.get("target", ""),
        bias_type=info.get("bias_type", ""),
        head=info.get("head", ""),
        relation=info.get("relation", ""),
        stereotype_tails=info.get("stereotype_tails", ""),
        anti_stereotype_tails=info.get("anti_stereotype_tails", ""),
    )
    return {
        "case_id": case_id,
        "experiment_id": 0,
        "target": info.get("target", ""),
        "bias_type": info.get("bias_type", ""),
        "alpha": None,
        "seed": int(row["seed"]),
        "image_type": row["image_type"],
        "image_path": row["image_path"],
        "score": result["score"] if result else None,
        "justification": result.get("justification", "") if result else "",
    }


def score(workers=16):
    """Run OpenRouter on the stereo+anti manifest."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("ERROR: OPENROUTER_API_KEY not set")
    if not STEREO_ANTI_MANIFEST.exists():
        sys.exit(f"Run --step build-manifest first ({STEREO_ANTI_MANIFEST} missing)")

    manifest = pd.read_csv(STEREO_ANTI_MANIFEST)
    print(f"Loaded {len(manifest)} rows to score")

    case_lookup = (
        pd.read_csv(BENCHMARK_PROMPTS).astype({"id": str}).set_index("id").to_dict("index")
    )

    processed = set()
    if STEREO_ANTI_JSONL.exists():
        with open(STEREO_ANTI_JSONL) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("score") is None:
                        continue
                    processed.add(
                        (str(r["case_id"]), int(r["seed"]), r["image_type"])
                    )
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resume: {len(processed)} already scored")

    todo = [r for _, r in manifest.iterrows() if _resume_key(r) not in processed]
    print(f"To score: {len(todo)} (skipping {len(manifest) - len(todo)})")
    if not todo:
        _convert_jsonl_to_csv()
        return

    with open(STEREO_ANTI_JSONL, "a") as fout, \
         ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_row, r, case_lookup): r for r in todo}
        pbar = tqdm(total=len(futs), desc="OpenRouter Qwen3-VL")
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is not None:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                pbar.set_postfix(score=rec["score"])
            pbar.update(1)
        pbar.close()

    _convert_jsonl_to_csv()


def _convert_jsonl_to_csv():
    if not STEREO_ANTI_JSONL.exists():
        return
    rows = []
    with open(STEREO_ANTI_JSONL) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if rows:
        pd.DataFrame(rows).to_csv(STEREO_ANTI_CSV, index=False)
        print(f"Wrote {len(rows)} rows -> {STEREO_ANTI_CSV}")


# ---------- Step 3: backfill benchmark_scores.csv ----------

def backfill():
    """Patch benchmark_scores.csv in place. Idempotent: only fills NaN cells."""
    if not EXP00_EVAL_CSV.exists():
        sys.exit(f"Missing {EXP00_EVAL_CSV}")
    if not STEREO_ANTI_CSV.exists():
        sys.exit(f"Run --step score first ({STEREO_ANTI_CSV} missing)")

    src = pd.read_csv(SOURCE_MANIFEST)
    src["case_id"] = src["case_id"].astype(str)
    src["seed"] = src["seed"].astype(int)
    manifest_keys = set(zip(src["case_id"], src["seed"]))
    print(f"Manifest cases: {len(manifest_keys)}")

    neutral = pd.read_csv(EXP00_EVAL_CSV)
    neutral["case_id"] = neutral["case_id"].astype(str)
    neutral["seed"] = neutral["seed"].astype(int)
    neutral_lookup = neutral.set_index(["case_id", "seed"])["score"].to_dict()

    sa = pd.read_csv(STEREO_ANTI_CSV)
    sa["case_id"] = sa["case_id"].astype(str)
    sa["seed"] = sa["seed"].astype(int)
    stereo_lookup = (
        sa[sa["image_type"] == "stereotype_trigger"]
        .set_index(["case_id", "seed"])["score"].to_dict()
    )
    anti_lookup = (
        sa[sa["image_type"] == "anti_stereotype_trigger"]
        .set_index(["case_id", "seed"])["score"].to_dict()
    )
    print(f"  exp_00_eval: {len(neutral_lookup)} (case,seed) pairs available")
    print(f"  stereo:      {len(stereo_lookup)}")
    print(f"  anti:        {len(anti_lookup)}")

    if not BENCHMARK_SCORES_BAK.exists():
        import shutil
        shutil.copy2(BENCHMARK_SCORES, BENCHMARK_SCORES_BAK)
        print(f"Backed up -> {BENCHMARK_SCORES_BAK}")

    df = pd.read_csv(BENCHMARK_SCORES)
    df["id"] = df["id"].astype(str)
    df["seed"] = df["seed"].astype(int)

    before = {c: int(df[c].isna().sum()) for c in
              ["qwen_neutral", "qwen_stereo", "qwen_anti"]}

    in_manifest = df.apply(lambda r: (r["id"], r["seed"]) in manifest_keys, axis=1)

    def _fill(col, lookup):
        n = 0
        for idx in df.index[in_manifest & df[col].isna()]:
            key = (df.at[idx, "id"], df.at[idx, "seed"])
            if key in lookup and pd.notna(lookup[key]):
                df.at[idx, col] = lookup[key]
                n += 1
        return n

    n_neut = _fill("qwen_neutral", neutral_lookup)
    n_ster = _fill("qwen_stereo", stereo_lookup)
    n_anti = _fill("qwen_anti", anti_lookup)

    df.to_csv(BENCHMARK_SCORES, index=False)

    after = {c: int(df[c].isna().sum()) for c in
             ["qwen_neutral", "qwen_stereo", "qwen_anti"]}

    print("\nNaN counts before -> after:")
    for c in ["qwen_neutral", "qwen_stereo", "qwen_anti"]:
        print(f"  {c}: {before[c]} -> {after[c]}")
    print(f"\nFilled: qwen_neutral={n_neut}, qwen_stereo={n_ster}, qwen_anti={n_anti}")
    print(f"Wrote {BENCHMARK_SCORES}")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["build-manifest", "score", "backfill", "all"],
        default="all",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if args.step in ("build-manifest", "all"):
        build_manifest()
    if args.step in ("score", "all"):
        score(workers=args.workers)
    if args.step in ("backfill", "all"):
        backfill()


if __name__ == "__main__":
    main()
