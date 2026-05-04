"""
Evaluate SD3-generated images for stereotype bias using OpenRouter API (Qwen3-VL).

Evaluates SD3 images from StereoSet and CrowS-Pairs benchmarks that correspond
to cases in benchmark_prompts.csv.

Usage:
    python evaluate_sd3.py
    python evaluate_sd3.py --source stereoset --sample 10
    python evaluate_sd3.py --source crows-pairs --workers 16
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from config import DATA_CSV, EVAL_RESULTS_DIR
from evaluate_all import OPENROUTER_API_KEY, score_image_api

BASE_DIR = "/data/gpfs/projects/punim2888"
STEREOSET_SD3_DIR = os.path.join(BASE_DIR, "stereoset", "generated_images", "sd3")
CROWSPAIRS_SD3_DIR = os.path.join(BASE_DIR, "crows-pairs", "generated_images", "sd3")
STEREOSET_CSV = os.path.join(BASE_DIR, "stereoset", "stereotype_trigger_results.csv")

IMAGE_TYPES = ["neutral"]


def build_image_list(cases_df, id_to_split):
    """Build list of (case_id, source, image_type, seed, image_path) tuples."""
    entries = []
    for _, row in cases_df.iterrows():
        case_id = row["id"]
        source = row["source"]

        if source == "StereoSet":
            split = id_to_split.get(case_id)
            if split is None:
                continue
            case_dir = os.path.join(STEREOSET_SD3_DIR, f"{split}_{case_id}")
        else:  # CrowS-Pairs
            case_dir = os.path.join(CROWSPAIRS_SD3_DIR, str(case_id))

        if not os.path.isdir(case_dir):
            continue

        for seed_dir in sorted(os.listdir(case_dir)):
            if not seed_dir.startswith("seed_"):
                continue
            seed = int(seed_dir.split("_")[1])
            for img_type in IMAGE_TYPES:
                img_path = os.path.join(case_dir, seed_dir, f"{img_type}.png")
                if os.path.exists(img_path):
                    entries.append((case_id, source, img_type, seed, img_path))

    return entries


def process_entry(entry, case_lookup):
    """Score a single SD3 image. Returns a result dict or None."""
    case_id, source, img_type, seed, img_path = entry
    case_info = case_lookup.get(case_id, {})

    result = score_image_api(
        img_path,
        target=case_info.get("target", ""),
        bias_type=case_info.get("bias_type", ""),
        head=case_info.get("head", ""),
        relation=case_info.get("relation", ""),
        stereotype_tails=case_info.get("stereotype_tails", ""),
        anti_stereotype_tails=case_info.get("anti_stereotype_tails", ""),
    )

    return {
        "case_id": case_id,
        "source": source,
        "target": case_info.get("target", ""),
        "bias_type": case_info.get("bias_type", ""),
        "image_type": img_type,
        "seed": seed,
        "image_path": img_path,
        "score": result["score"] if result else None,
        "justification": result.get("justification", "") if result else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SD3 images for bias")
    parser.add_argument("--source", choices=["stereoset", "crows-pairs", "all"],
                        default="all", help="Which benchmark to evaluate")
    parser.add_argument("--sample", type=int, default=0,
                        help="Limit to first N cases (0 = all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel API workers")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set in config.py")
        sys.exit(1)

    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    # Load case data
    cases_df = pd.read_csv(DATA_CSV)
    if args.source == "stereoset":
        cases_df = cases_df[cases_df["source"] == "StereoSet"]
    elif args.source == "crows-pairs":
        cases_df = cases_df[cases_df["source"] == "CrowS-Pairs"]

    if args.sample > 0:
        cases_df = cases_df.head(args.sample)

    case_lookup = cases_df.set_index("id").to_dict("index")

    # Load StereoSet split mapping
    id_to_split = {}
    if os.path.exists(STEREOSET_CSV):
        stereo_df = pd.read_csv(STEREOSET_CSV)
        id_to_split = dict(zip(stereo_df["id"], stereo_df["split"]))

    # Build image list
    entries = build_image_list(cases_df, id_to_split)
    print(f"Found {len(entries)} SD3 images to evaluate "
          f"({len(cases_df)} cases, source={args.source})")

    if not entries:
        print("No images found.")
        return

    # Output path
    results_jsonl = os.path.join(EVAL_RESULTS_DIR, "sd3_eval.jsonl")

    # Load existing results for resume (skip entries with null scores so they get retried)
    processed_keys = set()
    kept_lines = []
    if os.path.exists(results_jsonl):
        with open(results_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("score") is None:
                        continue
                    key = (r["case_id"], r["image_type"], str(r["seed"]))
                    processed_keys.add(key)
                    kept_lines.append(line.strip())
                except (json.JSONDecodeError, KeyError):
                    continue
        # Rewrite JSONL without failed entries so they don't accumulate
        with open(results_jsonl, "w") as f:
            for line in kept_lines:
                f.write(line + "\n")
        print(f"Resuming: {len(processed_keys)} entries already processed")

    # Filter out already-processed entries
    to_process = [
        e for e in entries
        if (e[0], e[2], str(e[3])) not in processed_keys
    ]
    print(f"To process: {len(to_process)} images "
          f"(skipped {len(entries) - len(to_process)})")

    if not to_process:
        print("Nothing to do.")
        return

    fout = open(results_jsonl, "a")
    pbar = tqdm(total=len(to_process), desc="Eval SD3")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_entry, entry, case_lookup): entry
            for entry in to_process
        }
        for future in as_completed(futures):
            record = future.result()
            if record is not None:
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                pbar.set_postfix(
                    score=record["score"],
                    img_type=record["image_type"],
                )
            pbar.update(1)

    pbar.close()
    fout.close()

    # Convert to CSV
    results = []
    with open(results_jsonl) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if results:
        csv_path = results_jsonl.replace(".jsonl", ".csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"\nDone! {len(results)} entries -> {csv_path}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
