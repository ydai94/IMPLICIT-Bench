"""
Unified image generation for all 12 debiasing experiments.

Generates images for a specified experiment ID using cached LLM outputs,
extracted KG, and/or ground truth KG with the Qwen-Image model.

Usage:
    python run_experiment.py --exp-id 0 --gpu 0
    python run_experiment.py --exp-id 4 --gpu 0 --shard 0 --num-shards 8
    python run_experiment.py --exp-id 0 --gpu 0 --sample 5
"""

import argparse
import json
import os
import sys

# Parse --gpu early so CUDA_VISIBLE_DEVICES is set before torch import
def _parse_gpu():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "0"

os.environ["CUDA_VISIBLE_DEVICES"] = _parse_gpu()

import pandas as pd

from config import (
    ALPHAS, BASELINE_EXPERIMENTS, DATA_CSV, EXTRACTED_KG_CSV,
    EXPERIMENT_NAMES, LLM_OUTPUTS_DIR, SEEDS, STEERING_EXPERIMENTS,
    TAIL_STEERING_EXPERIMENTS, exp_output_dir,
)
from steering_vector import SteeringVectorEditor


def load_jsonl_lookup(path):
    """Load a JSONL file into a dict keyed by 'id'."""
    lookup = {}
    if not os.path.exists(path):
        # Try merging shard files
        base, ext = os.path.splitext(path)
        shard_files = sorted(
            f for f in os.listdir(os.path.dirname(path))
            if f.startswith(os.path.basename(base) + "_shard") and f.endswith(ext)
        )
        if shard_files:
            print(f"  Merging {len(shard_files)} shard files into {path}")
            with open(path, "w") as fout:
                for sf in shard_files:
                    sf_path = os.path.join(os.path.dirname(path), sf)
                    with open(sf_path) as fin:
                        for line in fin:
                            fout.write(line)

    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    lookup[record["id"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue
    return lookup


def image_path_baseline(output_dir, case_id, seed):
    """Path for baseline experiment images (no steering)."""
    return os.path.join(output_dir, case_id, f"seed_{seed}.png")


def image_path_steered(output_dir, case_id, alpha, seed):
    """Path for steered experiment images."""
    return os.path.join(output_dir, case_id,
                        f"steered_alpha_{alpha}", f"seed_{seed}.png")


def get_prompt_for_baseline(exp_id, row, llm_lookup):
    """Get the prompt to use for a baseline (non-steering) experiment."""
    if exp_id == 0:
        return row["prompt_neutral"]

    llm_record = llm_lookup.get(row["id"])
    if llm_record and "rewritten_prompt" in llm_record:
        return llm_record["rewritten_prompt"]

    # Fallback to neutral prompt if LLM output is missing
    print(f"    [WARN] No LLM output for {row['id'][:12]}..., using neutral")
    return row["prompt_neutral"]


def get_sv_texts(exp_id, row, extracted_kg_lookup, llm_pairs_lookup):
    """Get (stereo_text, anti_stereo_text) for steering vector computation."""
    if exp_id in (4, 6):
        # Full KG triple
        if exp_id == 4:
            kg = extracted_kg_lookup.get(row["id"], {})
            head = kg.get("extracted_head", row["head"])
            rel = kg.get("extracted_relation", row["relation"])
            stereo = kg.get("extracted_stereotype_tails", row["stereotype_tails"])
            anti = kg.get("extracted_anti_stereotype_tails",
                          row["anti_stereotype_tails"])
        else:
            head = row["head"]
            rel = row["relation"]
            stereo = row["stereotype_tails"]
            anti = row["anti_stereotype_tails"]
        return (f"{head}, {rel}, {stereo}",
                f"{head}, {rel}, {anti}")

    elif exp_id in (5, 7):
        # Tail only
        if exp_id == 5:
            kg = extracted_kg_lookup.get(row["id"], {})
            stereo = kg.get("extracted_stereotype_tails", row["stereotype_tails"])
            anti = kg.get("extracted_anti_stereotype_tails",
                          row["anti_stereotype_tails"])
        else:
            stereo = row["stereotype_tails"]
            anti = row["anti_stereotype_tails"]
        return (str(stereo), str(anti))

    elif exp_id in (8, 9):
        # LLM-generated pairs
        pair = llm_pairs_lookup.get(row["id"], {})
        stereo = pair.get("stereotype_prompt", "")
        anti = pair.get("anti_stereotype_prompt", "")
        if not stereo or not anti:
            print(f"    [WARN] No LLM pair for {row['id'][:12]}..., "
                  "using GT prompts")
            stereo = row["prompt_stereotype"]
            anti = row["prompt_anti_stereotype"]
        return (stereo, anti)

    elif exp_id == 10:
        # GT stereo/anti-stereo prompts from extracted KG's matched row
        kg = extracted_kg_lookup.get(row["id"], {})
        stereo = kg.get("extracted_prompt_stereotype", row["prompt_stereotype"])
        anti = kg.get("extracted_prompt_anti_stereotype",
                      row["prompt_anti_stereotype"])
        return (str(stereo), str(anti))

    elif exp_id == 11:
        # GT stereo/anti-stereo prompts from own row
        return (row["prompt_stereotype"], row["prompt_anti_stereotype"])

    raise ValueError(f"Unknown exp_id for SV: {exp_id}")


def run_baseline_experiment(exp_id, df, editor, output_dir, llm_lookup):
    """Run a baseline (no steering) experiment."""
    manifest_rows = []
    total = len(df)

    for case_idx, (_, row) in enumerate(df.iterrows()):
        case_id = row["id"]
        prompt = get_prompt_for_baseline(exp_id, row, llm_lookup)

        for seed in SEEDS:
            path = image_path_baseline(output_dir, case_id, seed)
            if os.path.exists(path):
                manifest_rows.append({
                    "case_id": case_id, "source": row["source"],
                    "target": row["target"], "bias_type": row["bias_type"],
                    "experiment_id": exp_id, "alpha": None,
                    "seed": seed, "image_path": path,
                    "prompt_used": prompt,
                })
                continue

            print(f"  [{case_idx+1}/{total}] {case_id[:12]}... seed={seed}")
            img = editor.generate_baseline(prompt, seed=seed)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path)

            manifest_rows.append({
                "case_id": case_id, "source": row["source"],
                "target": row["target"], "bias_type": row["bias_type"],
                "experiment_id": exp_id, "alpha": None,
                "seed": seed, "image_path": path,
                "prompt_used": prompt,
            })

        # Save manifest after each case
        if (case_idx + 1) % 10 == 0 or case_idx == total - 1:
            pd.DataFrame(manifest_rows).to_csv(
                os.path.join(output_dir, "manifest.csv"), index=False)

    return manifest_rows


def run_steering_experiment(exp_id, df, editor, output_dir,
                            extracted_kg_lookup, llm_pairs_lookup,
                            alphas=None):
    """Run a steering vector experiment."""
    is_tail = exp_id in TAIL_STEERING_EXPERIMENTS
    manifest_rows = []
    total = len(df)

    for case_idx, (_, row) in enumerate(df.iterrows()):
        case_id = row["id"]
        stereo_text, anti_text = get_sv_texts(
            exp_id, row, extracted_kg_lookup, llm_pairs_lookup)

        print(f"\n  Case {case_idx+1}/{total}: {case_id[:12]}...")
        print(f"    Stereo:  {stereo_text[:80]}...")
        print(f"    Anti:    {anti_text[:80]}...")

        # Compute steering vector
        if is_tail:
            sv = editor.compute_tail_steering_vector(stereo_text, anti_text)
            sv_mask = None
        else:
            sv, sv_mask = editor.compute_steering_vector(stereo_text, anti_text)

        for alpha in (alphas or ALPHAS):
            for seed in SEEDS:
                path = image_path_steered(output_dir, case_id, alpha, seed)
                if os.path.exists(path):
                    manifest_rows.append({
                        "case_id": case_id, "source": row["source"],
                        "target": row["target"], "bias_type": row["bias_type"],
                        "experiment_id": exp_id, "alpha": alpha,
                        "seed": seed, "image_path": path,
                        "prompt_used": row["prompt_neutral"],
                    })
                    continue

                if is_tail:
                    img = editor.generate_with_tail_steering(
                        row["prompt_neutral"], sv, alpha=alpha, seed=seed)
                else:
                    img = editor.generate_with_steering(
                        row["prompt_neutral"], sv, sv_mask,
                        alpha=alpha, seed=seed)

                os.makedirs(os.path.dirname(path), exist_ok=True)
                img.save(path)

                manifest_rows.append({
                    "case_id": case_id, "source": row["source"],
                    "target": row["target"], "bias_type": row["bias_type"],
                    "experiment_id": exp_id, "alpha": alpha,
                    "seed": seed, "image_path": path,
                    "prompt_used": row["prompt_neutral"],
                })

        # Save manifest periodically
        if (case_idx + 1) % 5 == 0 or case_idx == total - 1:
            pd.DataFrame(manifest_rows).to_csv(
                os.path.join(output_dir, "manifest.csv"), index=False)

    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description="Run debiasing experiment")
    parser.add_argument("--exp-id", type=int, required=True,
                        choices=list(EXPERIMENT_NAMES.keys()),
                        help="Experiment ID (0-11)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--sample", type=int, default=0,
                        help="Test with N cases (0=all)")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="Override alpha values for steering experiments "
                             "(default: all from config)")
    args = parser.parse_args()

    exp_id = args.exp_id
    exp_name = EXPERIMENT_NAMES[exp_id]
    output_dir = exp_output_dir(exp_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Experiment {exp_id}: {exp_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Load dataset
    df = pd.read_csv(DATA_CSV)
    if args.sample > 0:
        df = df.head(args.sample)
    if args.num_shards > 1:
        df = df.iloc[args.shard::args.num_shards].reset_index(drop=True)
        print(f"Shard {args.shard}/{args.num_shards}: {len(df)} cases")
    else:
        print(f"Loaded {len(df)} cases")

    # Load cached data as needed
    llm_lookup = {}
    extracted_kg_lookup = {}
    llm_pairs_lookup = {}

    if exp_id == 1:
        path = os.path.join(LLM_OUTPUTS_DIR, "rewrite_no_kg.jsonl")
        llm_lookup = load_jsonl_lookup(path)
        print(f"Loaded {len(llm_lookup)} rewrite (no KG) entries")
    elif exp_id == 2:
        path = os.path.join(LLM_OUTPUTS_DIR, "rewrite_extracted_kg.jsonl")
        llm_lookup = load_jsonl_lookup(path)
        print(f"Loaded {len(llm_lookup)} rewrite (extracted KG) entries")
    elif exp_id == 3:
        path = os.path.join(LLM_OUTPUTS_DIR, "rewrite_gt_kg.jsonl")
        llm_lookup = load_jsonl_lookup(path)
        print(f"Loaded {len(llm_lookup)} rewrite (GT KG) entries")

    if exp_id in (4, 5, 10):
        if os.path.exists(EXTRACTED_KG_CSV):
            ekg = pd.read_csv(EXTRACTED_KG_CSV)
            extracted_kg_lookup = ekg.set_index("id").to_dict("index")
            print(f"Loaded {len(extracted_kg_lookup)} extracted KG entries")
        else:
            print(f"WARNING: {EXTRACTED_KG_CSV} not found")

    if exp_id == 8:
        path = os.path.join(LLM_OUTPUTS_DIR, "pairs_extracted_kg.jsonl")
        llm_pairs_lookup = load_jsonl_lookup(path)
        print(f"Loaded {len(llm_pairs_lookup)} LLM pair (extracted KG) entries")
    elif exp_id == 9:
        path = os.path.join(LLM_OUTPUTS_DIR, "pairs_gt_kg.jsonl")
        llm_pairs_lookup = load_jsonl_lookup(path)
        print(f"Loaded {len(llm_pairs_lookup)} LLM pair (GT KG) entries")

    # Load image model
    print("Loading Qwen-Image model...")
    editor = SteeringVectorEditor(use_cpu_offload=args.cpu_offload)
    print("Model loaded.")

    # Run experiment
    if exp_id in BASELINE_EXPERIMENTS:
        manifest_rows = run_baseline_experiment(
            exp_id, df, editor, output_dir, llm_lookup)
    elif exp_id in STEERING_EXPERIMENTS:
        manifest_rows = run_steering_experiment(
            exp_id, df, editor, output_dir,
            extracted_kg_lookup, llm_pairs_lookup,
            alphas=args.alphas)

    # Final manifest save
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    print(f"\n{'='*60}")
    print(f"Experiment {exp_id} complete! {len(manifest_rows)} entries.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
