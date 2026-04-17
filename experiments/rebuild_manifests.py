"""
Rebuild manifest.csv for each experiment from images actually on disk.

The per-shard generation overwrites manifest.csv, so only the last shard's
entries survive. This script walks the output directories and reconstructs
complete manifests.

Usage:
    python rebuild_manifests.py              # all experiments
    python rebuild_manifests.py 1 2 3        # specific experiments
"""

import os
import re
import sys

import pandas as pd

from config import (
    BASELINE_EXPERIMENTS, DATA_CSV, EXPERIMENT_NAMES,
    STEERING_EXPERIMENTS, exp_output_dir,
)


def rebuild_manifest(exp_id):
    output_dir = exp_output_dir(exp_id)
    if not os.path.isdir(output_dir):
        print(f"Exp {exp_id}: directory not found, skipping")
        return

    cases_df = pd.read_csv(DATA_CSV)
    case_lookup = cases_df.set_index("id").to_dict("index")

    rows = []
    is_baseline = exp_id in BASELINE_EXPERIMENTS
    is_steering = exp_id in STEERING_EXPERIMENTS

    for case_id in sorted(os.listdir(output_dir)):
        case_dir = os.path.join(output_dir, case_id)
        if not os.path.isdir(case_dir) or case_id == "__pycache__":
            continue

        case_info = case_lookup.get(case_id, {})

        if is_baseline:
            # Baseline: <case_id>/seed_<N>.png
            for fname in sorted(os.listdir(case_dir)):
                m = re.match(r"seed_(\d+)\.png$", fname)
                if m:
                    seed = int(m.group(1))
                    img_path = os.path.join(case_dir, fname)
                    rows.append({
                        "case_id": case_id,
                        "source": case_info.get("source", ""),
                        "target": case_info.get("target", ""),
                        "bias_type": case_info.get("bias_type", ""),
                        "experiment_id": exp_id,
                        "alpha": None,
                        "seed": seed,
                        "image_path": img_path,
                        "prompt_used": "",
                    })

        elif is_steering:
            # Steering: <case_id>/steered_alpha_<A>/seed_<N>.png
            for alpha_dir in sorted(os.listdir(case_dir)):
                am = re.match(r"steered_alpha_([\d.]+)$", alpha_dir)
                if not am:
                    continue
                alpha = float(am.group(1))
                alpha_path = os.path.join(case_dir, alpha_dir)
                if not os.path.isdir(alpha_path):
                    continue
                for fname in sorted(os.listdir(alpha_path)):
                    sm = re.match(r"seed_(\d+)\.png$", fname)
                    if sm:
                        seed = int(sm.group(1))
                        img_path = os.path.join(alpha_path, fname)
                        rows.append({
                            "case_id": case_id,
                            "source": case_info.get("source", ""),
                            "target": case_info.get("target", ""),
                            "bias_type": case_info.get("bias_type", ""),
                            "experiment_id": exp_id,
                            "alpha": alpha,
                            "seed": seed,
                            "image_path": img_path,
                            "prompt_used": "",
                        })

    manifest_path = os.path.join(output_dir, "manifest.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Exp {exp_id}: {len(rows)} entries -> {manifest_path}")


def main():
    if len(sys.argv) > 1:
        exp_ids = [int(x) for x in sys.argv[1:]]
    else:
        exp_ids = sorted(EXPERIMENT_NAMES.keys())

    for exp_id in exp_ids:
        rebuild_manifest(exp_id)


if __name__ == "__main__":
    main()
