"""Build a stable per-experiment manifest of alpha=1.0 images by disk walk.

`run_experiment.py` overwrites `manifest.csv` as its shards complete, so
eval cannot rely on it while generation is still in flight. This script
walks the on-disk layout

    <exp_output_dir>/<case_id>/steered_alpha_1.0/seed_*.png

and writes a separate, stable file `manifest_alpha1.csv` per experiment.
Filename is distinct from `manifest.csv`, so live generation jobs leave
it alone.

Usage:
    python build_alpha1_manifest.py                 # all steering exps (4-11)
    python build_alpha1_manifest.py 4 5 6           # specific subset

Re-run this whenever you want eval to pick up newly-generated images.
The eval scripts skip already-scored (case_id, alpha, seed) keys, so a
re-run + relaunch is idempotent.
"""

import os
import re
import sys

import pandas as pd

from config import (
    DATA_CSV, STEERING_EXPERIMENTS, exp_output_dir,
)

OUTPUT_NAME = "manifest_alpha1.csv"
ALPHA_DIR = "steered_alpha_1.0"


def build_for_exp(exp_id):
    output_dir = exp_output_dir(exp_id)
    if not os.path.isdir(output_dir):
        print(f"Exp {exp_id}: directory not found, skipping")
        return

    cases_df = pd.read_csv(DATA_CSV)
    case_lookup = cases_df.set_index("id").to_dict("index")

    rows = []
    for case_id in sorted(os.listdir(output_dir)):
        case_dir = os.path.join(output_dir, case_id)
        alpha_path = os.path.join(case_dir, ALPHA_DIR)
        if not os.path.isdir(alpha_path):
            continue
        case_info = case_lookup.get(case_id, {})
        for fname in sorted(os.listdir(alpha_path)):
            sm = re.match(r"seed_(\d+)\.png$", fname)
            if not sm:
                continue
            rows.append({
                "case_id": case_id,
                "source": case_info.get("source", ""),
                "target": case_info.get("target", ""),
                "bias_type": case_info.get("bias_type", ""),
                "experiment_id": exp_id,
                "alpha": 1.0,
                "seed": int(sm.group(1)),
                "image_path": os.path.join(alpha_path, fname),
                "prompt_used": "",
            })

    out_path = os.path.join(output_dir, OUTPUT_NAME)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Exp {exp_id}: {len(rows)} alpha=1.0 entries -> {out_path}")


def main():
    if len(sys.argv) > 1:
        exp_ids = [int(x) for x in sys.argv[1:]]
    else:
        exp_ids = sorted(STEERING_EXPERIMENTS)

    for exp_id in exp_ids:
        build_for_exp(exp_id)


if __name__ == "__main__":
    main()
