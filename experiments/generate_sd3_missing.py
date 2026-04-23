"""
Generate the 15 SD3 neutral images missing from exp_13_sd3_baseline.

Uses the same hyperparameters as the original CrowS-Pairs SD3 run:
  model:       /data/gpfs/projects/punim2888/models/stable-diffusion-3
  dtype:       torch.float16
  steps:       28
  guidance:    7.0
  seed_idx=n:  torch.Generator(CPU).manual_seed(42 + 1000*n)

Writes into experiment_outputs/exp_13_sd3_baseline/<case_id>/seed_<n>.png so
the existing manifest layout is preserved.

Usage:
    python generate_sd3_missing.py --gpu 0
"""

import argparse
import os
import sys

def _parse_gpu():
    for i, a in enumerate(sys.argv):
        if a == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "0"

os.environ["CUDA_VISIBLE_DEVICES"] = _parse_gpu()

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from diffusers import StableDiffusion3Pipeline  # noqa: E402

from config import DATA_CSV, exp_output_dir  # noqa: E402

EXP_ID = 13
MODEL_PATH = "/data/gpfs/projects/punim2888/models/stable-diffusion-3"
STEPS = 28
GUIDANCE = 7.0
BASE_SEED = 42
SEED_STRIDE = 1000  # match original seed_list formula


def seed_for(idx):
    return BASE_SEED + idx * SEED_STRIDE


def find_missing():
    out_dir = exp_output_dir(EXP_ID)
    data = pd.read_csv(DATA_CSV)
    missing = []
    for _, r in data.iterrows():
        for s in range(3):
            dest = os.path.join(out_dir, str(r["id"]), f"seed_{s}.png")
            if not os.path.exists(dest):
                missing.append({
                    "case_id": r["id"],
                    "seed": s,
                    "prompt_neutral": r["prompt_neutral"],
                    "dest": dest,
                })
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    missing = find_missing()
    print(f"Missing images: {len(missing)}")
    if not missing:
        return
    for m in missing:
        print(f"  {m['case_id']} seed={m['seed']}  seed_val={seed_for(m['seed'])}  prompt={m['prompt_neutral'][:60]}")

    print(f"\nLoading SD3 from {MODEL_PATH} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda:0")
    print("Model loaded.")

    for m in missing:
        gen = torch.Generator(device="cpu").manual_seed(seed_for(m["seed"]))
        img = pipe(
            prompt=m["prompt_neutral"],
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=gen,
        ).images[0]
        os.makedirs(os.path.dirname(m["dest"]), exist_ok=True)
        img.save(m["dest"])
        print(f"  wrote {m['dest']}")

    print(f"\nDone: generated {len(missing)} images.")


if __name__ == "__main__":
    main()
