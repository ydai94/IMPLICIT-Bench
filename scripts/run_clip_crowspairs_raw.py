"""Compute CLIP image-embedding similarities for raw CrowS-Pairs Qwen-Image
generations (all 1,508 prompt units, before lean_stereotype filtering).

Layout:
    crows-pairs/generated_images/qwen/<id>/seed_<n>/{neutral,stereotype_trigger,
    anti_stereotype_trigger}.png

Output columns: id, seed, sim_neutral_stereotype, sim_neutral_anti_stereotype,
sim_stereotype_anti_stereotype.

Usage (two-GPU split):
    python run_clip_crowspairs_raw.py --gpu 0 --shard 0 --num-shards 2 \
        --output crows-pairs/clip_similarities_shard0.csv
    python run_clip_crowspairs_raw.py --gpu 1 --shard 1 --num-shards 2 \
        --output crows-pairs/clip_similarities_shard1.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


BASE = Path("/path/to/data")
IMAGE_BASE = BASE / "crows-pairs" / "generated_images" / "qwen"
MODEL_NAME = "openai/clip-vit-large-patch14"
MODEL_CACHE = BASE / "models" / "clip-vit-large-patch14"
IMAGE_TYPES = ["neutral", "stereotype_trigger", "anti_stereotype_trigger"]
SEEDS = range(10)


def discover_cases(image_base, shard, num_shards):
    work = []
    for d in sorted(os.listdir(image_base), key=lambda x: (len(x), x)):
        if not d.isdigit():
            continue
        cid = int(d)
        if cid % num_shards != shard:
            continue
        case_dir = image_base / d
        if not case_dir.is_dir():
            continue
        for seed in SEEDS:
            seed_dir = case_dir / f"seed_{seed}"
            if not seed_dir.is_dir():
                continue
            paths = {}
            ok = True
            for t in IMAGE_TYPES:
                p = seed_dir / f"{t}.png"
                if not p.exists():
                    ok = False
                    break
                paths[t] = p
            if ok:
                work.append((d, seed, paths))
    return work


def load_completed(out_csv):
    if not os.path.exists(out_csv):
        return set()
    try:
        df = pd.read_csv(out_csv)
    except Exception:
        return set()
    return {(str(r["id"]), int(r["seed"])) for _, r in df.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=96,
                    help="Total images per CLIP forward pass (multiple of 3).")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--flush-every", type=int, default=500)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[shard {args.shard}/{args.num_shards}] device={device}, output={args.output}", flush=True)

    work = discover_cases(IMAGE_BASE, args.shard, args.num_shards)
    print(f"[shard {args.shard}] discovered {len(work)} (id, seed) triplets", flush=True)

    completed = load_completed(args.output)
    if completed:
        before = len(work)
        work = [w for w in work if (w[0], w[1]) not in completed]
        print(f"[shard {args.shard}] resuming, {before - len(work)} already done, {len(work)} remaining", flush=True)

    if not work:
        print(f"[shard {args.shard}] nothing to do", flush=True)
        return

    print(f"[shard {args.shard}] loading CLIP model {MODEL_NAME} (cache={MODEL_CACHE})", flush=True)
    model = CLIPModel.from_pretrained(MODEL_NAME, cache_dir=str(MODEL_CACHE), torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=str(MODEL_CACHE))

    triplets_per_batch = max(1, args.batch_size // 3)
    pending = []
    total = len(work)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for batch_start in range(0, total, triplets_per_batch):
        batch_items = work[batch_start:batch_start + triplets_per_batch]
        imgs = []
        for cid, seed, paths in batch_items:
            for t in IMAGE_TYPES:
                imgs.append(Image.open(paths[t]).convert("RGB"))
        inputs = processor(images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            if torch.is_tensor(out):
                feats = out
            else:
                feats = getattr(out, "image_embeds", None)
                if feats is None:
                    feats = out.pooler_output  # transformers >=5: already projected
            emb = feats / feats.norm(dim=-1, keepdim=True)
        emb = emb.float()
        for i, (cid, seed, _) in enumerate(batch_items):
            idx = i * 3
            n, s, a = emb[idx], emb[idx + 1], emb[idx + 2]
            pending.append({
                "id": cid,
                "seed": seed,
                "sim_neutral_stereotype": round((n @ s).item(), 6),
                "sim_neutral_anti_stereotype": round((n @ a).item(), 6),
                "sim_stereotype_anti_stereotype": round((s @ a).item(), 6),
            })

        done = batch_start + len(batch_items)
        if done % 200 < triplets_per_batch or done == total:
            print(f"[shard {args.shard}] processed {done}/{total}", flush=True)

        if len(pending) >= args.flush_every or done == total:
            df = pd.DataFrame(pending)
            write_header = not os.path.exists(args.output)
            df.to_csv(args.output, mode="a", header=write_header, index=False)
            pending = []

    if pending:
        df = pd.DataFrame(pending)
        write_header = not os.path.exists(args.output)
        df.to_csv(args.output, mode="a", header=write_header, index=False)

    print(f"[shard {args.shard}] done -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
