"""
GPT-Image 2 generation driver for neutral-prompt bias benchmark (exp 12).

Modes:
  sync  : one /v1/images/generations call per (case, seed). Good for small
          samples (~$0.006/img at low quality) and quick smoke tests.
  batch : submit a single Batch API job containing all (case, seed) pairs.
          50% cheaper (~$0.003/img), <=24h turnaround. Use for full runs.

Output layout matches the baseline flat layout that rebuild_manifests.py
expects (seed_N.png under each case dir), so no manifest-code changes needed.

Usage:
  python run_gpt_image.py --mode sync --sample 10
  python run_gpt_image.py --mode batch-submit
  python run_gpt_image.py --mode batch-check --batch-id batch_abc123
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm

from config import (
    CACHE_DIR, DATA_CSV, EXPERIMENT_NAMES, SEEDS, exp_output_dir,
)

EXP_ID = 12
MODEL = "gpt-image-2"
SIZE = "1024x1024"
QUALITY = "low"
IMAGE_ENDPOINT = "/v1/images/generations"

BATCH_META_DIR = os.path.join(CACHE_DIR, "gpt_image_batches")
# OpenAI Batch API per-file limit: 50k requests / 200 MB.
BATCH_MAX_REQUESTS = 50000


def load_env_file():
    """Mirror evaluate_all.py's .env loader so the same key works."""
    env_path = Path("/data/gpfs/projects/punim2888/stereoset/backup/"
                    "stereoset-augment/.env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def get_client():
    load_env_file()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("ERROR: OPENAI_API_KEY not set in environment or .env",
              file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=key)


def image_path_for(output_dir, case_id, seed):
    return os.path.join(output_dir, case_id, f"seed_{seed}.png")


def iter_jobs(df, output_dir, skip_existing=True):
    """Yield (case_id, prompt, seed, target_path) for every job to run."""
    for _, row in df.iterrows():
        case_id = row["id"]
        prompt = row["prompt_neutral"]
        for seed in SEEDS:
            path = image_path_for(output_dir, case_id, seed)
            if skip_existing and os.path.exists(path):
                continue
            yield case_id, prompt, seed, path


def save_b64_png(b64_str, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))


# ---------------- Sync mode ----------------

def _generate_one(client, case_id, prompt, seed, path, max_retries=5):
    """Generate a single image with retry. Returns (case_id, seed, ok: bool)."""
    for attempt in range(max_retries):
        try:
            resp = client.images.generate(
                model=MODEL, prompt=prompt, size=SIZE,
                quality=QUALITY, n=1,
            )
            save_b64_png(resp.data[0].b64_json, path)
            return (case_id, seed, True)
        except RateLimitError:
            wait = min(2 ** attempt * 5, 60)
            tqdm.write(f"  429 {case_id[:12]} seed={seed}, sleeping {wait}s")
            time.sleep(wait)
        except APIError as e:
            wait = 2 ** attempt * 2
            tqdm.write(f"  API error {case_id[:12]} seed={seed}: "
                       f"{e!r}, retry in {wait}s")
            time.sleep(wait)
    tqdm.write(f"  GAVE UP {case_id[:12]} seed={seed}")
    return (case_id, seed, False)


def run_sync(client, df, output_dir, workers=1, max_retries=5):
    jobs = list(iter_jobs(df, output_dir))
    if not jobs:
        print("Nothing to do (all images already exist).")
        return
    print(f"Sync: {len(jobs)} images to generate with {workers} worker(s) "
          f"(est. ${len(jobs) * 0.006:.2f} at low quality)")

    if workers <= 1:
        for case_id, prompt, seed, path in tqdm(jobs, desc="generate"):
            _generate_one(client, case_id, prompt, seed, path, max_retries)
        return

    ok_count = fail_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_generate_one, client, cid, prm, sd, pth,
                            max_retries): (cid, sd)
            for cid, prm, sd, pth in jobs
        }
        pbar = tqdm(total=len(jobs), desc="generate")
        for fut in as_completed(futures):
            _, _, ok = fut.result()
            if ok:
                ok_count += 1
            else:
                fail_count += 1
            pbar.set_postfix(ok=ok_count, fail=fail_count)
            pbar.update(1)
        pbar.close()

    if fail_count:
        print(f"\nWARNING: {fail_count} images failed permanently. "
              "Re-run to retry (existing files are skipped).")


# ---------------- Batch submit ----------------

def build_batch_jsonl(df, output_dir, jsonl_path):
    """Write a batch-input JSONL; return list of (custom_id, target_path)."""
    custom_map = []
    with open(jsonl_path, "w") as f:
        for case_id, prompt, seed, path in iter_jobs(df, output_dir):
            custom_id = f"{case_id}__seed_{seed}"
            body = {
                "model": MODEL,
                "prompt": prompt,
                "size": SIZE,
                "quality": QUALITY,
                "n": 1,
            }
            f.write(json.dumps({
                "custom_id": custom_id,
                "method": "POST",
                "url": IMAGE_ENDPOINT,
                "body": body,
            }) + "\n")
            custom_map.append((custom_id, path))
    return custom_map


def run_batch_submit(client, df, output_dir):
    os.makedirs(BATCH_META_DIR, exist_ok=True)
    jsonl_path = os.path.join(BATCH_META_DIR, "pending_input.jsonl")
    custom_map = build_batch_jsonl(df, output_dir, jsonl_path)

    if not custom_map:
        print("Nothing to submit (all images already exist).")
        return

    if len(custom_map) > BATCH_MAX_REQUESTS:
        print(f"ERROR: {len(custom_map)} > {BATCH_MAX_REQUESTS} requests; "
              "split into multiple batches.", file=sys.stderr)
        sys.exit(2)

    print(f"Uploading batch input ({len(custom_map)} requests, "
          f"est. ${len(custom_map) * 0.003:.2f} at low quality batch)...")
    up = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    print(f"  file_id={up.id}")

    batch = client.batches.create(
        input_file_id=up.id,
        endpoint=IMAGE_ENDPOINT,
        completion_window="24h",
        metadata={"project": "stereoimage", "exp_id": str(EXP_ID)},
    )
    print(f"  batch_id={batch.id}  status={batch.status}")

    # Persist the custom_id -> target-path mapping for later download.
    meta_path = os.path.join(BATCH_META_DIR, f"{batch.id}.meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "batch_id": batch.id,
            "input_file_id": up.id,
            "custom_map": dict(custom_map),
        }, f, indent=2)
    print(f"  meta saved -> {meta_path}")
    print(f"\nCheck with:\n  python run_gpt_image.py "
          f"--mode batch-check --batch-id {batch.id}")


# ---------------- Batch check / download ----------------

def run_batch_check(client, batch_id):
    meta_path = os.path.join(BATCH_META_DIR, f"{batch_id}.meta.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: no meta file for {batch_id} at {meta_path}",
              file=sys.stderr)
        sys.exit(2)
    with open(meta_path) as f:
        meta = json.load(f)
    custom_map = meta["custom_map"]

    batch = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(f"batch_id={batch_id}  status={batch.status}  "
          f"completed={counts.completed}/{counts.total}  "
          f"failed={counts.failed}")

    # Surface timing info
    import datetime as _dt
    for field in ("created_at", "in_progress_at", "finalizing_at",
                  "completed_at", "failed_at", "expired_at", "cancelled_at"):
        ts = getattr(batch, field, None)
        if ts:
            print(f"  {field:16s} {_dt.datetime.fromtimestamp(ts).isoformat()}")

    # Terminal failure states — surface errors and stop, don't pretend we're waiting
    if batch.status in ("failed", "expired", "cancelled"):
        print(f"\nBatch is {batch.status.upper()}. No images will be produced.")
        if batch.errors and batch.errors.data:
            print(f"First {min(3, len(batch.errors.data))} validation errors:")
            for err in batch.errors.data[:3]:
                print(f"  line={err.line} code={err.code}  {err.message}")
            if len(batch.errors.data) > 3:
                print(f"  ... and {len(batch.errors.data) - 3} more")
        if batch.error_file_id:
            print(f"  error_file_id={batch.error_file_id}")
        sys.exit(4)

    if batch.status not in ("completed", "finalizing"):
        print("Still running; re-run later.")
        return

    if not batch.output_file_id:
        print("ERROR: no output_file_id on batch", file=sys.stderr)
        if batch.error_file_id:
            print(f"  error_file_id={batch.error_file_id}")
        sys.exit(3)

    print(f"Downloading output file {batch.output_file_id}...")
    content = client.files.content(batch.output_file_id).read()
    lines = content.decode("utf-8").splitlines()
    print(f"  {len(lines)} result lines")

    saved = failed = 0
    for line in lines:
        rec = json.loads(line)
        cid = rec.get("custom_id")
        target = custom_map.get(cid)
        if not target:
            continue
        resp = rec.get("response") or {}
        if resp.get("status_code") != 200:
            failed += 1
            continue
        body = resp.get("body") or {}
        data = body.get("data") or []
        if not data or "b64_json" not in data[0]:
            failed += 1
            continue
        save_b64_png(data[0]["b64_json"], target)
        saved += 1

    if batch.error_file_id:
        err_path = os.path.join(BATCH_META_DIR, f"{batch_id}.errors.jsonl")
        err_content = client.files.content(batch.error_file_id).read()
        with open(err_path, "wb") as f:
            f.write(err_content)
        print(f"  errors written to {err_path}")

    print(f"Saved {saved} images, {failed} failures.")


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["sync", "batch-submit", "batch-check"])
    ap.add_argument("--sample", type=int, default=0,
                    help="Only first N prompts (for cheap smoke tests)")
    ap.add_argument("--batch-id", default=None,
                    help="Required for --mode batch-check")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent API calls for sync mode (default 1). "
                         "Set to roughly your tier IPM limit divided by 2.")
    args = ap.parse_args()

    output_dir = exp_output_dir(EXP_ID)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exp {EXP_ID}: {EXPERIMENT_NAMES[EXP_ID]}")
    print(f"Output dir: {output_dir}")

    client = get_client()

    if args.mode == "batch-check":
        if not args.batch_id:
            print("ERROR: --batch-id required with --mode batch-check",
                  file=sys.stderr)
            sys.exit(1)
        run_batch_check(client, args.batch_id)
        return

    df = pd.read_csv(DATA_CSV)
    if args.sample > 0:
        df = df.head(args.sample)
    print(f"Loaded {len(df)} prompts x {len(SEEDS)} seeds = "
          f"{len(df) * len(SEEDS)} target images")

    if args.mode == "sync":
        run_sync(client, df, output_dir, workers=args.workers)
    elif args.mode == "batch-submit":
        run_batch_submit(client, df, output_dir)


if __name__ == "__main__":
    main()
