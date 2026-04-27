"""
Nano Banana 2 (gemini-3.1-flash-image-preview) generation driver for the
neutral-prompt bias benchmark (exp 14). Mirror of run_gpt_image.py.

Modes:
  sync         : one generate_content call per (case, seed). ~$0.045/img at 512.
  batch-submit : submit a single Gemini Batch API job for all (case, seed)
                 pairs. 50% cheaper (~$0.022/img), <=24h turnaround.
  batch-check  : poll an existing batch and download finished images.

Output layout matches the baseline flat layout (seed_N.png under each case
dir) so rebuild_manifests.py / eval_alignment work unchanged.

Usage:
  python run_nano_banana.py --mode sync --sample 5
  python run_nano_banana.py --mode batch-submit
  python run_nano_banana.py --mode batch-check --batch-id batches/abc123
"""

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from config import (
    CACHE_DIR, DATA_CSV, EXPERIMENT_NAMES, SEEDS, exp_output_dir,
)

EXP_ID = 14
MODEL = "gemini-3.1-flash-image-preview"
IMAGE_SIZE = "1K"  # 1024x1024. Use "512" for half-res, "2K" or "4K" for higher.
ASPECT_RATIO = "1:1"

BATCH_META_DIR = os.path.join(CACHE_DIR, "nano_banana_batches")


def load_env_file():
    """Load API keys from the project-local .env (stereoimage/.env)."""
    env_path = Path("/data/gpfs/projects/punim2888/stereoimage/.env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def get_client():
    load_env_file()
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("ERROR: GEMINI_API_KEY not set in environment or .env",
              file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=key)


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


def save_png_bytes(data, path):
    """Save image bytes as PNG. Gemini returns JPEG; transcode to match the
    .png extension and stay consistent with the gpt-image-2 baseline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.open(io.BytesIO(data))
    if img.format != "PNG":
        img.save(path, format="PNG")
    else:
        with open(path, "wb") as f:
            f.write(data)


def _extract_image_bytes(response):
    """Return raw image bytes from a generate_content response, or None."""
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data
    return None


# ---------------- Sync mode ----------------

def _generate_one(client, case_id, prompt, seed, path, max_retries=5):
    """Generate a single image with retry. Returns (case_id, seed, ok: bool)."""
    cfg = types.GenerateContentConfig(
        response_modalities=["Image"],
        image_config=types.ImageConfig(
            aspect_ratio=ASPECT_RATIO,
            image_size=IMAGE_SIZE,
        ),
    )
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[prompt],
                config=cfg,
            )
            data = _extract_image_bytes(resp)
            if data is None:
                tqdm.write(f"  no image part for {case_id[:12]} seed={seed} "
                           f"(possibly safety blocked)")
                return (case_id, seed, False)
            save_png_bytes(data, path)
            return (case_id, seed, True)
        except genai_errors.ClientError as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status == 429:
                wait = min(2 ** attempt * 5, 60)
                tqdm.write(f"  429 {case_id[:12]} seed={seed}, sleeping {wait}s")
                time.sleep(wait)
                continue
            tqdm.write(f"  client error {case_id[:12]} seed={seed}: {e!r}")
            return (case_id, seed, False)
        except genai_errors.ServerError as e:
            wait = 2 ** attempt * 2
            tqdm.write(f"  server error {case_id[:12]} seed={seed}: "
                       f"{e!r}, retry in {wait}s")
            time.sleep(wait)
        except genai_errors.APIError as e:
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
          f"(est. ${len(jobs) * 0.067:.2f} sync at {IMAGE_SIZE})")

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
    """Write a batch-input JSONL; return list of (key, target_path)."""
    custom_map = []
    with open(jsonl_path, "w") as f:
        for case_id, prompt, seed, path in iter_jobs(df, output_dir):
            key = f"{case_id}__seed_{seed}"
            request_body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generation_config": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {
                        "aspectRatio": ASPECT_RATIO,
                        "imageSize": IMAGE_SIZE,
                    },
                },
            }
            f.write(json.dumps({"key": key, "request": request_body}) + "\n")
            custom_map.append((key, path))
    return custom_map


def run_batch_submit(client, df, output_dir, sample=0):
    os.makedirs(BATCH_META_DIR, exist_ok=True)
    tag = f"sample{sample}" if sample > 0 else "full"
    jsonl_path = os.path.join(BATCH_META_DIR, f"pending_input_{tag}.jsonl")
    custom_map = build_batch_jsonl(df, output_dir, jsonl_path)

    if not custom_map:
        print("Nothing to submit (all images already exist).")
        return

    print(f"Uploading batch input ({len(custom_map)} requests, "
          f"est. ${len(custom_map) * 0.034:.2f} batch at {IMAGE_SIZE})...")
    uploaded = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(
            display_name=f"nano-banana-bias-{tag}",
            mime_type="jsonl",
        ),
    )
    print(f"  file={uploaded.name}")

    batch_job = client.batches.create(
        model=MODEL,
        src=uploaded.name,
        config={"display_name": f"stereoimage-exp{EXP_ID}-{tag}"},
    )
    print(f"  batch_name={batch_job.name}  state={batch_job.state.name}")

    safe_id = batch_job.name.replace("/", "_")
    meta_path = os.path.join(BATCH_META_DIR, f"{safe_id}.meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "batch_name": batch_job.name,
            "input_file": uploaded.name,
            "custom_map": dict(custom_map),
        }, f, indent=2)
    print(f"  meta saved -> {meta_path}")
    print(f"\nCheck with:\n  python experiments/run_nano_banana.py "
          f"--mode batch-check --batch-id {batch_job.name}")


# ---------------- Batch check / download ----------------

def run_batch_check(client, batch_id):
    safe_id = batch_id.replace("/", "_")
    meta_path = os.path.join(BATCH_META_DIR, f"{safe_id}.meta.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: no meta file for {batch_id} at {meta_path}",
              file=sys.stderr)
        sys.exit(2)
    with open(meta_path) as f:
        meta = json.load(f)
    custom_map = meta["custom_map"]

    batch_job = client.batches.get(name=batch_id)
    state = batch_job.state.name
    print(f"batch_name={batch_id}  state={state}")

    terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    if state not in terminal:
        print("Still running; re-run later.")
        return

    if state != "JOB_STATE_SUCCEEDED":
        err = getattr(batch_job, "error", None)
        print(f"\nBatch is {state}. No images will be produced.")
        if err is not None:
            print(f"  error: {err}")
        sys.exit(4)

    dest = batch_job.dest
    result_file = getattr(dest, "file_name", None)
    if not result_file:
        print("ERROR: no dest.file_name on succeeded batch", file=sys.stderr)
        sys.exit(3)

    print(f"Downloading result file {result_file}...")
    raw = client.files.download(file=result_file)
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    lines = text.splitlines()
    print(f"  {len(lines)} result lines")

    import base64 as _b64
    saved = failed = 0
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        key = rec.get("key")
        target = custom_map.get(key)
        if not target:
            continue

        # Per-line errors come back as `error` instead of `response`.
        err = rec.get("error")
        if err:
            failed += 1
            print(f"  FAIL {key}: error={err}")
            continue

        resp = rec.get("response") or {}
        candidates = resp.get("candidates") or []
        data_b64 = None
        finish_reason = None
        safety = None
        for cand in candidates:
            finish_reason = cand.get("finishReason") or cand.get("finish_reason")
            safety = cand.get("safetyRatings") or cand.get("safety_ratings")
            for part in (cand.get("content") or {}).get("parts") or []:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    data_b64 = inline["data"]
                    break
            if data_b64:
                break

        if not data_b64:
            failed += 1
            blocked = [s for s in (safety or []) if s.get("blocked")]
            print(f"  FAIL {key}: finishReason={finish_reason} "
                  f"prompt_feedback={resp.get('promptFeedback')} "
                  f"blocked_categories={[s.get('category') for s in blocked]}")
            continue

        save_png_bytes(_b64.b64decode(data_b64), target)
        saved += 1

    print(f"Saved {saved} images, {failed} failures.")


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["sync", "batch-submit", "batch-check"])
    ap.add_argument("--sample", type=int, default=0,
                    help="Only first N prompts (for cheap smoke tests)")
    ap.add_argument("--batch-id", default=None,
                    help="Required for --mode batch-check (the batch name, "
                         "e.g. batches/abc123)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent API calls for sync mode (default 1).")
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
        run_batch_submit(client, df, output_dir, sample=args.sample)


if __name__ == "__main__":
    main()
