"""
Evaluate whether a generated image correctly depicts its neutral prompt.

For each image, makes ONE VLM call (OpenRouter / Qwen3-VL) and asks the model
to return a binary aligned=true/false verdict with a short justification.

The comparison is always against the *neutral* prompt from
`data/merged_all_aggregated.csv`, even for experiments that rewrite the prompt
(exp 1-3). This tests whether debiased images still depict the original concept.

Usage:
    python evaluate_alignment.py --exp-id 0
    python evaluate_alignment.py --all
    python evaluate_alignment.py --exp-id 9 --sample 10 --workers 16
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    DATA_CSV, EVAL_RESULTS_DIR, EXPERIMENT_NAMES,
    exp_output_dir,
)

# Load API key: prefer environment variable, fall back to .env file
from pathlib import Path as _Path
_env_path = _Path("/data/gpfs/projects/punim2888/stereoset/backup/stereoset-augment/.env")
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "qwen/qwen3-vl-30b-a3b-instruct"

ALIGN_PROMPT = """You are checking whether a generated image correctly depicts a text prompt.

## Prompt
"{prompt_neutral}"

## Task
Decide whether the image correctly aligns with the prompt:
- The main subject(s) of the prompt appear in the image.
- The action, setting, and attributes described are visibly depicted.
- Ignore artistic style, photorealism, and minor visual flaws.

Respond with a JSON object in this exact format:
{{"aligned": <true|false>, "justification": "<1-2 sentence explanation>"}}"""


def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_aligned(output_text):
    """Extract (aligned: bool|None, justification: str) from raw model text."""
    json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            aligned = obj.get("aligned")
            if isinstance(aligned, bool):
                return aligned, obj.get("justification", "")
            if isinstance(aligned, str):
                v = aligned.strip().lower()
                if v in ("true", "yes"):
                    return True, obj.get("justification", "")
                if v in ("false", "no"):
                    return False, obj.get("justification", "")
        except json.JSONDecodeError:
            pass
    # Fallback: regex on the raw text
    m = re.search(r'"aligned"\s*:\s*(true|false)', output_text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true", "(fallback parse)"
    return None, output_text[:300]


def judge_alignment_api(image_path, prompt_neutral, max_retries=3):
    """Ask the VLM if the image aligns with the neutral prompt. Returns dict or None."""
    prompt_text = ALIGN_PROMPT.format(prompt_neutral=prompt_neutral)

    img_b64 = encode_image_base64(image_path)
    ext = os.path.splitext(image_path)[1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg",
            "jpeg": "image/jpeg"}.get(ext.lstrip("."), "image/png")

    messages = [{"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        {"type": "text", "text": prompt_text},
    ]}]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers,
                                 json=payload, timeout=120)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                print(f"  [DEBUG] HTTP {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()

            data = resp.json()
            output_text = data["choices"][0]["message"]["content"]

            aligned, justification = _parse_aligned(output_text)
            if aligned is None:
                print(f"  [WARN] Unparseable output: {output_text[:200]}")
                return None
            return {"aligned": aligned, "justification": justification}

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            print(f"  [ERROR] API call failed after {max_retries} retries: {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"  [ERROR] Parse error: {e}")
            return None

    return None


def process_row(row_data, case_lookup, exp_id):
    _, row = row_data
    case_id = row["case_id"]
    alpha = row.get("alpha")
    seed = row["seed"]
    img_path = row["image_path"]

    if not os.path.exists(img_path):
        return None

    case_info = case_lookup.get(case_id, {})
    prompt_neutral = case_info.get("prompt_neutral", "")
    if not prompt_neutral or pd.isna(prompt_neutral):
        print(f"  [WARN] no prompt_neutral for {case_id}, skipping")
        return None

    result = judge_alignment_api(img_path, prompt_neutral)

    return {
        "case_id": case_id,
        "experiment_id": exp_id,
        "target": case_info.get("target", ""),
        "bias_type": case_info.get("bias_type", ""),
        "alpha": alpha if pd.notna(alpha) else None,
        "seed": int(seed),
        "image_path": img_path,
        "prompt_neutral": prompt_neutral,
        "aligned": result["aligned"] if result else None,
        "justification": (result.get("justification", "")
                          if result else ""),
    }


def evaluate_experiment(exp_id, case_lookup, sample=0, manifest_path=None,
                        output_jsonl=None, workers=8):
    output_dir = exp_output_dir(exp_id)
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    manifest_path = manifest_path or os.path.join(output_dir, "manifest.csv")
    if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) <= 1:
        print(f"Exp {exp_id}: manifest not found or empty, skipping")
        return 0

    try:
        manifest = pd.read_csv(manifest_path)
    except pd.errors.EmptyDataError:
        print(f"Exp {exp_id}: manifest has no data, skipping")
        return 0
    if len(manifest) == 0:
        print(f"Exp {exp_id}: manifest is empty, skipping")
        return 0

    print(f"\n{'='*60}")
    print(f"Exp {exp_id}: {EXPERIMENT_NAMES[exp_id]} (alignment)")
    print(f"Loaded manifest: {len(manifest)} entries")

    if sample > 0:
        manifest = manifest.head(sample)

    results_jsonl = output_jsonl or os.path.join(
        EVAL_RESULTS_DIR, f"exp_{exp_id:02d}_alignment.jsonl")

    processed_keys = set()
    if os.path.exists(results_jsonl):
        with open(results_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("aligned") is None:
                        continue
                    alpha = r.get("alpha")
                    key = (r["case_id"],
                           str(alpha) if alpha is not None else "",
                           str(r["seed"]))
                    processed_keys.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming: {len(processed_keys)} entries already processed")

    rows_to_process = []
    for idx, row in manifest.iterrows():
        case_id = row["case_id"]
        alpha = row.get("alpha")
        seed = row["seed"]
        key = (case_id, str(alpha if pd.notna(alpha) else ""), str(seed))
        if key not in processed_keys and os.path.exists(row["image_path"]):
            rows_to_process.append((idx, row))

    print(f"To process: {len(rows_to_process)} images "
          f"(skipped {len(manifest) - len(rows_to_process)})")

    if not rows_to_process:
        print("Nothing to do.")
        return 0

    fout = open(results_jsonl, "a")
    pbar = tqdm(total=len(rows_to_process), desc=f"Align exp {exp_id}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_row, row_data, case_lookup, exp_id): row_data
            for row_data in rows_to_process
        }

        for future in as_completed(futures):
            record = future.result()
            if record is not None:
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                pbar.set_postfix(aligned=record["aligned"])
            pbar.update(1)

    pbar.close()
    fout.close()

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
        print(f"Alignment eval complete! {len(results)} entries -> {csv_path}")
    else:
        print("No results generated.")

    return len(rows_to_process)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate image/neutral-prompt alignment")
    parser.add_argument("--exp-id", type=int, nargs="+", default=None,
                        choices=list(EXPERIMENT_NAMES.keys()),
                        help="Experiment ID(s) to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all experiments")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--manifest", type=str, default=None,
                        help="Custom manifest CSV path (only for single exp)")
    parser.add_argument("--output-jsonl", type=str, default=None,
                        help="Custom output JSONL path (only for single exp)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel API workers")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    if args.all:
        exp_ids = sorted(EXPERIMENT_NAMES.keys())
    elif args.exp_id:
        exp_ids = args.exp_id
    else:
        parser.error("Provide --exp-id or --all")

    cases_df = pd.read_csv(DATA_CSV)
    case_lookup = cases_df.set_index("id").to_dict("index")

    total_processed = 0
    for exp_id in exp_ids:
        n = evaluate_experiment(
            exp_id, case_lookup,
            sample=args.sample,
            manifest_path=args.manifest if len(exp_ids) == 1 else None,
            output_jsonl=args.output_jsonl if len(exp_ids) == 1 else None,
            workers=args.workers,
        )
        total_processed += n

    print(f"\n{'='*60}")
    print(f"All done! Processed {total_processed} images across "
          f"{len(exp_ids)} experiments.")


if __name__ == "__main__":
    main()
