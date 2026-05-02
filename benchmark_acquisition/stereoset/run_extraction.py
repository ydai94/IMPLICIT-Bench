"""
Use GPT API to extract stereotype knowledge graphs from StereoSet data.

Usage:
    python run_extraction.py                     # full run, 8 threads
    python run_extraction.py --sample 10         # test with 10 random pairs
    python run_extraction.py --workers 16        # 16 concurrent threads

Reads OPENAI_API_KEY from the environment (or .env).
"""

import argparse
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

from extraction_prompt import PROMPT

# ===================== CONFIG =====================
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-5.4-2026-03-05"
INPUT_CSV = os.path.join(os.path.dirname(__file__), "Stereoset - stereotypes.csv")
OUTPUT_JSONL = os.path.join(os.path.dirname(__file__), "stereotype_extraction_results.jsonl")
# ==================================================

# Thread-safe lock for writing to the output file
_write_lock = threading.Lock()


def load_processed_ids(output_path: str) -> set:
    """Load already-processed (split, id) pairs from existing output file."""
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                processed.add((obj["split"], obj["id"]))
    return processed


def build_pairs(df: pd.DataFrame) -> list[dict]:
    """Group rows by (split, id) and pair stereotype + anti-stereotype sentences."""
    pairs = []
    grouped = df.groupby(["split", "id"])
    for (split, gid), group in grouped:
        stereo_rows = group[group["gold_label"] == "stereotype"]
        anti_rows = group[group["gold_label"] == "anti-stereotype"]
        if stereo_rows.empty or anti_rows.empty:
            continue

        # Use the first stereotype and anti-stereotype sentence for each group
        stereo_row = stereo_rows.iloc[0]
        anti_row = anti_rows.iloc[0]

        pairs.append({
            "split": split,
            "id": gid,
            "target": stereo_row["target"],
            "bias_type": stereo_row["bias_type"],
            "context": stereo_row["context"],
            "stereotype": stereo_row["sentence"],
            "anti_stereotype": anti_row["sentence"],
        })
    return pairs


def call_gpt(client: OpenAI, pair: dict) -> dict | None:
    """Send prompt to GPT and parse the JSON response."""
    prompt_text = PROMPT.format(
        target=pair["target"],
        bias_category=pair["bias_type"],
        context=pair["context"],
        stereotype=pair["stereotype"],
        anti_stereotype=pair["anti_stereotype"],
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"  [ERROR] API call failed for {pair['split']}/{pair['id']}: {e}")
        return None


def process_one(client: OpenAI, pair: dict, fout, idx: int, total: int):
    """Process a single pair: call API and write result (thread-safe)."""
    print(f"[{idx}/{total}] Processing {pair['split']} | {pair['target']} | {pair['id'][:12]}...")
    result = call_gpt(client, pair)

    record = {
        "split": pair["split"],
        "id": pair["id"],
        "target": pair["target"],
        "bias_type": pair["bias_type"],
        "context": pair["context"],
        "stereotype": pair["stereotype"],
        "anti_stereotype": pair["anti_stereotype"],
        "extraction": result,
    }
    with _write_lock:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()


def main():
    parser = argparse.ArgumentParser(description="Extract stereotype knowledge graphs")
    parser.add_argument("--sample", type=int, default=0,
                        help="Randomly sample N pairs for testing (0 = all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent threads (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    if not API_KEY:
        raise ValueError("OPENAI_API_KEY not set (export it or add to .env)")

    client = OpenAI(api_key=API_KEY)

    # Load data
    df = pd.read_csv(INPUT_CSV)
    pairs = build_pairs(df)
    print(f"Total pairs: {len(pairs)}")

    # Resume support
    processed = load_processed_ids(OUTPUT_JSONL)
    remaining = [p for p in pairs if (p["split"], p["id"]) not in processed]
    print(f"Already processed: {len(processed)}, remaining: {len(remaining)}")

    # Optional sampling
    if args.sample > 0 and args.sample < len(remaining):
        random.seed(args.seed)
        remaining = random.sample(remaining, args.sample)
        print(f"Sampled {args.sample} pairs for testing")

    total = len(remaining)
    print(f"Processing {total} pairs with {args.workers} threads...\n")

    with open(OUTPUT_JSONL, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one, client, pair, fout, i + 1, total): pair
                for i, pair in enumerate(remaining)
            }
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    pair = futures[future]
                    print(f"  [ERROR] Unhandled exception for {pair['id'][:12]}: {exc}")

    print(f"\nDone! Results saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
