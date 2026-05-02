"""
Use GPT API to generate bias-triggering prompts from extracted stereotype knowledge graphs.

Adapted for CrowS-Pairs extraction format where one ID can have multiple axes (units).
Each (id, axis) pair produces a separate trigger prompt.

Usage:
    python run_trigger.py                     # full run, 8 threads
    python run_trigger.py --sample 10         # test with 10 random groups
    python run_trigger.py --workers 16        # 16 concurrent threads

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

from trigger_prompt import PROMPT

# ===================== CONFIG =====================
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-5.4-mini-2026-03-17"
INPUT_CSV = os.path.join(os.path.dirname(__file__), "crowspairs_extraction_results.csv")
OUTPUT_JSONL = os.path.join(os.path.dirname(__file__), "crowspairs_trigger_results.jsonl")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "crowspairs_trigger_results.csv")
# ==================================================

# Thread-safe lock for writing to the output file
_write_lock = threading.Lock()


def load_processed_keys(output_path: str) -> set:
    """Load already-processed (id, axis) tuples from existing output file."""
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                processed.add((obj["id"], obj["axis"]))
    return processed


def build_groups(df: pd.DataFrame) -> list[dict]:
    """Group rows by (id, axis) and merge concepts per group."""
    groups = []
    grouped = df.groupby(["id", "axis"])
    for (gid, axis), group in grouped:
        stereo_rows = group[group["graph_type"] == "stereotype"]
        anti_rows = group[group["graph_type"] == "anti_stereotype"]
        if stereo_rows.empty and anti_rows.empty:
            continue

        first = group.iloc[0]

        stereo_concepts = [row["tail"] for _, row in stereo_rows.iterrows()]
        anti_concepts = [row["tail"] for _, row in anti_rows.iterrows()]

        groups.append({
            "id": int(gid),
            "target": first["target"],
            "bias_type": first["bias_type"],
            "stereo_antistereo": first.get("stereo_antistereo", ""),
            "axis": axis,
            "head": first.get("head", ""),
            "relation": first.get("relation", ""),
            "construction_mode": first.get("construction_mode", ""),
            "shared_frame": first.get("shared_frame", ""),
            "frame_sensitive": str(first.get("frame_sensitive", "")),
            "stereotype_sentence": first.get("stereotype_sentence", ""),
            "anti_stereotype_sentence": first.get("anti_stereotype_sentence", ""),
            "stereotype_tails": ", ".join(stereo_concepts) if stereo_concepts else "(none)",
            "anti_stereotype_tails": ", ".join(anti_concepts) if anti_concepts else "(none)",
        })
    return groups


def call_gpt(client: OpenAI, group: dict) -> dict | None:
    """Send prompt to GPT and parse the JSON response."""
    prompt_text = PROMPT.format(
        target=group["target"],
        bias_type=group["bias_type"],
        axis=group["axis"],
        head=group["head"],
        relation=group["relation"],
        construction_mode=group["construction_mode"],
        shared_frame=group["shared_frame"],
        frame_sensitive=group["frame_sensitive"],
        stereotype_tails=group["stereotype_tails"],
        anti_stereotype_tails=group["anti_stereotype_tails"],
        stereotype_sentence=group["stereotype_sentence"],
        anti_stereotype_sentence=group["anti_stereotype_sentence"],
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
        print(f"  [ERROR] API call failed for id={group['id']}/{group['axis']}: {e}")
        return None


def process_one(client: OpenAI, group: dict, fout, idx: int, total: int):
    """Process a single group: call API and write result (thread-safe)."""
    print(f"[{idx}/{total}] Processing id={group['id']} | {group['target']} | {group['axis']}...")
    result = call_gpt(client, group)

    record = {
        "id": group["id"],
        "target": group["target"],
        "bias_type": group["bias_type"],
        "stereo_antistereo": group["stereo_antistereo"],
        "axis": group["axis"],
        "head": group["head"],
        "relation": group["relation"],
        "construction_mode": group["construction_mode"],
        "shared_frame": group["shared_frame"],
        "frame_sensitive": group["frame_sensitive"],
        "stereotype_sentence": group["stereotype_sentence"],
        "anti_stereotype_sentence": group["anti_stereotype_sentence"],
        "stereotype_tails": group["stereotype_tails"],
        "anti_stereotype_tails": group["anti_stereotype_tails"],
        "result": result,
    }
    with _write_lock:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()


def jsonl_to_csv(jsonl_path: str, csv_path: str):
    """Convert output JSONL to a flat CSV."""
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row = {
                "id": obj["id"],
                "target": obj["target"],
                "bias_type": obj["bias_type"],
                "stereo_antistereo": obj.get("stereo_antistereo", ""),
                "axis": obj.get("axis", ""),
                "head": obj.get("head", ""),
                "relation": obj.get("relation", ""),
                "construction_mode": obj.get("construction_mode", ""),
                "shared_frame": obj.get("shared_frame", ""),
                "frame_sensitive": obj.get("frame_sensitive", ""),
                "stereotype_sentence": obj.get("stereotype_sentence", ""),
                "anti_stereotype_sentence": obj.get("anti_stereotype_sentence", ""),
                "stereotype_tails": obj.get("stereotype_tails", ""),
                "anti_stereotype_tails": obj.get("anti_stereotype_tails", ""),
            }
            result = obj.get("result")
            if result:
                row["neutral"] = result.get("neutral", "")
                row["stereotype_trigger"] = result.get("stereotype_trigger", "")
                row["anti_stereotype_trigger"] = result.get("anti_stereotype_trigger", "")
            else:
                row["neutral"] = ""
                row["stereotype_trigger"] = ""
                row["anti_stereotype_trigger"] = ""
            records.append(row)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate bias-triggering prompts from CrowS-Pairs KGs")
    parser.add_argument("--sample", type=int, default=0,
                        help="Randomly sample N groups for testing (0 = all)")
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
    groups = build_groups(df)
    print(f"Total groups (id x axis): {len(groups)}")

    # Resume support
    processed = load_processed_keys(OUTPUT_JSONL)
    remaining = [g for g in groups if (g["id"], g["axis"]) not in processed]
    print(f"Already processed: {len(processed)}, remaining: {len(remaining)}")

    # Optional sampling
    if args.sample > 0 and args.sample < len(remaining):
        random.seed(args.seed)
        remaining = random.sample(remaining, args.sample)
        print(f"Sampled {args.sample} groups for testing")

    total = len(remaining)
    print(f"Processing {total} groups with {args.workers} threads...\n")

    with open(OUTPUT_JSONL, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one, client, group, fout, i + 1, total): group
                for i, group in enumerate(remaining)
            }
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    group = futures[future]
                    print(f"  [ERROR] Unhandled exception for id={group['id']}: {exc}")

    print(f"\nDone! Results saved to {OUTPUT_JSONL}")

    # Convert to CSV
    jsonl_to_csv(OUTPUT_JSONL, OUTPUT_CSV)


if __name__ == "__main__":
    main()
