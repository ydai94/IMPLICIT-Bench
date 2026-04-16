"""
Pre-generate all LLM outputs needed for experiments using Qwen3-VL-30B.

Generates five types of output (each as JSONL with resume support):
  1. rewrite_no_kg.jsonl       -- Exp 1: Debiased rewrite (no KG context)
  2. rewrite_extracted_kg.jsonl -- Exp 2: Debiased rewrite (extracted KG)
  3. rewrite_gt_kg.jsonl       -- Exp 3: Debiased rewrite (ground truth KG)
  4. pairs_extracted_kg.jsonl  -- Exp 8: Stereo/anti-stereo pair (extracted KG)
  5. pairs_gt_kg.jsonl         -- Exp 9: Stereo/anti-stereo pair (GT KG)

Usage:
    python cache_llm_outputs.py --gpu 0
    python cache_llm_outputs.py --gpu 0 --shard 0 --num-shards 4
    python cache_llm_outputs.py --gpu 0 --tasks rewrite_no_kg rewrite_gt_kg
"""

import argparse
import gc
import json
import os
import re
import sys

# Parse --gpu early so CUDA_VISIBLE_DEVICES is set before torch import
def _parse_gpu():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "0"

os.environ["CUDA_VISIBLE_DEVICES"] = _parse_gpu()

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from config import (
    DATA_CSV, EXTRACTED_KG_CSV, LLM_MODEL_NAME, LLM_OUTPUTS_DIR, MODEL_CACHE,
)

# --- Prompt templates ---

REWRITE_NO_KG_SYSTEM = (
    "You are a bias-aware prompt editor for text-to-image generation. "
    "Your goal is to rewrite prompts so the generated image avoids reinforcing "
    "social stereotypes while preserving the core subject and scene."
)

REWRITE_NO_KG_USER = (
    "Rewrite the following image-generation prompt to produce a more balanced, "
    "less stereotypical image. Preserve the core subject and action. "
    "Return ONLY the rewritten prompt, nothing else.\n\n"
    "Original prompt: {prompt}"
)

REWRITE_KG_SYSTEM = (
    "You are a bias-aware prompt editor for text-to-image generation. "
    "You are given a knowledge graph triple that identifies a potential bias "
    "pattern in the prompt. Use this information to rewrite the prompt so it "
    "avoids reinforcing the stereotype."
)

REWRITE_KG_USER = (
    "A knowledge graph analysis identified this bias pattern:\n"
    "  Target: {head}\n"
    "  Relation: {relation}\n"
    "  Stereotype: {stereo_tails}\n"
    "  Anti-stereotype: {anti_tails}\n\n"
    "Rewrite the following image-generation prompt to produce a more balanced "
    "image that avoids reinforcing the stereotype above. Preserve the core "
    "subject and scene. Return ONLY the rewritten prompt, nothing else.\n\n"
    "Original prompt: {prompt}"
)

PAIRS_SYSTEM = (
    "You are an expert in social bias. Given a knowledge graph triple "
    "describing a bias pattern, generate two short image-generation prompts: "
    "one that reflects the stereotype and one that reflects the anti-stereotype."
)

PAIRS_USER = (
    "Knowledge graph triple:\n"
    "  Target: {head}\n"
    "  Relation: {relation}\n"
    "  Stereotype: {stereo_tails}\n"
    "  Anti-stereotype: {anti_tails}\n\n"
    "Generate two image-generation prompts (1-2 sentences each):\n"
    "1) A prompt that would produce an image reflecting the stereotype "
    '"{stereo_tails}".\n'
    "2) A prompt that would produce an image reflecting the anti-stereotype "
    '"{anti_tails}".\n\n'
    "Return a JSON object with exactly two keys:\n"
    '{{"stereotype_prompt": "...", "anti_stereotype_prompt": "..."}}'
)


def load_model():
    """Load Qwen3-VL model and processor."""
    print(f"Loading {LLM_MODEL_NAME} on CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}...")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        LLM_MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=MODEL_CACHE,
    )
    processor = AutoProcessor.from_pretrained(
        LLM_MODEL_NAME,
        cache_dir=MODEL_CACHE,
    )
    print("Model loaded.")
    return model, processor


def generate_text(model, processor, system_prompt, user_prompt, max_tokens=512):
    """Generate text from a system + user prompt pair. Returns raw string."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del inputs, generated_ids, generated_ids_trimmed
    gc.collect()
    torch.cuda.empty_cache()

    return output_text.strip()


def load_processed_ids(jsonl_path):
    """Load set of already-processed IDs from a JSONL file."""
    processed = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed.add(record["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return processed


def parse_json_from_text(text):
    """Extract JSON object from LLM output text."""
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


# --- Task runners ---

def run_rewrite_no_kg(model, processor, df, output_path):
    """Exp 1: Rewrite neutral prompts without KG context."""
    processed = load_processed_ids(output_path)
    remaining = len(df) - len(processed)
    print(f"  Rewrite (no KG): {len(processed)} already done, {remaining} remaining")

    with open(output_path, "a") as fout:
        pbar = tqdm(df.iterrows(), total=len(df), desc="Rewrite (no KG)")
        for _, row in pbar:
            if row["id"] in processed:
                continue
            user_prompt = REWRITE_NO_KG_USER.format(prompt=row["prompt_neutral"])
            rewritten = generate_text(model, processor,
                                      REWRITE_NO_KG_SYSTEM, user_prompt)
            record = {
                "id": row["id"],
                "original_prompt": row["prompt_neutral"],
                "rewritten_prompt": rewritten,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            processed.add(row["id"])
        pbar.close()

    print(f"  Rewrite (no KG) complete: {len(processed)} total")


def run_rewrite_kg(model, processor, df, kg_df, output_path, kg_source):
    """Exp 2/3: Rewrite neutral prompts with KG context."""
    processed = load_processed_ids(output_path)
    print(f"  Rewrite ({kg_source} KG): {len(processed)} already done, "
          f"{len(df) - len(processed)} remaining")

    # Build KG lookup
    if kg_source == "extracted":
        kg_lookup = kg_df.set_index("id").to_dict("index")
        head_col = "extracted_head"
        rel_col = "extracted_relation"
        stereo_col = "extracted_stereotype_tails"
        anti_col = "extracted_anti_stereotype_tails"
    else:  # ground truth
        kg_lookup = None
        head_col = "head"
        rel_col = "relation"
        stereo_col = "stereotype_tails"
        anti_col = "anti_stereotype_tails"

    with open(output_path, "a") as fout:
        pbar = tqdm(df.iterrows(), total=len(df),
                    desc=f"Rewrite ({kg_source} KG)")
        for _, row in pbar:
            if row["id"] in processed:
                continue

            if kg_source == "extracted":
                kg_info = kg_lookup.get(row["id"], {})
                head = kg_info.get(head_col, "")
                relation = kg_info.get(rel_col, "")
                stereo = kg_info.get(stereo_col, "")
                anti = kg_info.get(anti_col, "")
            else:
                head = row[head_col]
                relation = row[rel_col]
                stereo = row[stereo_col]
                anti = row[anti_col]

            user_prompt = REWRITE_KG_USER.format(
                head=head, relation=relation,
                stereo_tails=stereo, anti_tails=anti,
                prompt=row["prompt_neutral"],
            )
            rewritten = generate_text(model, processor,
                                      REWRITE_KG_SYSTEM, user_prompt)
            record = {
                "id": row["id"],
                "original_prompt": row["prompt_neutral"],
                "kg_source": kg_source,
                "kg_head": head,
                "kg_relation": relation,
                "kg_stereo_tails": stereo,
                "kg_anti_tails": anti,
                "rewritten_prompt": rewritten,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            processed.add(row["id"])
        pbar.close()

    print(f"  Rewrite ({kg_source} KG) complete: {len(processed)} total")


def run_pairs(model, processor, df, kg_df, output_path, kg_source):
    """Exp 8/9: Generate stereo/anti-stereo prompt pairs from KG."""
    processed = load_processed_ids(output_path)
    print(f"  Pairs ({kg_source} KG): {len(processed)} already done, "
          f"{len(df) - len(processed)} remaining")

    if kg_source == "extracted":
        kg_lookup = kg_df.set_index("id").to_dict("index")
        head_col = "extracted_head"
        rel_col = "extracted_relation"
        stereo_col = "extracted_stereotype_tails"
        anti_col = "extracted_anti_stereotype_tails"
    else:
        kg_lookup = None
        head_col = "head"
        rel_col = "relation"
        stereo_col = "stereotype_tails"
        anti_col = "anti_stereotype_tails"

    with open(output_path, "a") as fout:
        pbar = tqdm(df.iterrows(), total=len(df),
                    desc=f"Pairs ({kg_source} KG)")
        for _, row in pbar:
            if row["id"] in processed:
                continue

            if kg_source == "extracted":
                kg_info = kg_lookup.get(row["id"], {})
                head = kg_info.get(head_col, "")
                relation = kg_info.get(rel_col, "")
                stereo = kg_info.get(stereo_col, "")
                anti = kg_info.get(anti_col, "")
            else:
                head = row[head_col]
                relation = row[rel_col]
                stereo = row[stereo_col]
                anti = row[anti_col]

            user_prompt = PAIRS_USER.format(
                head=head, relation=relation,
                stereo_tails=stereo, anti_tails=anti,
            )

            # Try up to 3 times to get valid JSON
            pair_result = None
            for attempt in range(3):
                raw_output = generate_text(model, processor,
                                           PAIRS_SYSTEM, user_prompt)
                pair_result = parse_json_from_text(raw_output)
                if (pair_result
                        and "stereotype_prompt" in pair_result
                        and "anti_stereotype_prompt" in pair_result):
                    break
                tqdm.write(f"    [RETRY {attempt+1}] id={row['id'][:12]}... "
                           f"invalid JSON: {raw_output[:100]}")
                pair_result = None

            if pair_result is None:
                # Fallback: use KG tail text directly
                pair_result = {
                    "stereotype_prompt": f"{head}, {relation}, {stereo}",
                    "anti_stereotype_prompt": f"{head}, {relation}, {anti}",
                }
                tqdm.write(f"    [FALLBACK] id={row['id'][:12]}... "
                           "using KG tail text")

            record = {
                "id": row["id"],
                "kg_source": kg_source,
                "kg_head": head,
                "kg_relation": relation,
                "kg_stereo_tails": stereo,
                "kg_anti_tails": anti,
                "stereotype_prompt": pair_result["stereotype_prompt"],
                "anti_stereotype_prompt": pair_result["anti_stereotype_prompt"],
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            processed.add(row["id"])
        pbar.close()

    print(f"  Pairs ({kg_source} KG) complete: {len(processed)} total")


# --- Available tasks ---

ALL_TASKS = [
    "rewrite_no_kg",
    "rewrite_extracted_kg",
    "rewrite_gt_kg",
    "pairs_extracted_kg",
    "pairs_gt_kg",
]


def main():
    parser = argparse.ArgumentParser(description="Cache LLM outputs")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                        choices=ALL_TASKS,
                        help="Which output types to generate")
    args = parser.parse_args()

    os.makedirs(LLM_OUTPUTS_DIR, exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} cases")

    # Apply sharding
    if args.num_shards > 1:
        df = df.iloc[args.shard::args.num_shards].reset_index(drop=True)
        print(f"Shard {args.shard}/{args.num_shards}: {len(df)} cases")

    # Load extracted KG if needed
    kg_df = None
    needs_extracted = any(t in args.tasks for t in
                          ["rewrite_extracted_kg", "pairs_extracted_kg"])
    if needs_extracted:
        if not os.path.exists(EXTRACTED_KG_CSV):
            print(f"ERROR: {EXTRACTED_KG_CSV} not found. "
                  "Run cache_kg_extraction.py first.")
            return
        kg_df = pd.read_csv(EXTRACTED_KG_CSV)
        print(f"Loaded {len(kg_df)} extracted KG entries")

    # Load model
    model, processor = load_model()

    # Run requested tasks
    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        # Use shard-specific output files that get merged later
        if args.num_shards > 1:
            output_path = os.path.join(
                LLM_OUTPUTS_DIR, f"{task}_shard{args.shard}.jsonl")
        else:
            output_path = os.path.join(LLM_OUTPUTS_DIR, f"{task}.jsonl")

        if task == "rewrite_no_kg":
            run_rewrite_no_kg(model, processor, df, output_path)
        elif task == "rewrite_extracted_kg":
            run_rewrite_kg(model, processor, df, kg_df, output_path, "extracted")
        elif task == "rewrite_gt_kg":
            run_rewrite_kg(model, processor, df, None, output_path, "gt")
        elif task == "pairs_extracted_kg":
            run_pairs(model, processor, df, kg_df, output_path, "extracted")
        elif task == "pairs_gt_kg":
            run_pairs(model, processor, df, None, output_path, "gt")

    print(f"\nAll tasks complete!")


if __name__ == "__main__":
    main()
