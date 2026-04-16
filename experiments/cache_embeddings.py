"""
Cache text embeddings using Qwen3-Embedding-8B.

Encodes neutral prompts (with instruction prefix) and KG triples (as documents)
for use in KG extraction via retrieval.

Usage:
    python cache_embeddings.py --gpu 0
    python cache_embeddings.py --gpu 0 --batch-size 64
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    DATA_CSV, EMBEDDING_MODEL_NAME, EMBEDDINGS_DIR, MODEL_CACHE,
)


def add_instruction(text, instruction):
    """Add Qwen3-Embedding instruction prefix."""
    return f"Instruct: {instruction}\nQuery: {text}"


def build_kg_repr(row):
    """KG representation: head | relation | stereo_tails -> anti_tails."""
    return (f"{row['head']} | {row['relation']} | "
            f"{row['stereotype_tails']} \u2192 {row['anti_stereotype_tails']}")


def main():
    parser = argparse.ArgumentParser(description="Cache text embeddings")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Load dataset (deduplicated to case level)
    df = pd.read_csv(DATA_CSV)
    n = len(df)
    print(f"Loaded {n} unique cases")

    # Build text representations
    neutral_prompts = df["prompt_neutral"].tolist()
    kg_texts = [build_kg_repr(row) for _, row in tqdm(df.iterrows(), total=n,
                                                       desc="Building KG texts")]

    # Instruction prefix for queries
    instruct = "Given a description, retrieve the matching knowledge graph triple"
    neutral_queries = [add_instruction(t, instruct) for t in neutral_prompts]

    print(f"\nExamples:")
    for i in [0, 1]:
        print(f"  Prompt: {neutral_prompts[i]}")
        print(f"  Query:  {neutral_queries[i][:100]}...")
        print(f"  KG:     {kg_texts[i]}")
        print()

    # Load model
    t0 = time.time()
    print(f"Loading {EMBEDDING_MODEL_NAME}... (this may take a minute)")
    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        cache_folder=MODEL_CACHE,
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Encode neutral prompts (as queries with instruction)
    t0 = time.time()
    print(f"\n[1/2] Encoding {n} neutral prompts (with instruction)...")
    emb_neutral = model.encode(
        neutral_queries,
        normalize_embeddings=True,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Encode KG triples (as documents, no instruction)
    t0 = time.time()
    print(f"\n[2/2] Encoding {n} KG triples...")
    emb_kg = model.encode(
        kg_texts,
        normalize_embeddings=True,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Save
    t0 = time.time()
    print("\nSaving embeddings...")
    np.save(os.path.join(EMBEDDINGS_DIR, "neutral_prompts.npy"), emb_neutral)
    np.save(os.path.join(EMBEDDINGS_DIR, "kg_triples.npy"), emb_kg)

    ids = df["id"].tolist()
    with open(os.path.join(EMBEDDINGS_DIR, "ids.json"), "w") as f:
        json.dump(ids, f)

    print(f"Saved to {EMBEDDINGS_DIR} in {time.time() - t0:.1f}s:")
    print(f"  neutral_prompts.npy  shape={emb_neutral.shape}")
    print(f"  kg_triples.npy       shape={emb_kg.shape}")
    print(f"  ids.json             n={len(ids)}")
    print("\nAll done!")


if __name__ == "__main__":
    main()
