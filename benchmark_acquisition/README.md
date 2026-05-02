# Benchmark Prompt Acquisition

This folder holds the two-stage LLM pipeline that turns the **StereoSet** and
**CrowS-Pairs** datasets into image-generation prompt triplets
(`neutral`, `stereotype_trigger`, `anti_stereotype_trigger`) for the rest of
this repo. Full methodology lives in
[`reports/benchmark_prompt_acquisition.md`](../reports/benchmark_prompt_acquisition.md).

## Layout

```
benchmark_acquisition/
├── stereoset/
│   ├── extraction_prompt.py     # PROMPT — KG extraction template
│   ├── trigger_prompt.py        # PROMPT — trigger-prompt template
│   ├── run_extraction.py        # stage 1: sentence pair → KG (gpt-5.4)
│   └── run_trigger.py           # stage 2: KG → 3 image prompts (gpt-5.4-mini)
└── crows_pairs/
    ├── extraction_prompt.py
    ├── trigger_prompt.py
    ├── run_extraction.py
    └── run_trigger.py
```

Each prompt file exposes a single `PROMPT` string. Only the version actually
wired into the published benchmark is checked in here — historical versions
(`PROMPT_V2`, `prompt_v4`, `prompt_v6`, …) live in the upstream
`/data/gpfs/projects/punim2888/{stereoset,crows-pairs}/` repos for reference.

## Environment

Both run scripts read `OPENAI_API_KEY` from the environment (or `.env`):

```bash
export OPENAI_API_KEY=sk-...
# or put OPENAI_API_KEY=sk-... in stereoimage/.env (already git-ignored)
```

## Input data

The source CSVs are not stored in this repo. Drop or symlink them into:

| Pipeline | Expected path |
|---|---|
| StereoSet | `benchmark_acquisition/stereoset/Stereoset - stereotypes.csv` |
| CrowS-Pairs | `benchmark_acquisition/crows_pairs/data/crows_pairs_anonymized.csv` |

Originals: `/data/gpfs/projects/punim2888/stereoset/Stereoset - stereotypes.csv`
and `/data/gpfs/projects/punim2888/crows-pairs/data/crows_pairs_anonymized.csv`.

## Running

From inside each dataset folder (paths are computed relative to the script):

```bash
cd benchmark_acquisition/stereoset
python run_extraction.py --sample 10 --workers 1   # smoke test
python run_extraction.py                            # full run, 8 threads, resumable

# Convert the extraction JSONL to CSV (the trigger script reads CSV).
# A small ad-hoc converter exists in the upstream stereoset/ repo; the
# trigger script then picks up stereotype_extraction_results.csv.

python run_trigger.py
```

Both run scripts support `--sample N`, `--workers N`, `--seed N`, and resume
from the existing `*_results.jsonl` on re-run.

## Outputs

- `*_extraction_results.jsonl` — one row per (split, id) [StereoSet] or per id [CrowS-Pairs]
- `*_trigger_results.jsonl` and `*_trigger_results.csv` — one row per (id, axis); the CSV adds flat `neutral` / `stereotype_trigger` / `anti_stereotype_trigger` columns
