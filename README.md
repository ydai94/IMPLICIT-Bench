# StereoImage: Bias Evaluation for Text-to-Image Models

A benchmark and analysis pipeline for measuring stereotype bias in
text-to-image (T2I) models. Each prompt unit in the dataset has three
arms — *neutral*, *stereotype-trigger*, *anti-stereotype-trigger* —
allowing both bias measurement (does the model default to the
stereotype?) and bias controllability (does explicit prompting move
the model?).

## Dataset

| Property | Value |
|---|---|
| Prompt units | 1,831 (filtered to `lean_stereotype` — prompts whose neutral output already leans stereotypical) |
| Sources | StereoSet (1,393) + CrowS-Pairs (438) |
| Bias categories | 11: `gender`, `profession`, `race`, `religion`, `socioeconomic`, `race-color`, `age`, `nationality`, `sexual-orientation`, `disability`, `physical-appearance` |
| Arms per unit | 3 (neutral, stereotype-trigger, anti-stereotype-trigger) |
| Seeds per arm | 3 |
| Total image evaluations per generator | 5,493 (1,831 × 3 seeds) |

Canonical files in `data/`:
- `merged_all.csv` — per-(id, seed) VLM scores from Qwen3-VL and Gemma-4 across all three arms (this is the analysis-ready file)
- `merged_all_aggregated.csv` — one row per prompt unit (used by experiment scripts)
- `merged_stereoset.csv` / `merged_crowspairs.csv` — per-source raw data
- `lean_stereotype_union.csv` — the filtered subset

## Experiments

15 experiments registered in `experiments/config.py`:

| ID | Name | Type |
|:-:|---|---|
| 0 | baseline (Qwen-Image) | baseline generation |
| 1 | llm_rewrite_no_kg | prompt rewriting |
| 2–3 | extracted_kg_llm_rewrite, gt_kg_llm_rewrite | KG-grounded prompt rewriting |
| 4–5 | extracted_kg_full_triple_sv, extracted_kg_tail_sv | steering vectors (extracted KG) |
| 6–7 | gt_kg_full_triple_sv, gt_kg_tail_sv | steering vectors (GT KG) |
| 8–11 | *_llm_pair_sv, *_gt_pair_sv | steering vectors (LLM- vs GT-generated pairs) |
| 12 | gpt_image_2_baseline | alternative T2I model |
| 13 | sd3_baseline | Stable Diffusion 3 |
| 14 | nano_banana_2_baseline | Nano Banana 2 |

Per-image stereotype scores (0–5 rubric) are written to
`cache/eval_results/exp_NN_eval.csv` by `experiments/evaluate_*.py`.

## Repository Layout

```
data/                  Benchmark prompts, VLM scores, KG annotations, human-eval bundle
experiments/           Generation + evaluation pipeline (config, evaluate_*, cache_*)
scripts/               Standalone analyses (agreement, CLIP, category difficulty, form gen)
cache/                 LLM/VLM outputs, embeddings, eval CSVs (embeddings/checkpoints gitignored)
reports/               Aggregate CSVs and analysis markdown
plots/                 Generated figures (gitignored)
experiment_outputs/    Generated images per experiment (gitignored, large)
```

## Key Reports

| File | What it covers |
|---|---|
| `qwen_image_bias_analysis.md` | Per-bias-type means, S − N, S − A on the Qwen-Image baseline (Qwen3-VL + Gemma-4 + CLIP evaluators) |
| `baseline_comparison.md` | Cross-model comparison across Qwen-Image / SD3 / GPT-Image-2 |
| `qwen_image_pre_post_filter_bias.md` | Effect of the lean_stereotype filter |
| `nano_banana_failure_report.md` | Failure-case write-up for the Nano Banana 2 baseline |
| `reports/category_difficulty_analysis.md` | Per-category difficulty score (Hedges' g, S vs A) with tier interpretation |
| `reports/agreement_report.md` | LLM-judge inter-annotator agreement (Fleiss' κ, Cohen's κ) on prompt labeling |
| `reports/all_experiments_comparison.md` | Aggregate cross-experiment results |

## Setup

The pipeline is built around an HPC environment using conda:

```bash
module load Anaconda3/2024.02-1
conda activate videoGen
```

Dependencies are mostly standard: `numpy`, `pandas`, `matplotlib`, `torch`,
`transformers`, `diffusers`, `scikit-learn`, plus per-script extras (`google-api-python-client`
for the form generators, `openai` / `anthropic` SDKs for LLM judging).

GPU is required for image generation and VLM evaluation; analysis scripts
in `scripts/` run on CPU.

## Reproducing Analyses

**Per-category difficulty score** (Hedges' g per bias type with cluster-bootstrap CI):
```bash
python scripts/compute_category_difficulty.py
# → reports/category_difficulty.csv
# → plots/category_difficulty_ranking.png
```

**LLM-judge agreement** on prompt labeling:
```bash
python scripts/compute_agreement.py
# → reports/agreement_report.md
# → reports/agreement_by_bias_type.csv
```

**CLIP comparison** (image-embedding similarities + per-bias-type box plots):
```bash
python scripts/run_clip_comparison.py
```

**Run / re-evaluate a generation experiment** (Qwen-Image baseline):
```bash
python experiments/evaluate_all.py --exp_id 0
python experiments/evaluate_bias_local.py --exp_id 0    # 0–5 stereotype rubric
python experiments/evaluate_alignment.py --exp_id 0     # neutral-prompt alignment check
```

## Human Evaluation Pipeline

A 50-case stratified sample is rated by humans via Google Forms; see
[`data/human_eval/README.md`](data/human_eval/README.md). Pipeline scripts:

- `scripts/sample_human_eval.py` — stratified case sampler
- `scripts/build_forms_package.py` — packages images and questions
- `scripts/build_google_form.py` / `scripts/create_google_form_api.py` — form materializers

The image bundle (`data/human_eval/images/`, `images.zip`) is gitignored.
Regenerate locally from the manifest if you need to host the form again.

## Data Provenance

- **StereoSet**: Nadeem et al., 2021. *StereoSet: Measuring stereotypical bias in pretrained language models.*
- **CrowS-Pairs**: Nangia et al., 2020. *CrowS-Pairs: A challenge dataset for measuring social biases in masked language models.*
- Both adapted to T2I prompts via the three-arm template defined in this repo.
