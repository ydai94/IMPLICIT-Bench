# Stereotype Bias Evaluation Summary

**Date**: 2026-04-17  
**Dataset**: 1,831 cases × 3 seeds = 5,493 images per experiment  
**Evaluation**: Qwen3-VL (via OpenRouter) rates each image's stereotype strength on 0–5 scale  
**Steering direction**: All steering experiments push towards anti-stereotype (d = encode(anti) − encode(stereo))

## Data Completeness

| Exp | Name | Images Evaluated | Cases | Complete? |
|-----|------|-----------------|-------|-----------|
| 0 | Baseline | 5,493 | 1,831 | Yes |
| 1 | LLM rewrite (no KG) | 2,757 | 919 | ~50% |
| 2 | Extracted KG + LLM rewrite | 2,070 | 690 | ~38% |
| 3 | GT KG + LLM rewrite | 2,757 | 919 | ~50% |
| 4 | Extracted KG full triple SV | 2,070 | 690 | ~38% |
| 5 | Extracted KG tail SV | 2,070 | 690 | ~38% |
| 6 | GT KG full triple SV | 2,070 | 690 | ~38% |
| 7 | GT KG tail SV | 2,070 | 690 | ~38% |
| 8 | Extracted KG LLM pair SV | 2,070 | 690 | ~38% |
| 9 | GT KG LLM pair SV | 2,070 | 690 | ~38% |
| 10 | Extracted KG GT pair SV | 2,070 | 690 | ~38% |
| 11 | GT KG GT pair SV | 1,380 | 460 | ~25% |

> All analyses below use the **460 common cases** present in every experiment for fair comparison.

## Overall Results

| Exp | Method | Mean Score | Reduction from Baseline | % Reduction |
|-----|--------|-----------|------------------------|-------------|
| 0 | **Baseline** | **3.409** | — | — |
| 1 | LLM rewrite (no KG) | 3.011 | −0.399 | 11.7% |
| 2 | Extracted KG + LLM rewrite | 2.851 | −0.558 | 16.4% |
| 3 | GT KG + LLM rewrite | 2.637 | −0.772 | 22.7% |
| 4 | Extracted KG full triple SV | 3.184 | −0.225 | 6.6% |
| 5 | Extracted KG tail SV | 2.888 | −0.521 | 15.3% |
| 6 | GT KG full triple SV | 2.875 | −0.534 | 15.7% |
| 7 | GT KG tail SV | 2.225 | −1.184 | 34.7% |
| 8 | Extracted KG LLM pair SV | 1.730 | −1.680 | 49.3% |
| 9 | **GT KG LLM pair SV** | **1.102** | **−2.307** | **67.7%** |
| 10 | Extracted KG GT pair SV | 2.841 | −0.568 | 16.7% |
| 11 | GT KG GT pair SV | 2.043 | −1.366 | 40.1% |

## Paired Case-Level Analysis

For each case, the mean score across 3 seeds is compared to baseline. "Improved" = score dropped by >0.5, "Worsened" = score increased by >0.5.

| Exp | Method | Improved | Same | Worsened | Mean Delta |
|-----|--------|----------|------|----------|------------|
| 1 | LLM rewrite (no KG) | 31.7% | 48.9% | 19.3% | −0.399 |
| 2 | Extracted KG + LLM rewrite | 36.5% | 46.5% | 17.0% | −0.558 |
| 3 | GT KG + LLM rewrite | 42.8% | 40.2% | 17.0% | −0.772 |
| 4 | Extracted KG full triple SV | 25.7% | 52.8% | 21.5% | −0.225 |
| 5 | Extracted KG tail SV | 36.1% | 44.8% | 19.1% | −0.521 |
| 6 | GT KG full triple SV | 37.4% | 44.8% | 17.8% | −0.534 |
| 7 | GT KG tail SV | 58.3% | 33.5% | 8.3% | −1.184 |
| 8 | Extracted KG LLM pair SV | 69.1% | 21.5% | 9.3% | −1.680 |
| 9 | **GT KG LLM pair SV** | **84.8%** | **11.7%** | **3.5%** | **−2.307** |
| 10 | Extracted KG GT pair SV | 37.6% | 44.6% | 17.8% | −0.568 |
| 11 | GT KG GT pair SV | 61.3% | 29.6% | 9.1% | −1.366 |

## Score Distribution (% of images)

| Exp | Method | 0 | 1 | 2 | 3 | 4 | 5 | Low (0–2) | High (3–5) |
|-----|--------|---|---|---|---|---|---|-----------|------------|
| 0 | Baseline | 5.0% | 2.6% | 13.1% | 16.4% | 53.4% | 9.4% | 20.7% | 79.3% |
| 1 | LLM rewrite (no KG) | 14.5% | 3.4% | 18.4% | 8.8% | 38.5% | 16.5% | 36.2% | 63.8% |
| 2 | Extracted KG + LLM rewrite | 15.8% | 5.5% | 19.1% | 6.6% | 36.6% | 16.4% | 40.4% | 59.6% |
| 3 | GT KG + LLM rewrite | 19.9% | 10.1% | 18.8% | 7.4% | 29.7% | 14.2% | 48.7% | 51.3% |
| 4 | Extracted KG full triple SV | 10.3% | 4.0% | 17.8% | 8.3% | 40.8% | 18.8% | 32.1% | 67.9% |
| 5 | Extracted KG tail SV | 17.5% | 5.6% | 16.4% | 8.3% | 35.4% | 16.8% | 39.6% | 60.4% |
| 6 | GT KG full triple SV | 14.7% | 7.2% | 20.3% | 7.3% | 34.5% | 15.9% | 42.2% | 57.8% |
| 7 | GT KG tail SV | 27.0% | 10.7% | 21.1% | 6.9% | 23.6% | 10.6% | 58.8% | 41.2% |
| 8 | Extracted KG LLM pair SV | 44.9% | 7.8% | 13.5% | 5.1% | 19.4% | 9.3% | 66.2% | 33.8% |
| 9 | GT KG LLM pair SV | 54.6% | 14.3% | 15.4% | 3.2% | 9.0% | 3.5% | 84.3% | 15.7% |
| 10 | Extracted KG GT pair SV | 16.7% | 5.8% | 18.6% | 7.7% | 35.4% | 15.8% | 41.1% | 58.9% |
| 11 | GT KG GT pair SV | 32.5% | 10.1% | 19.2% | 4.9% | 25.6% | 7.8% | 61.7% | 38.3% |

## Breakdown by Bias Type (Mean Score, 460 Common Cases)

| Bias Type | Baseline | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 | E11 |
|-----------|----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|------|
| age | 3.26 | 3.10 | 2.46 | 1.90 | 2.46 | 2.51 | 2.08 | 1.85 | 1.51 | 0.49 | 2.56 | 1.28 |
| disability | 3.19 | 2.57 | 2.57 | 2.62 | 3.05 | 3.38 | 2.19 | 2.71 | 1.86 | 1.10 | 2.62 | 2.48 |
| gender | 3.29 | 2.83 | 2.84 | 2.83 | 3.11 | 2.79 | 2.42 | 2.01 | 1.67 | 0.91 | 2.64 | 1.80 |
| nationality | 3.26 | 2.54 | 2.44 | 2.95 | 2.46 | 3.13 | 1.92 | 2.56 | 0.92 | 0.85 | 2.79 | 2.05 |
| physical-appearance | 3.73 | 3.67 | 3.87 | 2.47 | 3.93 | 3.53 | 3.47 | 4.13 | 3.00 | 0.47 | 4.07 | 1.00 |
| profession | 3.58 | 3.45 | 3.13 | 2.74 | 3.40 | 3.06 | 3.25 | 2.36 | 2.11 | 1.40 | 3.09 | 2.23 |
| race | 3.36 | 2.69 | 2.45 | 2.40 | 3.11 | 2.65 | 2.92 | 1.94 | 1.26 | 0.99 | 2.64 | 2.21 |
| race-color | 3.09 | 2.48 | 2.59 | 2.04 | 2.85 | 2.39 | 2.43 | 2.04 | 0.96 | 0.70 | 2.28 | 0.80 |
| religion | 3.89 | 2.83 | 3.53 | 2.94 | 3.50 | 3.75 | 3.22 | 3.17 | 2.14 | 1.81 | 3.86 | 2.75 |
| socioeconomic | 2.81 | 2.30 | 2.87 | 2.63 | 2.85 | 2.89 | 2.70 | 2.19 | 1.87 | 0.43 | 2.43 | 1.59 |

## Key Findings

### 1. Steering vectors with LLM-generated pairs are the most effective

The top 3 methods by stereotype reduction:
1. **Exp 9 — GT KG + LLM pair SV** (−67.7%): By far the strongest debiaser. 84.8% of cases improved, only 3.5% worsened.
2. **Exp 8 — Extracted KG + LLM pair SV** (−49.3%): Second strongest. 69.1% improved, 9.3% worsened.
3. **Exp 11 — GT KG + GT pair SV** (−40.1%): Third strongest. 61.3% improved, 9.1% worsened.

### 2. GT knowledge graphs consistently outperform extracted KGs

Comparing matched pairs using the same debiasing method:
- LLM rewrite: GT KG (E3, −22.7%) > Extracted KG (E2, −16.4%) > No KG (E1, −11.7%)
- Full triple SV: GT KG (E6, −15.7%) > Extracted KG (E4, −6.6%)
- Tail SV: GT KG (E7, −34.7%) > Extracted KG (E5, −15.3%)
- LLM pair SV: GT KG (E9, −67.7%) > Extracted KG (E8, −49.3%)
- GT pair SV: GT KG (E11, −40.1%) > Extracted KG (E10, −16.7%)

### 3. LLM-generated steering pairs outperform GT pairs and other SV signals

Comparing steering signal types (with GT KG):
- LLM pair SV (E9, −67.7%) > GT pair SV (E11, −40.1%) > Tail SV (E7, −34.7%) > Full triple SV (E6, −15.7%)

### 4. Tail-only SV outperforms full-triple SV

- GT KG tail (E7, −34.7%) > GT KG full triple (E6, −15.7%)
- Extracted KG tail (E5, −15.3%) > Extracted KG full triple (E4, −6.6%)

Mean-pooled tail embeddings provide a more targeted steering signal than full KG triples.

### 5. Exp 9 may be over-debiasing

With a mean score of 1.10 (vs 3.41 baseline) and 54.6% of images scoring 0, Exp 9 may be pushing images too far from their original content. This warrants a quality/fidelity check to ensure images remain semantically coherent.

### 6. Hardest bias types to debias

Religion (baseline 3.89) and physical-appearance (baseline 3.73) remain the most resistant to debiasing across most methods, except for the strongest steering approaches (E8, E9, E11).

## Caveats

- **Partial data**: Experiments 1–11 are 25–50% complete. Results may shift as remaining cases are generated and evaluated.
- **Single evaluator**: All scores come from one VLM (Qwen3-VL). Cross-model validation is recommended.
- **Single alpha**: All steering experiments use alpha=2.0. Optimal alpha may vary by method and bias type.
- **No image quality metric**: Lower stereotype scores are desirable, but we don't yet measure whether images remain faithful to the neutral prompt.


## Ziyang start

## Comparison of Average Stereotype Scores Across Baseline Prompt Types and Neutral-Only Experimental Settings
| Configuration | Prompt Type | Mean Score | Sample Size |
| --- | --- | ---: | ---: |
| Baseline | Anti-stereotype prompt (`qwen_anti`) | 0.718 | 1896 |
| Baseline | Neutral prompt (`qwen_neutral`) | 3.462 | 1895 |
| Baseline | Stereotype prompt (`qwen_stereo`) | 4.243 | 1896 |
| Exp-01 | Neutral prompt only | 3.051 | 2070 |
| Exp-02 | Neutral prompt only | 2.917 | 2070 |
| Exp-03 | Neutral prompt only | 2.600 | 2070 |

Notes:
- Results are computed on the shared subset across `baseline`, `exp_01`, `exp_02`, and `exp_03`.
- The shared subset contains 690 cases and 2070 `(case, seed)` samples in total.
- Baseline scores are taken from `benchmark_scores.csv` using Qwen annotations only: `qwen_anti`, `qwen_neutral`, and `qwen_stereo`.
- `Exp-01`, `Exp-02`, and `Exp-03` use neutral prompts only for image generation, and the reported values are the mean evaluation `score` from the corresponding JSONL files.
- Some baseline rows have blank Qwen score fields, so the effective sample sizes differ across baseline prompt types.


## Seed-wise Comparison of Average Stereotype Scores Across Baseline Prompt Types and Neutral-Only Experimental Settings
| Configuration | Prompt Type | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | ---: | ---: | ---: |
| Baseline | Anti-stereotype prompt (`qwen_anti`) | 0.802 (`n=516`) | 0.683 (`n=690`) | 0.689 (`n=690`) |
| Baseline | Neutral prompt (`qwen_neutral`) | 3.495 (`n=515`) | 3.444 (`n=690`) | 3.455 (`n=690`) |
| Baseline | Stereotype prompt (`qwen_stereo`) | 4.210 (`n=516`) | 4.238 (`n=690`) | 4.273 (`n=690`) |
| Exp-01 | Neutral prompt only | 3.077 (`n=690`) | 3.036 (`n=690`) | 3.039 (`n=690`) |
| Exp-02 | Neutral prompt only | 2.897 (`n=690`) | 2.901 (`n=690`) | 2.954 (`n=690`) |
| Exp-03 | Neutral prompt only | 2.581 (`n=690`) | 2.603 (`n=690`) | 2.617 (`n=690`) |

Notes:
- Results are computed on the shared subset across `baseline`, `exp_01`, `exp_02`, and `exp_03`.
- The shared subset contains 690 cases for each seed, i.e., 2070 `(case, seed)` samples in total.
- Baseline scores are taken from `benchmark_scores.csv` using Qwen annotations only: `qwen_anti`, `qwen_neutral`, and `qwen_stereo`.
- `Exp-01`, `Exp-02`, and `Exp-03` use neutral prompts only for image generation, and each seed contains 690 valid evaluation scores.
- Some baseline rows have blank Qwen score fields, so the effective sample sizes for baseline differ by prompt type and seed.


## 19/04/2026 updated
### Table 1: Overall Comparison of Neutral-Prompt Stereotype Scores Across Experimental Settings
| Configuration | Prompt Type | Mean Score | Sample Size |
| --- | --- | ---: | ---: |
| Exp-00 (Baseline) | Neutral prompt only | 3.391 | 5493 |
| Exp-01 | Neutral prompt only | 3.009 | 5493 |
| Exp-02 | Neutral prompt only | 2.965 | 5493 |
| Exp-03 | Neutral prompt only | 2.597 | 5493 |

Notes:
- Exp-01 -> LLM Prompt Rewrite (No KG)
- Exp-02 -> Extracted KG + LLM Prompt Rewrite
- Exp-03 ->  Ground Truth KG + LLM Prompt Rewrite
- Results are computed on the shared subset across `exp_00`, `exp_01`, `exp_02`, and `exp_03`.
- The shared subset contains 1831 cases and 5493 `(case, seed)` samples in total.
- `Exp-00` is treated as the baseline.
- All configurations use neutral prompts only for image generation, and the reported values are the mean evaluation `score` from the corresponding JSONL files.


### Table 2: Seed-wise Comparison of Neutral-Prompt Stereotype Scores Across Experimental Settings
| Configuration | Prompt Type | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | ---: | ---: | ---: |
| Exp-00 (Baseline) | Neutral prompt only | 3.429 (`n=1831`) | 3.372 (`n=1831`) | 3.371 (`n=1831`) |
| Exp-01 | Neutral prompt only | 3.004 (`n=1831`) | 3.001 (`n=1831`) | 3.023 (`n=1831`) |
| Exp-02 | Neutral prompt only | 2.957 (`n=1831`) | 2.939 (`n=1831`) | 2.999 (`n=1831`) |
| Exp-03 | Neutral prompt only | 2.602 (`n=1831`) | 2.602 (`n=1831`) | 2.587 (`n=1831`) |


Notes:
- Exp-01 -> LLM Prompt Rewrite (No KG)
- Exp-02 -> Extracted KG + LLM Prompt Rewrite
- Exp-03 ->  Ground Truth KG + LLM Prompt Rewrite
- Results are computed on the shared subset across `exp_00`, `exp_01`, `exp_02`, and `exp_03`.
- The shared subset contains 1831 cases for each seed, i.e., 5493 `(case, seed)` samples in total.
- `Exp-00` is treated as the baseline.
- All configurations use neutral prompts only for image generation.

## Ziyang end