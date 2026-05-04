# Baseline Comparison: Qwen-Image vs SD3 vs GPT-Image-2

Three text-to-image models evaluated on the same 1,831-prompt benchmark (3 seeds per prompt, 5,493 target images per model). For each model we report (1) per-image stereotype-bias score on neutral prompts (0–5 scale, judged by Qwen3-VL) and (2) per-image prompt-alignment rate (binary true/false, judged by Qwen3-VL). Lower bias = less stereotypical; higher alignment = more faithful to the neutral prompt.

## Setup

| Item | Value |
|---|---|
| Prompt set | `data/benchmark_prompts.csv` — 1,831 prompt units (1,393 StereoSet + 438 CrowS-Pairs) |
| Seeds | 3 per prompt → 5,493 expected images |
| Evaluator (bias) | Qwen3-VL-30B-A3B-Instruct via OpenRouter, deterministic decoding |
| Evaluator (alignment) | Qwen3-VL-30B-A3B-Instruct — local bf16 for SD3 / GPT-Image-2; OpenRouter for Qwen-Image exp 0 |
| Aggregation | Mean across seeds per prompt unit, then across prompt units |

Image-count notes:

- **Qwen-Image**: 5,493 / 5,493 (exp 0)
- **SD3**: 5,493 / 5,493 (exp 13 — 15 CrowS-Pairs seeds were regenerated locally with identical hyperparameters: `float16`, 28 steps, guidance 7.0, `Generator(CPU).manual_seed(42+1000·seed_idx)`)
- **GPT-Image-2**: 5,475 / 5,493 (exp 12 — 5 StereoSet prompts rejected by OpenAI content moderation, losing 15 images)

## 1. Bias Score (0–5, higher = more stereotypical)

### 1.1 Overall

| Model | Images | Prompt Units | Mean | Std |
|---|:-:|:-:|:-:|:-:|
| Qwen-Image | 5,493 | 1,831 | **3.391** | 1.026 |
| SD3 | 5,493 | 1,831 | **3.218** | 1.273 |
| GPT-Image-2 | 5,475 | 1,826 | **3.485** | 1.401 |

All three models sit above 3.0 — the benchmark's "lean-stereotype" subset produces measurable bias from every model tested. Total spread: ~0.27 points. GPT-Image-2 is the most biased, SD3 the least (but see §3 on alignment-conditioning).

### 1.2 By Source Dataset

| Source | Qwen-Image | SD3 | GPT-Image-2 |
|--------|:----------:|:---:|:-----------:|
| StereoSet (n=1,393) | 3.442 | 3.279 | 3.570 |
| CrowS-Pairs (n=438) | 3.229 | 3.024 | 3.217 |

### 1.3 By Bias Type (sorted by Qwen-Image mean, descending)

| Bias Type | Source | N | Qwen-Image | SD3 | GPT-Image-2 |
|-----------|--------|:-:|:----------:|:---:|:-----------:|
| Religion | SS | 67 | 3.682 | 3.527 | 3.776 |
| Profession | SS | 698 | 3.527 | 3.420 | 3.645 |
| Gender | SS | 370 | 3.484 | 3.324 | 3.663 |
| Physical Appearance | CP | 14 | 3.452 | 3.357 | 3.619 |
| Race | SS | 428 | 3.267 | 2.987 | 3.311 |
| Age | CP | 37 | 3.216 | 2.829 | 3.315 |
| Nationality | CP | 35 | 3.048 | 3.000 | 3.105 |
| Socioeconomic | CP | 73 | 3.005 | 2.881 | 3.251 |
| Race-Color | CP | 66 | 2.955 | 2.732 | 2.672 |
| Sexual Orientation | CP | 25 | 2.947 | 2.880 | 2.987 |
| Disability | CP | 18 | 2.833 | 2.259 | 2.296 |

*SS = StereoSet, CP = CrowS-Pairs.*

## 2. Alignment Rate (does the image depict the neutral prompt?)

### 2.1 Overall

| Model | Images | Aligned | Rate |
|---|:-:|:-:|:-:|
| Qwen-Image | 5,493 | 4,808 | **87.53%** |
| SD3 | 5,493 | 4,279 | **77.90%** |
| GPT-Image-2 | 5,475 | 5,451 | **99.56%** |

GPT-Image-2 nearly always depicts the prompt. Qwen-Image misses ~12 % and SD3 misses ~22 %.

### 2.2 By Source Dataset

| Source | Qwen-Image | SD3 | GPT-Image-2 |
|--------|:----------:|:---:|:-----------:|
| StereoSet | 85.09% | 74.23% | 99.45% |
| CrowS-Pairs | 95.28% | 89.57% | 99.92% |

StereoSet prompts (verb-action style, e.g. "a nurse removing an IV") are harder for all three models, but the gap is much wider for SD3 than for the other two.

### 2.3 By Bias Type (sorted by SD3 alignment, ascending — worst-aligned first)

| Bias Type | Source | N | Qwen-Image | SD3 | GPT-Image-2 |
|-----------|--------|:-:|:----------:|:---:|:-----------:|
| Race | SS | 1,284 | 81.62% | 67.91% | 99.45% |
| Religion | SS | 201 | 90.55% | 73.63% | 100.00% |
| Profession | SS | 2,094 | 87.01% | 77.65% | 99.57% |
| Gender | SS | 1,110 | 88.74% | 81.53% | 99.28% |
| Disability | CP | 54 | 98.15% | 87.04% | 100.00% |
| Age | CP | 111 | 94.59% | 87.39% | 100.00% |
| Race-Color | CP | 198 | 96.97% | 89.39% | 100.00% |
| Physical Appearance | CP | 42 | 92.86% | 90.48% | 100.00% |
| Socioeconomic | CP | 219 | 94.06% | 90.87% | 100.00% |
| Nationality | CP | 105 | 100.00% | 94.29% | 100.00% |
| Sexual Orientation | CP | 75 | 94.67% | 94.67% | 100.00% |

*Counts (N) are images (prompt units × 3 seeds).*

## 3. Composite (Bias × Alignment)

To reward models that are both faithful to the prompt *and* low in stereotype, we combine the two metrics into a single score:

**Composite** = `aligned_rate × (1 − bias_mean/5) × 100`

This matches the definition used in `reports/all_experiments_comparison.md` for the 12-experiment Qwen-Image study. Higher = better. Computed over each model's full image pool (not the 460-case intersection).

| Model | Images | Bias mean | Aligned % | **Composite** |
|---|:-:|:-:|:-:|:-:|
| Qwen-Image (exp 0) | 5,493 | 3.391 | 87.53% | **28.17** |
| SD3 (exp 13) | 5,493 | 3.218 | 77.90% | **27.76** |
| GPT-Image-2 (exp 12) | 5,475 | 3.485 | 99.56% | **30.17** |

**Rank reversal under composite.** On raw bias, SD3 looked like the winner (3.218, lowest). On composite, SD3 falls *below* Qwen-Image (27.76 vs 28.17) because its alignment penalty wipes out the bias-score advantage. GPT-Image-2 moves from worst on bias to best on composite — nearly perfect alignment (99.56 %) more than pays for the 0.09-point bias deficit.

## 4. Combined Reading

1. **Bias headline**: GPT-Image-2 > Qwen-Image > SD3 by raw score (3.485 > 3.391 > 3.218).
2. **Alignment headline**: GPT-Image-2 > Qwen-Image > SD3 by faithfulness (99.56 % > 87.53 % > 77.90 %).
3. **Composite headline**: **GPT-Image-2 > Qwen-Image > SD3** (30.17 > 28.17 > 27.76). SD3's low bias does not survive alignment-weighting.
4. **The bias ordering is partly a consequence of the alignment ordering.** A misaligned image — one that does not depict the prompt — tends to get scored lower by the bias grader because it cannot "reinforce" the stereotype in a scene that is off-topic. SD3 has the lowest bias score *and* the worst alignment; a portion of its apparent "less biased" result is mechanical, not attitudinal.
4. **Race is the sharpest example.** SD3 aligns only 67.9 % of race prompts and scores a low 2.987 on race stereotype; GPT-Image-2 aligns 99.5 % of race prompts and scores higher 3.311. Whether SD3's lower race-bias score would survive alignment-conditioning is unclear from this data alone.
5. **Qwen-Image is the middle ground on both dimensions**: not the best-aligned, not the worst; not the most biased, not the least. It is the only model that can be directly compared to GPT-Image-2 at roughly similar alignment rates for most CrowS-Pairs categories.
6. **Evaluator note**: Qwen-Image alignment was judged via OpenRouter (`qwen/qwen3-vl-30b-a3b-instruct`); SD3 and GPT-Image-2 alignment via the same model run locally in bf16. Same model, same prompt, same decoding settings, but slightly different serving stacks — differences of a few points between the API and local runs are plausible and should not be over-interpreted.

## 5. Source Files

| Table | CSV |
|---|---|
| Qwen-Image bias | `cache/eval_results/exp_00_eval.csv` |
| Qwen-Image alignment | `cache/eval_results/exp_00_alignment.csv` |
| SD3 bias | `cache/eval_results/exp_13_eval.csv` |
| SD3 alignment | `cache/eval_results/exp_13_alignment_local.csv` |
| GPT-Image-2 bias | `cache/eval_results/exp_12_eval.csv` |
| GPT-Image-2 alignment | `cache/eval_results/exp_12_alignment_local.csv` |

Generation configs, evaluator scripts, and the per-prompt breakdown live in `experiments/`. See `qwen_image_bias_analysis.md` for the full 12-experiment Qwen-Image steering analysis that motivated this cross-model comparison.
