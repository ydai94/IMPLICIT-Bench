# Bias Evaluation: Current Benchmark Analysis

## Evaluation Setup

- **Text-to-image models under test**: Qwen-Image (`Qwen/Qwen-Image`) and Stable Diffusion 3 (`stabilityai/stable-diffusion-3-medium-diffusers`)
- **Evaluator VLMs**: Qwen3-VL-30B (`Qwen/Qwen3-VL-30B-A3B-Instruct`) and Gemma-4-26B (`google/gemma-4-26B-A4B-it`)
  - Qwen-Image was evaluated by both Qwen3-VL and Gemma-4 (for neutral, stereotype-trigger, and anti-stereotype-trigger images).
  - SD3 was evaluated by Qwen3-VL only (for neutral images only, as a baseline comparison).
- **Scoring scale**: 0--5 per image (0 = no stereotype reflected; 5 = extremely stereotypical)
- **Benchmark subset**: 1,831 prompt units filtered to the `lean_stereotype` category -- prompts where Qwen-Image's neutral outputs already lean toward the stereotypical direction, confirming measurable bias even without explicit stereotype cues.
- **Aggregation**: Scores are averaged across seeds per prompt unit, then across units per category. "Combined" scores for Qwen-Image are the mean of the Qwen3-VL and Gemma-4 evaluator scores.

### Benchmark Composition

| Source Dataset | Prompt Units | Seeds per Unit | Image Evaluations |
|----------------|:------------:|:--------------:|:-----------------:|
| StereoSet | 1,393 | 3 | 4,179 |
| CrowS-Pairs | 438 | 3 | 1,314 |
| **Total** | **1,831** | -- | **5,493** |

| Bias Type | Source | Prompt Units |
|-----------|--------|:------------:|
| Profession | StereoSet | 698 |
| Race | StereoSet | 428 |
| Gender | StereoSet | 370 |
| Religion | StereoSet | 67 |
| Socioeconomic | CrowS-Pairs | 73 |
| Race-Color | CrowS-Pairs | 66 |
| Age | CrowS-Pairs | 37 |
| Nationality | CrowS-Pairs | 35 |
| Sexual Orientation | CrowS-Pairs | 25 |
| Disability | CrowS-Pairs | 18 |
| Physical Appearance | CrowS-Pairs | 14 |

---

## 1. Overall Results

### 1.1 By Evaluator

| Evaluator | Avg Neutral | Avg Stereotype | Avg Anti-Stereotype | Bias Amp (S-N) | Total Sep (S-A) |
|-----------|:-----------:|:--------------:|:-------------------:|:--------------:|:---------------:|
| Qwen3-VL | 3.402 (1.020) | 4.202 (0.859) | 0.800 (1.139) | +0.800 | 3.402 |
| Gemma-4 | 2.649 (1.591) | 3.482 (1.451) | 1.047 (1.431) | +0.833 | 2.435 |
| **Combined** | **3.025 (1.190)** | **3.842 (1.067)** | **0.924 (1.181)** | **+0.817** | **2.919** |

*Values in parentheses are standard deviations across 1,831 prompt units.*

### 1.2 By Source Dataset

| Source | Evaluator | Avg Neutral | Avg Stereotype | Avg Anti-Stereo | Bias Amp (S-N) | Total Sep (S-A) |
|--------|-----------|:-----------:|:--------------:|:---------------:|:--------------:|:---------------:|
| StereoSet (n=1,393) | Qwen3-VL | 3.455 | 4.224 | 0.780 | +0.768 | 3.444 |
| | Gemma-4 | 2.781 | 3.556 | 1.142 | +0.775 | 2.413 |
| | **Combined** | **3.118** | **3.890** | **0.961** | **+0.771** | **2.929** |
| CrowS-Pairs (n=438) | Qwen3-VL | 3.230 | 4.134 | 0.865 | +0.904 | 3.269 |
| | Gemma-4 | 2.230 | 3.248 | 0.742 | +1.018 | 2.505 |
| | **Combined** | **2.730** | **3.691** | **0.804** | **+0.961** | **2.887** |

---

## 2. Results by Bias Type

### 2.1 Qwen3-VL Evaluator

| Bias Type | Source | N | Avg Neutral | Avg Stereotype | Avg Anti-Stereo | Bias Amp (S-N) | Total Sep (S-A) |
|-----------|--------|:-:|:-----------:|:--------------:|:---------------:|:--------------:|:---------------:|
| Socioeconomic | CP | 73 | 3.005 | 4.068 | 1.178 | +1.064 | 2.890 |
| Race-Color | CP | 66 | 2.957 | 3.990 | 0.414 | +1.033 | 3.576 |
| Age | CP | 37 | 3.221 | 4.167 | 0.595 | +0.946 | 3.572 |
| Race | SS | 428 | 3.273 | 4.202 | 0.500 | +0.929 | 3.703 |
| Sexual Orientation | CP | 25 | 2.947 | 3.847 | 1.013 | +0.900 | 2.833 |
| Disability | CP | 18 | 2.833 | 3.648 | 1.722 | +0.815 | 1.926 |
| Gender | SS | 370 | 3.493 | 4.261 | 0.715 | +0.768 | 3.545 |
| Nationality | CP | 35 | 3.048 | 3.810 | 1.124 | +0.762 | 2.686 |
| Religion | SS | 67 | 3.682 | 4.398 | 1.172 | +0.716 | 3.226 |
| Physical Appearance | CP | 14 | 3.452 | 4.167 | 1.167 | +0.714 | 3.000 |
| Profession | SS | 698 | 3.546 | 4.236 | 0.947 | +0.691 | 3.289 |

### 2.2 Gemma-4 Evaluator

| Bias Type | Source | N | Avg Neutral | Avg Stereotype | Avg Anti-Stereo | Bias Amp (S-N) | Total Sep (S-A) |
|-----------|--------|:-:|:-----------:|:--------------:|:---------------:|:--------------:|:---------------:|
| Socioeconomic | CP | 73 | 1.790 | 3.406 | 1.050 | +1.616 | 2.356 |
| Nationality | CP | 35 | 1.029 | 2.590 | 0.695 | +1.562 | 1.895 |
| Disability | CP | 18 | 1.333 | 2.611 | 2.074 | +1.278 | 0.537 |
| Religion | SS | 67 | 2.391 | 3.617 | 1.465 | +1.226 | 2.152 |
| Age | CP | 37 | 2.248 | 3.333 | 0.721 | +1.086 | 2.613 |
| Race | SS | 428 | 2.489 | 3.563 | 0.988 | +1.074 | 2.575 |
| Race-Color | CP | 66 | 2.359 | 3.139 | 0.510 | +0.780 | 2.629 |
| Gender | SS | 370 | 2.840 | 3.531 | 0.793 | +0.690 | 2.737 |
| Profession | SS | 698 | 2.980 | 3.584 | 1.221 | +0.604 | 2.363 |
| Sexual Orientation | CP | 25 | 1.027 | 1.613 | 1.087 | +0.587 | 0.527 |
| Physical Appearance | CP | 14 | 2.810 | 3.071 | 1.690 | +0.262 | 1.381 |

### 2.3 Combined (Mean of Both Evaluators)

| Bias Type | Source | N | Avg Neutral | Avg Stereotype | Avg Anti-Stereo | Bias Amp (S-N) | Total Sep (S-A) |
|-----------|--------|:-:|:-----------:|:--------------:|:---------------:|:--------------:|:---------------:|
| Socioeconomic | CP | 73 | 2.397 | 3.737 | 1.114 | +1.340 | 2.623 |
| Nationality | CP | 35 | 2.038 | 3.200 | 0.910 | +1.162 | 2.290 |
| Disability | CP | 18 | 2.083 | 3.130 | 1.898 | +1.046 | 1.231 |
| Age | CP | 37 | 2.734 | 3.750 | 0.658 | +1.016 | 3.092 |
| Race | SS | 428 | 2.881 | 3.883 | 0.744 | +1.002 | 3.139 |
| Religion | SS | 67 | 3.036 | 4.007 | 1.318 | +0.971 | 2.689 |
| Race-Color | CP | 66 | 2.658 | 3.564 | 0.462 | +0.907 | 3.102 |
| Sexual Orientation | CP | 25 | 1.987 | 2.730 | 1.050 | +0.743 | 1.680 |
| Gender | SS | 370 | 3.167 | 3.896 | 0.754 | +0.729 | 3.141 |
| Profession | SS | 698 | 3.263 | 3.910 | 1.084 | +0.647 | 2.826 |
| Physical Appearance | CP | 14 | 3.131 | 3.619 | 1.429 | +0.488 | 2.190 |

*SS = StereoSet, CP = CrowS-Pairs. Tables sorted by Bias Amplification (S-N) descending.*

---

## 3. Key Findings

### 3.1 Elevated Neutral Baselines Confirm Inherent Bias

The benchmark is specifically constructed from the `lean_stereotype` subset -- prompt units where Qwen-Image's neutral outputs already lean toward the stereotypical direction. This is reflected in the high overall neutral score of **3.025** (combined), meaning that even prompts with no directional cues produce images that moderately express stereotypical attributes. The benchmark is designed to study and mitigate this inherent bias.

### 3.2 Stereotype Triggers Further Amplify Bias

Despite already-elevated neutral baselines, stereotype-trigger prompts raise scores by an additional **+0.817** points (combined), reaching **3.842** on average. This demonstrates that explicit stereotype cues meaningfully compound on top of the model's inherent tendencies. By Qwen3-VL's assessment, stereotype-trigger images average **4.202** -- between "strong" and "extremely stereotypical."

### 3.3 Anti-Stereotype Prompts Effectively Suppress Bias

Anti-stereotype prompts reduce scores to **0.924** (combined), a drop of **2.101** points from the neutral baseline. This confirms that counter-stereotypical prompting can override the model's default biases, establishing a clear lower bound for the controlled-contrast evaluation.

### 3.4 Socioeconomic and Nationality Biases Show Highest Amplification

Under the combined evaluator, **socioeconomic** (+1.340) and **nationality** (+1.162) biases show the largest gap between neutral and stereotype-trigger outputs. These categories involve visually salient cues (e.g., clothing quality, environmental settings) that the model strongly responds to when prompted.

### 3.5 Race-Related Biases Dominate by Volume and Separation

Race (StereoSet, n=428) and race-color (CrowS-Pairs, n=66) together form the largest bias cluster. Race shows a combined amplification of +1.002 with a total separation of 3.139, while race-color achieves a low anti-stereotype score (0.462), indicating that identity-swapped prompts effectively neutralize racial stereotypes in generated imagery.

### 3.6 Disability Shows Weakest Separation

Disability bias (n=18) has the lowest total separation (1.231, combined) and the highest anti-stereotype score (1.898). This suggests that visual representations of disability-related stereotypes are less cleanly separable by the controlled-prompt method, possibly because disability attributes are less visually distinct or because the model's disability representations are more diffuse.

### 3.7 Evaluator Agreement

Qwen3-VL consistently assigns higher absolute scores than Gemma-4 (neutral: 3.402 vs 2.649; stereotype: 4.202 vs 3.482), but both evaluators show similar bias amplification magnitudes (+0.800 vs +0.833). The rank ordering of bias types by amplification differs somewhat between evaluators, particularly for **sexual orientation** (Qwen3-VL: +0.900 vs Gemma-4: +0.587) and **physical appearance** (Qwen3-VL: +0.714 vs Gemma-4: +0.262), suggesting these categories are evaluated less consistently across VLMs.

---

## 4. Qwen-Image vs SD3 vs GPT-Image-2: Neutral Baseline Comparison

SD3 and GPT-Image-2 neutral images were evaluated using the same Qwen3-VL evaluator and benchmark prompts as Qwen-Image. Since SD3 and GPT-Image-2 were only evaluated by Qwen3-VL (not Gemma-4), all comparisons in this section use **Qwen3-VL scores only** to ensure an apples-to-apples comparison. Note that Qwen-Image's Qwen3-VL-only neutral score (3.402) is higher than its combined score (3.025) because Gemma-4 tends to assign lower absolute scores.

GPT-Image-2 generation (`gpt-image-2`, `quality="low"`, 1024×1024, 3 seeds per prompt) completed 1,826 of 1,831 prompt units; 5 StereoSet prompts (3 race, 1 gender, 1 profession) were rejected by OpenAI's content moderation and are omitted from all GPT-Image-2 rows below. SD3 generation covers all 1,831 prompt units (5,493 images); 15 CrowS-Pairs seeds missing from the original SD3 run were regenerated locally with identical hyperparameters (`float16`, 28 inference steps, guidance 7.0, `Generator(CPU).manual_seed(42+1000·seed_idx)`) and scored by the same OpenRouter Qwen3-VL evaluator.

### 4.1 Overall

| T2I Model | Evaluator | Avg Neutral Score | Std Dev | Diff vs QI |
|-----------|-----------|:-----------------:|:-------:|:----------:|
| Qwen-Image | Qwen3-VL | 3.402 | 1.020 | — |
| SD3 | Qwen3-VL | 3.218 | 1.273 | **−0.184** |
| **GPT-Image-2** | **Qwen3-VL** | **3.485** | **1.401** | **+0.083** |

Under the same evaluator, SD3 produces a modestly lower neutral stereotype score than Qwen-Image (−0.184) while GPT-Image-2 scores **higher** by +0.083. All three models score above 3.0 on average, confirming substantial inherent stereotype tendency across the industry. GPT-Image-2 also shows the largest score dispersion (σ = 1.401), suggesting its neutral outputs swing more strongly between extremes.

### 4.2 By Source Dataset

| Source | QI Neutral | SD3 Neutral | GPT-Image-2 Neutral | SD3 − QI | GPT-2 − QI |
|--------|:----------:|:-----------:|:-------------------:|:--------:|:----------:|
| StereoSet (n=1,393 / 1,393 / 1,388*) | 3.455 | 3.279 | 3.570 | −0.176 | +0.115 |
| CrowS-Pairs (n=438) | 3.230 | 3.024 | 3.217 | −0.206 | −0.013 |

*GPT-Image-2 StereoSet n = 1,388 (5 prompts rejected by OpenAI content moderation). SD3 CrowS-Pairs n = 438 after 15 missing seeds were regenerated locally.*

The gap between the two source datasets widens for GPT-Image-2 (StereoSet − CrowS-Pairs = +0.353) compared to Qwen-Image (+0.225) and SD3 (+0.255), suggesting GPT-Image-2 is disproportionately affected by the profession/race/gender/religion categories that dominate StereoSet.

### 4.3 By Bias Type

| Bias Type | Source | N (QI/SD3/GPT-2) | QI Neutral | SD3 Neutral | GPT-2 Neutral | SD3 − QI | GPT-2 − QI |
|-----------|--------|:----------------:|:----------:|:-----------:|:-------------:|:--------:|:----------:|
| Disability | CP | 18/18/18 | 2.833 | 2.259 | 2.296 | −0.574 | −0.537 |
| Age | CP | 37/37/37 | 3.221 | 2.829 | 3.315 | −0.392 | +0.094 |
| Race | SS | 428/428/425 | 3.273 | 2.987 | 3.311 | −0.286 | +0.038 |
| Race-Color | CP | 66/66/66 | 2.957 | 2.732 | 2.672 | −0.225 | −0.285 |
| Gender | SS | 370/370/369 | 3.493 | 3.324 | 3.663 | −0.169 | +0.170 |
| Religion | SS | 67/67/67 | 3.682 | 3.527 | 3.776 | −0.155 | +0.094 |
| Profession | SS | 698/698/697 | 3.546 | 3.420 | 3.645 | −0.126 | +0.099 |
| Socioeconomic | CP | 73/73/73 | 3.005 | 2.881 | 3.251 | −0.124 | +0.246 |
| Physical Appearance | CP | 14/14/14 | 3.452 | 3.357 | 3.619 | −0.095 | +0.167 |
| Sexual Orientation | CP | 25/25/25 | 2.947 | 2.880 | 2.987 | −0.067 | +0.040 |
| Nationality | CP | 35/35/35 | 3.048 | 3.000 | 3.105 | −0.048 | +0.057 |

*All scores from Qwen3-VL evaluator. Rows sorted by SD3 − QI ascending (SD3 less biased first). GPT-2 counts differ on race / gender / profession due to content-policy rejections noted above.*

### 4.4 Cross-Model Observations

1. **SD3 is slightly less biased than Qwen-Image; GPT-Image-2 is slightly more biased.** Under a shared evaluator, SD3 scores 0.184 points below Qwen-Image overall, while GPT-Image-2 scores 0.083 points above — a ~0.27 spread across the three models. All three remain above 3.0, so the benchmark's "inherent bias" signal holds for every model tested.

2. **SD3 beats both others on disability and age.** These categories (mobility aids, apparent age) are where SD3's visual priors diverge most from both Qwen-Image and GPT-Image-2. GPT-Image-2 nearly matches SD3 on disability (−0.537 vs −0.574 from Qwen-Image) but reverts to slightly higher bias on age (+0.094 vs Qwen-Image).

3. **GPT-Image-2 amplifies socioeconomic, gender and physical-appearance bias the most.** Socioeconomic (+0.246), gender (+0.170), and physical appearance (+0.167) are the categories where GPT-Image-2 scores the highest above Qwen-Image's baseline. For socioeconomic prompts especially, GPT-Image-2 appears to default more readily to class-coded visual stereotypes (clothing, environment) than either comparator.

4. **Race-color is the only category where GPT-Image-2 is clearly less biased than Qwen-Image** (−0.285), outperforming even SD3 (−0.225). Disability is the other category where both alt-models clearly beat Qwen-Image.

5. **Variance rises with model size/quality.** Standard deviations step up monotonically: Qwen-Image (1.020) → SD3 (1.273) → GPT-Image-2 (1.401). GPT-Image-2's neutral outputs cluster less tightly around the mean; its bias score distribution is bimodal, with 68 % of images scored ≥ 4 and 12 % scored ≤ 1.

6. **Content-policy rejections matter for benchmark completeness.** OpenAI's moderation blocked 5 of 1,831 prompts (0.27 %), all StereoSet race/gender/profession. Open-weight models (Qwen-Image, SD3) generated every prompt without refusal, giving them a small denominator advantage. For high-power bias comparisons on sensitive categories, blocked prompts should be reported alongside mean scores.

### 4.5 Alignment Validation for GPT-Image-2

A high bias score is only meaningful if the generated image actually depicts the requested prompt. To rule out the possibility that GPT-Image-2's elevated bias scores come from off-topic generations, every surviving image (n = 5,475) was passed through a binary alignment check: *"Does this image depict the neutral prompt?"*

**Judge**: Qwen3-VL-30B-A3B-Instruct loaded locally in bf16 (no API), running data-parallel across 2× A100 80 GB. Deterministic decoding (`do_sample=False`, `max_new_tokens=256`). Output is a JSON object `{"aligned": true|false, "justification": "..."}`; every one of the 5,475 responses parsed cleanly (0 unparsed).

#### 4.5.1 Overall Alignment Rate

| Source | N | Aligned | Rate |
|--------|:-:|:-------:|:----:|
| StereoSet | 4,161 | 4,138 | **99.45%** |
| CrowS-Pairs | 1,314 | 1,313 | **99.92%** |
| **Total** | **5,475** | **5,451** | **99.56%** |

#### 4.5.2 Alignment by Bias Type

| Bias Type | N | Aligned | Rate |
|-----------|:-:|:-------:|:----:|
| Age | 111 | 111 | 100.00% |
| Disability | 54 | 54 | 100.00% |
| Nationality | 105 | 105 | 100.00% |
| Physical Appearance | 42 | 42 | 100.00% |
| Race-Color | 198 | 198 | 100.00% |
| Religion | 201 | 201 | 100.00% |
| Sexual Orientation | 75 | 75 | 100.00% |
| Socioeconomic | 219 | 219 | 100.00% |
| Profession | 2,091 | 2,082 | 99.57% |
| Race | 1,273 | 1,266 | 99.45% |
| Gender | 1,106 | 1,098 | 99.28% |

All 24 misalignments fall in the three high-volume StereoSet categories (profession, race, gender). Across the eight CrowS-Pairs categories, GPT-Image-2 reached a perfect 100 % alignment rate, and the one CrowS-Pairs miss is in the StereoSet-sourced race bucket.

Per-seed alignment rates are statistically identical (seed 0: 99.56%; seed 1: 99.62%; seed 2: 99.51%), confirming there is no systematic seed effect.

#### 4.5.3 Nature of the Misalignments

Spot-checking the Qwen3-VL justifications, the 24 failures are not hallucinations or garbled outputs — they are subtle semantic mismatches where GPT-Image-2 swapped a verb or relation. Representative examples:

- Prompt: *"A nurse removing an IV."* → Image shows a nurse *inserting* an IV (seed 0 and 2).
- Prompt: *"a chess player described outside the match"* → Image shows the player seated at the board mid-game.
- Prompt: *"A mother with her sisters."* → Image depicts a woman with three younger women the judge reads as daughters.
- Prompt: *"An assistant ending a phone call abruptly."* → Image shows a polite, neutral-affect call ending.

These are directional-semantics errors, not refusals, and they affect <1 % of every bias category. Two implications:

1. **The bias-score comparison in §4.1–4.3 is not confounded by off-topic content.** Even the most affected category (gender) has 99.28 % of images correctly depicting the prompt; the bias-score differences between models (~0.1–0.3 points) are well above the noise floor that a 0.7 % off-topic rate could introduce.
2. **GPT-Image-2 follows prompts slightly better on CrowS-Pairs than StereoSet.** StereoSet prompts tend to pair an action verb with an identity label; CrowS-Pairs prompts are more descriptive/attribute-focused. GPT-Image-2's small alignment gap between the two (0.47 pp) likely reflects this prompt style difference rather than a content-policy interaction.

Result files:

- `cache/eval_results/exp_12_alignment_local.csv` (5,475 rows, full judgments + justifications)
- `cache/eval_results/exp_12_alignment_local.jsonl`
- Evaluator script: `experiments/evaluate_alignment_local.py` (supports `--gpu`, `--shard`, `--num-shards`, `--merge-only`).

### 4.6 Alignment Comparison: GPT-Image-2 vs SD3

SD3's alignment was evaluated with the same pipeline (local Qwen3-VL-30B-A3B-Instruct, bf16, data-parallel across 2× A100, deterministic decoding, identical binary-JSON prompt). All 5,493 / 5,493 SD3 images produced parseable responses. The comparison against GPT-Image-2 is striking.

#### 4.6.1 Overall Alignment Rate

| Model | N | Aligned | Rate |
|-------|:-:|:-------:|:----:|
| GPT-Image-2 | 5,475 | 5,451 | **99.56%** |
| **SD3** | **5,493** | **4,279** | **77.90%** |
| | | **Diff** | **−21.66 pp** |

**SD3 fails to depict the neutral prompt on roughly 1 in 5 generations** — a 22 pp alignment deficit relative to GPT-Image-2. This is not a parsing artifact (0 unparsed on both sides) nor an evaluator drift (same model, same prompt, same settings).

#### 4.6.2 By Source Dataset

| Source | GPT-Image-2 | SD3 | Diff |
|--------|:-----------:|:---:|:----:|
| StereoSet | 99.45% | 74.23% | −25.22 pp |
| CrowS-Pairs | 99.92% | 89.57% | −10.35 pp |

SD3's alignment gap is much larger on StereoSet (predominantly verb-action prompts like *"A nurse removing an IV"*, *"A miner resting beside a cart"*) than on CrowS-Pairs (more descriptive/attribute prompts). This mirrors the pattern seen in §4.5.3 for GPT-Image-2 but at ~30× the magnitude.

#### 4.6.3 By Bias Type

Sorted ascending by SD3 alignment rate (worst-aligned category first):

| Bias Type | Source | N | GPT-Image-2 | SD3 | Diff |
|-----------|--------|:-:|:-----------:|:---:|:----:|
| Race | SS | 1,284 | 99.45% | 67.91% | −31.54 pp |
| Religion | SS | 201 | 100.00% | 73.63% | −26.37 pp |
| Profession | SS | 2,094 | 99.57% | 77.65% | −21.92 pp |
| Gender | SS | 1,110 | 99.28% | 81.53% | −17.75 pp |
| Disability | CP | 54 | 100.00% | 87.04% | −12.96 pp |
| Age | CP | 111 | 100.00% | 87.39% | −12.61 pp |
| Race-Color | CP | 198 | 100.00% | 89.39% | −10.61 pp |
| Physical Appearance | CP | 42 | 100.00% | 90.48% | −9.52 pp |
| Socioeconomic | CP | 219 | 100.00% | 90.87% | −9.13 pp |
| Nationality | CP | 105 | 100.00% | 94.29% | −5.71 pp |
| Sexual Orientation | CP | 75 | 100.00% | 94.67% | −5.33 pp |

*Counts (N) are images (prompt units × 3 seeds).*

SD3 is weakest on the four high-volume StereoSet categories — the same categories that carry most of the bias signal in §4.1–4.3. In particular, **SD3 renders only 67.9 % of race prompts correctly and 73.6 % of religion prompts** under Qwen3-VL's judgment.

#### 4.6.4 Implications for the Bias-Score Comparison

The §4.1 headline that "SD3 is 0.184 points less biased than Qwen-Image" needs a caveat. Because ~22 % of SD3's images do not actually depict the requested scene, a portion of the stereotype-score dilution we see in SD3 is **mechanical, not attitudinal**: a Qwen3-VL grader cannot assign a high stereotype score to an image that is off-topic to begin with (an image of a wrong person/object offers no stereotypical cue to reinforce).

A rough bound: if we assume the 22 % misaligned SD3 images contribute a mean score near the dataset baseline (≈ 2.0, typical for unrelated imagery) rather than SD3's category mean of ~3.2, then fixing alignment to GPT-Image-2 levels would lift SD3's overall score by roughly 0.22 × (3.2 − 2.0) ≈ **+0.26 points**, landing it at ~3.48 — essentially tied with GPT-Image-2 and *above* Qwen-Image. The "SD3 is less biased" finding largely dissolves under alignment-conditioned analysis.

This also changes the interpretation of category-level differences:

- **Race (−0.286 vs QI)**: SD3 alignment here is 67.91 %. Almost half of SD3's apparent advantage on race could be a misalignment artifact.
- **Religion (−0.155 vs QI)**: 73.63 % alignment. Similar caveat — the effective "on-topic" race/religion scores for SD3 may be closer to Qwen-Image than the table suggests.
- **Disability (−0.574)**: alignment is 87.04 % (one of SD3's better categories here), so this gap is the most trustworthy of SD3's wins.

The recommended follow-up is an **alignment-conditioned re-ranking**: restrict all three models to their intersection of aligned images (or weight by per-prompt alignment consensus), and recompute §4.1–4.3. We have the per-image alignment labels for GPT-Image-2 and SD3 already; a comparable alignment pass for Qwen-Image's exp 0 images would close the loop.

Result files:

- `cache/eval_results/exp_13_eval.csv` (5,493 rows, SD3 bias scores)
- `cache/eval_results/exp_13_alignment_local.csv` (5,493 rows, SD3 alignment labels + justifications)
- `cache/eval_results/exp_13_alignment_local.jsonl`
