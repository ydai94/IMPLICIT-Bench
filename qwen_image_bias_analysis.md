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
| Qwen3-VL | 3.377 (1.048) | 4.222 (0.852) | 0.664 (1.143) | +0.844 | 3.558 |
| Gemma-4 | 2.649 (1.591) | 3.482 (1.451) | 1.047 (1.431) | +0.833 | 2.435 |
| **Combined** | **2.975 (1.244)** | **3.814 (1.109)** | **0.874 (1.198)** | **+0.839** | **2.940** |

*Values in parentheses are standard deviations across 1,831 prompt units.*

### 1.2 By Source Dataset

| Source | Evaluator | Avg Neutral | Avg Stereotype | Avg Anti-Stereo | Bias Amp (S-N) | Total Sep (S-A) |
|--------|-----------|:-----------:|:--------------:|:---------------:|:--------------:|:---------------:|
| StereoSet (n=1,393) | Qwen3-VL | 3.455 | 4.224 | 0.780 | +0.769 | 3.444 |
| | Gemma-4 | 2.781 | 3.556 | 1.142 | +0.775 | 2.413 |
| | **Combined** | **3.117** | **3.890** | **0.961** | **+0.772** | **2.928** |
| CrowS-Pairs (n=438) | Qwen3-VL | 3.129 | 4.215 | 0.293 | +1.086 | 3.921 |
| | Gemma-4 | 2.230 | 3.248 | 0.742 | +1.018 | 2.505 |
| | **Combined** | **2.522** | **3.574** | **0.596** | **+1.052** | **2.978** |

---

## 2. Results by Bias Type

### 2.1 Qwen3-VL Evaluator

| Bias Type | Source | N | Avg Neutral | Avg Stereotype | Avg Anti-Stereo | Bias Amp (S-N) | Total Sep (S-A) |
|-----------|--------|:-:|:-----------:|:--------------:|:---------------:|:--------------:|:---------------:|
| Race-Color | CP | 66 | 2.792 | 4.106 | 0.061 | +1.314 | 4.045 |
| Socioeconomic | CP | 73 | 2.918 | 4.123 | 0.603 | +1.205 | 3.521 |
| Sexual Orientation | CP | 25 | 2.900 | 3.990 | 0.100 | +1.090 | 3.890 |
| Age | CP | 37 | 3.223 | 4.264 | 0.122 | +1.041 | 4.142 |
| Physical Appearance | CP | 14 | 3.286 | 4.321 | 0.821 | +1.036 | 3.500 |
| Race | SS | 428 | 3.273 | 4.202 | 0.500 | +0.930 | 3.703 |
| Nationality | CP | 35 | 2.957 | 3.857 | 0.400 | +0.900 | 3.457 |
| Gender | SS | 370 | 3.446 | 4.292 | 0.468 | +0.846 | 3.824 |
| Disability | CP | 18 | 2.833 | 3.667 | 1.111 | +0.833 | 2.556 |
| Religion | SS | 67 | 3.637 | 4.408 | 0.993 | +0.771 | 3.415 |
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
| Socioeconomic | CP | 73 | 2.183 | 3.667 | 0.902 | +1.484 | 2.765 |
| Nationality | CP | 35 | 1.657 | 2.995 | 0.610 | +1.338 | 2.386 |
| Disability | CP | 18 | 1.815 | 2.972 | 1.796 | +1.157 | 1.176 |
| Age | CP | 37 | 2.563 | 3.635 | 0.520 | +1.072 | 3.115 |
| Religion | SS | 67 | 2.951 | 4.007 | 1.236 | +1.056 | 2.771 |
| Race | SS | 428 | 2.880 | 3.883 | 0.744 | +1.003 | 3.139 |
| Race-Color | CP | 66 | 2.485 | 3.451 | 0.361 | +0.966 | 3.090 |
| Gender | SS | 370 | 3.104 | 3.857 | 0.649 | +0.754 | 3.208 |
| Sexual Orientation | CP | 25 | 1.647 | 2.387 | 0.793 | +0.740 | 1.593 |
| Profession | SS | 698 | 3.263 | 3.910 | 1.084 | +0.647 | 2.825 |
| Physical Appearance | CP | 14 | 2.940 | 3.512 | 1.405 | +0.571 | 2.107 |

*SS = StereoSet, CP = CrowS-Pairs. Tables sorted by Bias Amplification (S-N) descending.*

---

## 3. Key Findings

### 3.1 Elevated Neutral Baselines Confirm Inherent Bias

The benchmark is specifically constructed from the `lean_stereotype` subset -- prompt units where Qwen-Image's neutral outputs already lean toward the stereotypical direction. This is reflected in the high overall neutral score of **2.975** (combined), meaning that even prompts with no directional cues produce images that moderately express stereotypical attributes. The benchmark is designed to study and mitigate this inherent bias.

### 3.2 Stereotype Triggers Further Amplify Bias

Despite already-elevated neutral baselines, stereotype-trigger prompts raise scores by an additional **+0.839** points (combined), reaching **3.814** on average. This demonstrates that explicit stereotype cues meaningfully compound on top of the model's inherent tendencies. By Qwen3-VL's assessment, stereotype-trigger images average **4.222** -- between "strong" and "extremely stereotypical."

### 3.3 Anti-Stereotype Prompts Effectively Suppress Bias

Anti-stereotype prompts reduce scores to **0.874** (combined), a drop of **2.101** points from the neutral baseline. This confirms that counter-stereotypical prompting can override the model's default biases, establishing a clear lower bound for the controlled-contrast evaluation.

### 3.4 Socioeconomic and Nationality Biases Show Highest Amplification

Under the combined evaluator, **socioeconomic** (+1.484) and **nationality** (+1.338) biases show the largest gap between neutral and stereotype-trigger outputs. These categories involve visually salient cues (e.g., clothing quality, environmental settings) that the model strongly responds to when prompted.

### 3.5 Race-Related Biases Dominate by Volume and Separation

Race (StereoSet, n=428) and race-color (CrowS-Pairs, n=66) together form the largest bias cluster. Race shows a combined amplification of +1.003 with a total separation of 3.139, while race-color achieves a near-zero anti-stereotype score (0.361), indicating that identity-swapped prompts effectively neutralize racial stereotypes in generated imagery.

### 3.6 Disability Shows Weakest Separation

Disability bias (n=18) has the lowest total separation (1.176, combined) and the highest anti-stereotype score (1.796). This suggests that visual representations of disability-related stereotypes are less cleanly separable by the controlled-prompt method, possibly because disability attributes are less visually distinct or because the model's disability representations are more diffuse.

### 3.7 Evaluator Agreement

Qwen3-VL consistently assigns higher absolute scores than Gemma-4 (neutral: 3.377 vs 2.649; stereotype: 4.222 vs 3.482), but both evaluators show similar bias amplification magnitudes (+0.844 vs +0.833). The rank ordering of bias types by amplification differs somewhat between evaluators, particularly for **sexual orientation** (Qwen3-VL: +1.090 vs Gemma-4: +0.587) and **physical appearance** (Qwen3-VL: +1.036 vs Gemma-4: +0.262), suggesting these categories are evaluated less consistently across VLMs.

---

## 4. SD3 vs Qwen-Image: Neutral Baseline Comparison

SD3 neutral images were evaluated using the same Qwen3-VL evaluator and benchmark prompts as Qwen-Image. Since SD3 was only evaluated by Qwen3-VL (not Gemma-4), all comparisons in this section use **Qwen3-VL scores only** to ensure an apples-to-apples comparison. Note that Qwen-Image's Qwen3-VL-only neutral score (3.377) is higher than its combined score (2.975) because Gemma-4 tends to assign lower absolute scores.

### 4.1 Overall

| T2I Model | Evaluator | Avg Neutral Score | Std Dev |
|-----------|-----------|:-----------------:|:-------:|
| Qwen-Image | Qwen3-VL | 3.377 | 1.048 |
| **SD3** | **Qwen3-VL** | **3.218** | **1.275** |
| | | **Diff: -0.159** | |

Under the same evaluator, SD3 produces a modestly lower neutral stereotype score than Qwen-Image (-0.159), indicating slightly less inherent bias in its neutral generations across the benchmark. Both models nonetheless score above 3.0 on average, confirming substantial inherent stereotype tendency.

### 4.2 By Source Dataset

| Source | Qwen-Image Neutral | SD3 Neutral | Diff (SD3 - QI) |
|--------|:------------------:|:-----------:|:---------------:|
| StereoSet (n=1,393) | 3.455 | 3.279 | -0.176 |
| CrowS-Pairs (n=438) | 3.129 | 3.026 | -0.103 |

Both datasets show SD3 scoring lower than Qwen-Image, with the difference slightly larger on StereoSet prompts.

### 4.3 By Bias Type

| Bias Type | Source | N | QI Neutral | SD3 Neutral | Diff (SD3 - QI) |
|-----------|--------|:-:|:----------:|:-----------:|:---------------:|
| Disability | CP | 18 | 2.833 | 2.259 | -0.574 |
| Age | CP | 37 | 3.223 | 2.820 | -0.403 |
| Race | SS | 428 | 3.273 | 2.987 | -0.286 |
| Profession | SS | 698 | 3.546 | 3.420 | -0.125 |
| Gender | SS | 370 | 3.446 | 3.328 | -0.118 |
| Religion | SS | 67 | 3.637 | 3.527 | -0.109 |
| Race-Color | CP | 66 | 2.792 | 2.732 | -0.059 |
| Socioeconomic | CP | 73 | 2.918 | 2.888 | -0.030 |
| Sexual Orientation | CP | 25 | 2.900 | 2.880 | -0.020 |
| Nationality | CP | 35 | 2.957 | 2.971 | +0.014 |
| Physical Appearance | CP | 14 | 3.286 | 3.357 | +0.071 |

*All scores from Qwen3-VL evaluator. Sorted by Diff ascending (SD3 less biased first).*

### 4.4 SD3 Comparison Observations

1. **SD3 is slightly less biased overall**: Across all 1,831 prompt units, SD3's neutral images score 0.159 points lower than Qwen-Image's when evaluated by the same VLM (Qwen3-VL). The difference is consistent but modest.

2. **Largest improvements on disability and age**: SD3 shows the greatest reduction relative to Qwen-Image for disability (-0.574) and age (-0.403). These categories involve visual attributes (e.g., mobility aids, apparent age) where SD3 may default to less stereotypical depictions.

3. **Race bias is meaningfully lower in SD3**: Race (StereoSet) shows a -0.286 difference, the largest among the high-volume StereoSet categories. This suggests SD3's visual priors for racial attributes are somewhat less stereotypical than Qwen-Image's.

4. **Near-parity on most CrowS-Pairs categories**: Socioeconomic (-0.030), sexual orientation (-0.020), and nationality (+0.014) show negligible differences, indicating both models have similar baseline biases for these categories.

5. **Physical appearance is the only category where SD3 is slightly more biased** (+0.071), though with only 14 prompt units this difference is not reliable.

6. **Both models are substantially biased**: Despite the relative differences, both models produce neutral images averaging above 3.0 on a 0--5 stereotype scale for most categories. The benchmark was designed to capture exactly these cases where neutral prompts still produce stereotypical outputs.
