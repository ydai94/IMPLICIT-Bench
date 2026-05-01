# Human Evaluation Summary (Round 1 + Round 2)

- **Round 1**: 50 cases.
- **Round 2**: 50 cases.
- **Raters**: 3 (the same annotators completed both rounds, paired by submission order); each rater scored 100 unique cases.

Each case asks one KG-validity question (Yes / No / Unsure) and three image-rating questions (0-5) for the stereotype-trigger, neutral, and anti-stereotype-trigger images. Auto-evaluation scores come from Qwen3-VL and Gemma-4, recorded in each round's manifest.

## 1. KG validity -- Is this a real societal stereotype?

| Set | n ratings | % Yes | % Unsure | % No |
| --- | --- | --- | --- | --- |
| Round 1 | 150 | 89.3 | 10.0 | 0.7 |
| Round 2 | 150 | 84.7 | 8.0 | 7.3 |
| Combined | 300 | 87.0 | 9.0 | 4.0 |

| Set | Cases | All-rater Yes | Majority Yes (>= half) | Unanimous (any label) |
| --- | --- | --- | --- | --- |
| Round 1 | 50 | 41 | 45 | 43 |
| Round 2 | 50 | 37 | 44 | 39 |
| Combined | 100 | 78 | 89 | 82 |

Across both rounds, **87.0%** of KG-validity ratings answered "Yes", **9.0%** "Unsure", and **4.0%** "No".

**KG validity by bias type (combined):**

| bias_type | n | pct_yes | pct_unsure | pct_no |
| --- | --- | --- | --- | --- |
| age | 12 | 66.7 | 25.0 | 8.3 |
| disability | 9 | 55.6 | 11.1 | 33.3 |
| gender | 69 | 94.2 | 4.3 | 1.4 |
| nationality | 6 | 33.3 | 66.7 | 0.0 |
| profession | 84 | 98.8 | 1.2 | 0.0 |
| race | 63 | 82.5 | 15.9 | 1.6 |
| race-color | 18 | 83.3 | 5.6 | 11.1 |
| religion | 18 | 94.4 | 5.6 | 0.0 |
| sexual-orientation | 3 | 100.0 | 0.0 | 0.0 |
| socioeconomic | 18 | 61.1 | 16.7 | 22.2 |

## 2. Image ratings (0-5) by condition

| Set | Condition | Mean | Std | n |
| --- | --- | --- | --- | --- |
| Round 1 | stereotype_trigger | 4.333 | 0.994 | 150 |
| Round 1 | neutral | 3.327 | 1.755 | 150 |
| Round 1 | anti_stereotype_trigger | 1.700 | 1.824 | 150 |
| Round 2 | stereotype_trigger | 3.893 | 1.489 | 150 |
| Round 2 | neutral | 3.540 | 1.689 | 150 |
| Round 2 | anti_stereotype_trigger | 1.080 | 1.544 | 150 |
| Combined | stereotype_trigger | 4.113 | 1.283 | 300 |
| Combined | neutral | 3.433 | 1.723 | 300 |
| Combined | anti_stereotype_trigger | 1.390 | 1.715 | 300 |

**Bias amplification and total separation:**

| Set | Bias amplification (S - N) | Total separation (S - A) |
| --- | --- | --- |
| Round 1 | 1.007 | 2.633 |
| Round 2 | 0.353 | 2.813 |
| Combined | 0.680 | 2.723 |

**Mean human rating by bias type x condition (combined):**

| bias_type | stereotype_trigger | neutral | anti_stereotype_trigger |
| --- | --- | --- | --- |
| age | 4.25 | 2.0 | 1.25 |
| disability | 2.33 | 1.44 | 0.22 |
| gender | 4.48 | 3.96 | 1.14 |
| nationality | 3.5 | 1.17 | 2.5 |
| profession | 4.0 | 3.58 | 1.29 |
| race | 4.05 | 3.52 | 1.56 |
| race-color | 4.61 | 2.83 | 1.61 |
| religion | 4.67 | 4.0 | 2.56 |
| sexual-orientation | 4.67 | 4.67 | 0.33 |
| socioeconomic | 3.33 | 2.94 | 1.33 |

## 3. Inter-rater reliability (image ratings)

| Set | Raters | Images | Pairwise Pearson r (mean) | Pairwise weighted kappa (mean) | ICC(2,1) | ICC(2,k) |
| --- | --- | --- | --- | --- | --- | --- |
| Round 1 | 3 | 150 | 0.785 | 0.739 | 0.742 | 0.896 |
| Round 2 | 3 | 150 | 0.713 | 0.663 | 0.666 | 0.857 |
| Combined | 3 | 300 | 0.707 | 0.701 | 0.702 | 0.876 |

The same 3 raters scored both rounds (paired by submission order), so the Combined row pools 300 images per rater (100 cases x 3 conditions).

## 4. Human vs auto-evaluation (Qwen3-VL, Gemma-4)

| Set | Images | Pearson r (Qwen3-VL) | Spearman rho (Qwen3-VL) | MAE (Qwen3-VL) | Pearson r (Gemma-4) | Spearman rho (Gemma-4) | MAE (Gemma-4) | Cases S>A human/qwen/gemma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 1 | 150 | 0.727 | 0.765 | 0.957 | 0.803 | 0.769 | 0.913 | 43/47/40 of 50 |
| Round 2 | 150 | 0.751 | 0.747 | 0.910 | 0.682 | 0.708 | 1.177 | 39/46/39 of 50 |
| Combined | 300 | 0.738 | 0.753 | 0.933 | 0.744 | 0.746 | 1.045 | 82/93/79 of 100 |

**Per-rater Pearson r with VLMs (combined across rounds):**

| Rater | Images rated | Pearson r vs Qwen3-VL | Pearson r vs Gemma-4 |
| --- | --- | --- | --- |
| rater_1 | 300 | 0.662 | 0.733 |
| rater_2 | 300 | 0.669 | 0.668 |
| rater_3 | 300 | 0.656 | 0.606 |

## 4b. Overall agreement (3 humans + Qwen3-VL + Gemma-4)

Treats all 5 annotators as raters of the same 300 images and reports pooled inter-annotator reliability.

| Annotators | Images | Mean pairwise Pearson r | Mean pairwise Spearman rho | Mean pairwise weighted kappa | ICC(2,1) | ICC(2,k) |
| --- | --- | --- | --- | --- | --- | --- |
| 3 humans + 2 VLMs | 300 | 0.685 | 0.653 | 0.670 | 0.671 | 0.911 |

**Full pairwise Pearson r matrix:**

|  | rater_1 | rater_2 | rater_3 | qwen3vl | gemma4 |
| --- | --- | --- | --- | --- | --- |
| rater_1 | 1.0 | 0.747 | 0.663 | 0.662 | 0.733 |
| rater_2 | 0.747 | 1.0 | 0.711 | 0.669 | 0.668 |
| rater_3 | 0.663 | 0.711 | 1.0 | 0.656 | 0.606 |
| qwen3vl | 0.662 | 0.669 | 0.656 | 1.0 | 0.733 |
| gemma4 | 0.733 | 0.668 | 0.606 | 0.733 | 1.0 |

## 5. Caveats

- Only 3 raters total, so per-bias-type breakdowns have wide CIs and should be read as directional.
- Round 1 and Round 2 use disjoint case samples drawn from the same lean-stereotype pool, with the same 3 raters scoring both rounds (paired by submission order).
- Only seed 1 was rated for each case.
- VLM scores are stored in each round's manifest (`vlm_qwen_score`, `vlm_gemma_score`) and were generated by Qwen3-VL-30B and Gemma-4 respectively.
