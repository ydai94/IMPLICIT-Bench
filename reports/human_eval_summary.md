# Human Evaluation Summary (Round 1 + Round 2)

- **Round 1**: 50 cases.
- **Round 2**: 50 cases.
- **Raters**: 5 (the same annotators completed both rounds, paired by submission order); each rater scored 100 unique cases.

Each case asks one KG-validity question (Yes / No / Unsure) and three image-rating questions (0-5) for the stereotype-trigger, neutral, and anti-stereotype-trigger images. Auto-evaluation scores come from Qwen3-VL and Gemma-4, recorded in each round's manifest.

## 1. KG validity -- Is this a real societal stereotype?

| Set | n ratings | % Yes | % Unsure | % No |
| --- | --- | --- | --- | --- |
| Round 1 | 250 | 83.6 | 12.4 | 4.0 |
| Round 2 | 250 | 79.6 | 9.6 | 10.8 |
| Combined | 500 | 81.6 | 11.0 | 7.4 |

| Set | Cases | All-rater Yes | Majority Yes (>= half) | Unanimous (any label) |
| --- | --- | --- | --- | --- |
| Round 1 | 50 | 22 | 46 | 22 |
| Round 2 | 50 | 26 | 42 | 26 |
| Combined | 100 | 48 | 88 | 48 |

Across both rounds, **81.6%** of KG-validity ratings answered "Yes", **11.0%** "Unsure", and **7.4%** "No".

**KG validity by bias type (combined):**

| bias_type | n | pct_yes | pct_unsure | pct_no |
| --- | --- | --- | --- | --- |
| age | 20 | 60.0 | 20.0 | 20.0 |
| disability | 15 | 60.0 | 6.7 | 33.3 |
| gender | 115 | 91.3 | 6.1 | 2.6 |
| nationality | 10 | 40.0 | 50.0 | 10.0 |
| profession | 140 | 87.1 | 5.7 | 7.1 |
| race | 105 | 79.0 | 18.1 | 2.9 |
| race-color | 30 | 66.7 | 16.7 | 16.7 |
| religion | 30 | 93.3 | 6.7 | 0.0 |
| sexual-orientation | 5 | 100.0 | 0.0 | 0.0 |
| socioeconomic | 30 | 66.7 | 13.3 | 20.0 |

## 2. Image ratings (0-5) by condition

| Set | Condition | Mean | Std | n |
| --- | --- | --- | --- | --- |
| Round 1 | stereotype_trigger | 4.028 | 1.351 | 250 |
| Round 1 | neutral | 3.116 | 1.828 | 250 |
| Round 1 | anti_stereotype_trigger | 1.568 | 1.769 | 250 |
| Round 2 | stereotype_trigger | 3.744 | 1.552 | 250 |
| Round 2 | neutral | 3.356 | 1.711 | 250 |
| Round 2 | anti_stereotype_trigger | 1.184 | 1.570 | 250 |
| Combined | stereotype_trigger | 3.886 | 1.461 | 500 |
| Combined | neutral | 3.236 | 1.773 | 500 |
| Combined | anti_stereotype_trigger | 1.376 | 1.682 | 500 |

**Bias amplification and total separation:**

| Set | Bias amplification (S - N) | Total separation (S - A) |
| --- | --- | --- |
| Round 1 | 0.912 | 2.460 |
| Round 2 | 0.388 | 2.560 |
| Combined | 0.650 | 2.510 |

**Mean human rating by bias type x condition (combined):**

| bias_type | stereotype_trigger | neutral | anti_stereotype_trigger |
| --- | --- | --- | --- |
| age | 3.85 | 2.2 | 1.5 |
| disability | 2.2 | 1.67 | 0.53 |
| gender | 4.37 | 3.76 | 1.12 |
| nationality | 2.6 | 1.2 | 2.4 |
| profession | 3.72 | 3.32 | 1.22 |
| race | 3.92 | 3.33 | 1.52 |
| race-color | 4.23 | 2.6 | 1.63 |
| religion | 4.27 | 3.77 | 2.4 |
| sexual-orientation | 4.6 | 4.6 | 0.4 |
| socioeconomic | 3.1 | 2.53 | 1.43 |

## 3. Inter-rater reliability (image ratings)

| Set | Raters | Images | Pairwise Pearson r (mean) | Pairwise weighted kappa (mean) | ICC(2,1) | ICC(2,k) |
| --- | --- | --- | --- | --- | --- | --- |
| Round 1 | 5 | 150 | 0.720 | 0.659 | 0.650 | 0.903 |
| Round 2 | 5 | 150 | 0.664 | 0.615 | 0.608 | 0.886 |
| Combined | 5 | 300 | 0.667 | 0.633 | 0.628 | 0.894 |

The same 5 raters scored both rounds (paired by submission order), so the Combined row pools 300 images per rater (100 cases x 3 conditions).

## 4. Human vs auto-evaluation (Qwen3-VL, Gemma-4)

| Set | Images | Pearson r (Qwen3-VL) | Spearman rho (Qwen3-VL) | MAE (Qwen3-VL) | Pearson r (Gemma-4) | Spearman rho (Gemma-4) | MAE (Gemma-4) | Cases S>A human/qwen/gemma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 1 | 150 | 0.736 | 0.774 | 0.930 | 0.819 | 0.782 | 0.849 | 43/47/40 of 50 |
| Round 2 | 150 | 0.755 | 0.748 | 0.914 | 0.696 | 0.713 | 1.110 | 40/46/39 of 50 |
| Combined | 300 | 0.746 | 0.759 | 0.922 | 0.758 | 0.752 | 0.980 | 83/93/79 of 100 |

**Per-rater Pearson r with VLMs (combined across rounds):**

| Rater | Images rated | Pearson r vs Qwen3-VL | Pearson r vs Gemma-4 |
| --- | --- | --- | --- |
| rater_1 | 300 | 0.662 | 0.733 |
| rater_2 | 300 | 0.669 | 0.668 |
| rater_3 | 300 | 0.656 | 0.606 |
| rater_4 | 300 | 0.598 | 0.629 |
| rater_5 | 300 | 0.603 | 0.600 |

## 4b. Overall agreement (5 humans + Qwen3-VL + Gemma-4)

Treats all 7 annotators as raters of the same 300 images and reports pooled inter-annotator reliability.

| Annotators | Images | Mean pairwise Pearson r | Mean pairwise Spearman rho | Mean pairwise weighted kappa | ICC(2,1) | ICC(2,k) |
| --- | --- | --- | --- | --- | --- | --- |
| 5 humans + 2 VLMs | 300 | 0.659 | 0.634 | 0.630 | 0.629 | 0.922 |

**Full pairwise Pearson r matrix:**

|  | rater_1 | rater_2 | rater_3 | rater_4 | rater_5 | qwen3vl | gemma4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rater_1 | 1.0 | 0.747 | 0.663 | 0.661 | 0.729 | 0.662 | 0.733 |
| rater_2 | 0.747 | 1.0 | 0.711 | 0.656 | 0.726 | 0.669 | 0.668 |
| rater_3 | 0.663 | 0.711 | 1.0 | 0.546 | 0.642 | 0.656 | 0.606 |
| rater_4 | 0.661 | 0.656 | 0.546 | 1.0 | 0.592 | 0.598 | 0.629 |
| rater_5 | 0.729 | 0.726 | 0.642 | 0.592 | 1.0 | 0.603 | 0.6 |
| qwen3vl | 0.662 | 0.669 | 0.656 | 0.598 | 0.603 | 1.0 | 0.733 |
| gemma4 | 0.733 | 0.668 | 0.606 | 0.629 | 0.6 | 0.733 | 1.0 |

## 5. Caveats

- Only 5 raters total, so per-bias-type breakdowns have wide CIs and should be read as directional.
- Round 1 and Round 2 use disjoint case samples drawn from the same lean-stereotype pool, with the same 5 raters scoring both rounds (paired by submission order).
- Only seed 1 was rated for each case.
- VLM scores are stored in each round's manifest (`vlm_qwen_score`, `vlm_gemma_score`) and were generated by Qwen3-VL-30B and Gemma-4 respectively.
