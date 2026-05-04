# Human Evaluation Summary (Round 1 + Round 2)

- **Round 1**: 50 cases, 10 raters.
- **Round 2**: 50 cases, 10 raters.
- **Total ratings**: 1000 KG-validity + 3000 image ratings. Raters are anonymous form respondents and are not paired across rounds (the form collects only a Timestamp), so per-rater pooling across rounds is omitted.
- **Round 1 subsample**: 10 of 13 form responses kept (rows [1, 2, 4, 5, 6, 9, 10, 11, 12, 13], 1-indexed; seed 42). Round 2 kept all 10 responses.

Each case asks one KG-validity question (Yes / No / Unsure) and three image-rating questions (0-5) for the stereotype-trigger, neutral, and anti-stereotype-trigger images. Auto-evaluation scores come from Qwen3-VL and Gemma-4, recorded in each round's manifest.

## 1. KG validity -- Is this a real societal stereotype?

| Set | n ratings | % Yes | % Unsure | % No |
| --- | --- | --- | --- | --- |
| Round 1 | 500 | 76.8 | 18.2 | 5.0 |
| Round 2 | 500 | 70.2 | 18.6 | 11.2 |
| Combined | 1000 | 73.5 | 18.4 | 8.1 |

| Set | Cases | All-rater Yes | Majority Yes (>= half) | Unanimous (any label) |
| --- | --- | --- | --- | --- |
| Round 1 | 50 | 7 | 42 | 7 |
| Round 2 | 50 | 10 | 38 | 10 |
| Combined | 100 | 17 | 80 | 17 |

Across both rounds, **73.5%** of KG-validity ratings answered "Yes", **18.4%** "Unsure", and **8.1%** "No".

**KG validity by bias type (combined):**

| bias_type | n | pct_yes | pct_unsure | pct_no |
| --- | --- | --- | --- | --- |
| age | 40 | 55.0 | 22.5 | 22.5 |
| disability | 30 | 56.7 | 13.3 | 30.0 |
| gender | 230 | 79.1 | 15.7 | 5.2 |
| nationality | 20 | 45.0 | 45.0 | 10.0 |
| profession | 280 | 79.6 | 13.9 | 6.4 |
| race | 210 | 67.6 | 27.1 | 5.2 |
| race-color | 60 | 70.0 | 20.0 | 10.0 |
| religion | 60 | 81.7 | 15.0 | 3.3 |
| sexual-orientation | 10 | 100.0 | 0.0 | 0.0 |
| socioeconomic | 60 | 65.0 | 15.0 | 20.0 |

## 2. Image ratings (0-5) by condition

| Set | Condition | Mean | Std | n |
| --- | --- | --- | --- | --- |
| Round 1 | stereotype_trigger | 3.858 | 1.292 | 500 |
| Round 1 | neutral | 3.192 | 1.596 | 500 |
| Round 1 | anti_stereotype_trigger | 1.902 | 1.635 | 500 |
| Round 2 | stereotype_trigger | 3.758 | 1.352 | 500 |
| Round 2 | neutral | 3.294 | 1.545 | 500 |
| Round 2 | anti_stereotype_trigger | 1.442 | 1.564 | 500 |
| Combined | stereotype_trigger | 3.808 | 1.322 | 1000 |
| Combined | neutral | 3.243 | 1.571 | 1000 |
| Combined | anti_stereotype_trigger | 1.672 | 1.616 | 1000 |

**Bias amplification and total separation:**

| Set | Bias amplification (S - N) | Total separation (S - A) |
| --- | --- | --- |
| Round 1 | 0.666 | 1.956 |
| Round 2 | 0.464 | 2.316 |
| Combined | 0.565 | 2.136 |

**Mean human rating by bias type x condition (combined):**

| bias_type | stereotype_trigger | neutral | anti_stereotype_trigger |
| --- | --- | --- | --- |
| age | 3.68 | 2.22 | 1.58 |
| disability | 2.7 | 2.0 | 0.87 |
| gender | 4.13 | 3.6 | 1.36 |
| nationality | 2.8 | 1.95 | 2.7 |
| profession | 3.74 | 3.36 | 1.62 |
| race | 3.79 | 3.31 | 1.8 |
| race-color | 4.33 | 2.95 | 2.12 |
| religion | 4.03 | 3.77 | 2.55 |
| sexual-orientation | 4.0 | 4.1 | 0.9 |
| socioeconomic | 3.17 | 2.47 | 1.58 |

## 3. Inter-rater reliability (image ratings)

| Set | Raters | Images | Pairwise Pearson r (mean) | Pairwise weighted kappa (mean) | ICC(2,1) | ICC(2,k) |
| --- | --- | --- | --- | --- | --- | --- |
| Round 1 | 10 | 150 | 0.568 | 0.495 | 0.516 | 0.914 |
| Round 1 (excl. outliers) | 9 | 150 | 0.646 | 0.561 | 0.570 | 0.923 |
| Round 2 | 10 | 150 | 0.604 | 0.546 | 0.561 | 0.927 |
| Round 2 (excl. outliers) | 9 | 150 | 0.681 | 0.620 | 0.624 | 0.937 |

Outliers are raters whose mean pairwise Pearson r with the other raters in their round falls below 0.40. Round 1: dropped `r1_rater_8` (mean pairwise r = 0.259). Round 2: dropped `r2_rater_9` (mean pairwise r = 0.294).

Each round's raters are independent. R1 and R2 are reported separately; per-rater pooling across rounds is omitted because the two forms collect no per-rater identifier.

## 4. Human vs auto-evaluation (Qwen3-VL, Gemma-4)

| Set | Images | Pearson r (Qwen3-VL) | Spearman rho (Qwen3-VL) | MAE (Qwen3-VL) | Pearson r (Gemma-4) | Spearman rho (Gemma-4) | MAE (Gemma-4) | Cases S>A human/qwen/gemma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 1 | 150 | 0.751 | 0.784 | 1.006 | 0.812 | 0.781 | 0.951 | 46/47/40 of 50 |
| Round 2 | 150 | 0.779 | 0.779 | 0.940 | 0.736 | 0.740 | 1.125 | 44/46/39 of 50 |
| Combined | 300 | 0.765 | 0.779 | 0.973 | 0.772 | 0.762 | 1.038 | 90/93/79 of 100 |

**Per-rater Pearson r with VLMs (per round, raters not paired across rounds):**

| Round | Rater | Images rated | Pearson r vs Qwen3-VL | Pearson r vs Gemma-4 |
| --- | --- | --- | --- | --- |
| Round 1 | r1_rater_1 | 150 | 0.682 | 0.781 |
| Round 1 | r1_rater_10 | 150 | 0.563 | 0.608 |
| Round 1 | r1_rater_2 | 150 | 0.622 | 0.689 |
| Round 1 | r1_rater_3 | 150 | 0.619 | 0.690 |
| Round 1 | r1_rater_4 | 150 | 0.594 | 0.681 |
| Round 1 | r1_rater_5 | 150 | 0.690 | 0.729 |
| Round 1 | r1_rater_6 | 150 | 0.721 | 0.708 |
| Round 1 | r1_rater_7 | 150 | 0.606 | 0.599 |
| Round 1 | r1_rater_8 | 150 | 0.331 | 0.188 |
| Round 1 | r1_rater_9 | 150 | 0.419 | 0.571 |
| Round 2 | r2_rater_1 | 150 | 0.672 | 0.688 |
| Round 2 | r2_rater_10 | 150 | 0.572 | 0.537 |
| Round 2 | r2_rater_2 | 150 | 0.719 | 0.640 |
| Round 2 | r2_rater_3 | 150 | 0.641 | 0.546 |
| Round 2 | r2_rater_4 | 150 | 0.578 | 0.582 |
| Round 2 | r2_rater_5 | 150 | 0.611 | 0.524 |
| Round 2 | r2_rater_6 | 150 | 0.707 | 0.720 |
| Round 2 | r2_rater_7 | 150 | 0.712 | 0.719 |
| Round 2 | r2_rater_8 | 150 | 0.709 | 0.688 |
| Round 2 | r2_rater_9 | 150 | 0.305 | 0.224 |

## 4b. Overall agreement (humans + Qwen3-VL + Gemma-4, per round)

Treats each round's human raters together with Qwen3-VL and Gemma-4 as a single rater pool over that round's 150 images (50 cases x 3 conditions). Rounds are reported separately because human raters are not paired across rounds.

| Set | Annotators | Images | Mean pairwise Pearson r | Mean pairwise Spearman rho | Mean pairwise weighted kappa | ICC(2,1) | ICC(2,k) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Round 1 | 10 humans + 2 VLMs | 150 | 0.582 | 0.583 | 0.515 | 0.539 | 0.933 |
| Round 2 | 10 humans + 2 VLMs | 150 | 0.606 | 0.610 | 0.552 | 0.567 | 0.940 |

**Full pairwise Pearson r matrix -- Round 1:**

|  | r1_rater_1 | r1_rater_10 | r1_rater_2 | r1_rater_3 | r1_rater_4 | r1_rater_5 | r1_rater_6 | r1_rater_7 | r1_rater_8 | r1_rater_9 | qwen3vl | gemma4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r1_rater_1 | 1.0 | 0.592 | 0.766 | 0.671 | 0.757 | 0.758 | 0.774 | 0.669 | 0.274 | 0.58 | 0.682 | 0.781 |
| r1_rater_10 | 0.592 | 1.0 | 0.607 | 0.505 | 0.669 | 0.651 | 0.579 | 0.593 | 0.271 | 0.474 | 0.563 | 0.608 |
| r1_rater_2 | 0.766 | 0.607 | 1.0 | 0.655 | 0.768 | 0.7 | 0.726 | 0.672 | 0.309 | 0.57 | 0.622 | 0.689 |
| r1_rater_3 | 0.671 | 0.505 | 0.655 | 1.0 | 0.596 | 0.67 | 0.723 | 0.594 | 0.123 | 0.557 | 0.619 | 0.69 |
| r1_rater_4 | 0.757 | 0.669 | 0.768 | 0.596 | 1.0 | 0.715 | 0.67 | 0.586 | 0.263 | 0.517 | 0.594 | 0.681 |
| r1_rater_5 | 0.758 | 0.651 | 0.7 | 0.67 | 0.715 | 1.0 | 0.768 | 0.676 | 0.275 | 0.625 | 0.69 | 0.729 |
| r1_rater_6 | 0.774 | 0.579 | 0.726 | 0.723 | 0.67 | 0.768 | 1.0 | 0.693 | 0.25 | 0.55 | 0.721 | 0.708 |
| r1_rater_7 | 0.669 | 0.593 | 0.672 | 0.594 | 0.586 | 0.676 | 0.693 | 1.0 | 0.3 | 0.57 | 0.606 | 0.599 |
| r1_rater_8 | 0.274 | 0.271 | 0.309 | 0.123 | 0.263 | 0.275 | 0.25 | 0.3 | 1.0 | 0.264 | 0.331 | 0.188 |
| r1_rater_9 | 0.58 | 0.474 | 0.57 | 0.557 | 0.517 | 0.625 | 0.55 | 0.57 | 0.264 | 1.0 | 0.419 | 0.571 |
| qwen3vl | 0.682 | 0.563 | 0.622 | 0.619 | 0.594 | 0.69 | 0.721 | 0.606 | 0.331 | 0.419 | 1.0 | 0.723 |
| gemma4 | 0.781 | 0.608 | 0.689 | 0.69 | 0.681 | 0.729 | 0.708 | 0.599 | 0.188 | 0.571 | 0.723 | 1.0 |

**Full pairwise Pearson r matrix -- Round 2:**

|  | r2_rater_1 | r2_rater_10 | r2_rater_2 | r2_rater_3 | r2_rater_4 | r2_rater_5 | r2_rater_6 | r2_rater_7 | r2_rater_8 | r2_rater_9 | qwen3vl | gemma4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r2_rater_1 | 1.0 | 0.632 | 0.755 | 0.686 | 0.697 | 0.73 | 0.759 | 0.822 | 0.737 | 0.34 | 0.672 | 0.688 |
| r2_rater_10 | 0.632 | 1.0 | 0.66 | 0.594 | 0.503 | 0.669 | 0.619 | 0.675 | 0.627 | 0.21 | 0.572 | 0.537 |
| r2_rater_2 | 0.755 | 0.66 | 1.0 | 0.699 | 0.676 | 0.695 | 0.713 | 0.767 | 0.754 | 0.382 | 0.719 | 0.64 |
| r2_rater_3 | 0.686 | 0.594 | 0.699 | 1.0 | 0.482 | 0.635 | 0.634 | 0.701 | 0.644 | 0.301 | 0.641 | 0.546 |
| r2_rater_4 | 0.697 | 0.503 | 0.676 | 0.482 | 1.0 | 0.589 | 0.677 | 0.694 | 0.672 | 0.221 | 0.578 | 0.582 |
| r2_rater_5 | 0.73 | 0.669 | 0.695 | 0.635 | 0.589 | 1.0 | 0.615 | 0.676 | 0.558 | 0.344 | 0.611 | 0.524 |
| r2_rater_6 | 0.759 | 0.619 | 0.713 | 0.634 | 0.677 | 0.615 | 1.0 | 0.838 | 0.809 | 0.264 | 0.707 | 0.72 |
| r2_rater_7 | 0.822 | 0.675 | 0.767 | 0.701 | 0.694 | 0.676 | 0.838 | 1.0 | 0.817 | 0.29 | 0.712 | 0.719 |
| r2_rater_8 | 0.737 | 0.627 | 0.754 | 0.644 | 0.672 | 0.558 | 0.809 | 0.817 | 1.0 | 0.299 | 0.709 | 0.688 |
| r2_rater_9 | 0.34 | 0.21 | 0.382 | 0.301 | 0.221 | 0.344 | 0.264 | 0.29 | 0.299 | 1.0 | 0.305 | 0.224 |
| qwen3vl | 0.672 | 0.572 | 0.719 | 0.641 | 0.578 | 0.611 | 0.707 | 0.712 | 0.709 | 0.305 | 1.0 | 0.749 |
| gemma4 | 0.688 | 0.537 | 0.64 | 0.546 | 0.582 | 0.524 | 0.72 | 0.719 | 0.688 | 0.224 | 0.749 | 1.0 |

## 4c. Overall Pearson summary (averaged across rounds)

Single-row view of every Pearson-related metric, averaged across Round 1 and Round 2. Inter-rater stats are the mean of each round's within-round pairwise mean (raters are not paired across rounds, so they cannot be pooled directly). Human-vs-VLM Pearson is computed on per-image human means and pooled across both rounds (300 images). The "excl. outliers" row drops, in each round, any rater whose mean pairwise Pearson with the other raters is below 0.40.

| Set | Raters (R1, R2) | Inter-rater Pearson r | Inter-rater kappa | Inter-rater ICC(2,1) | Inter-rater ICC(2,k) | Human vs Qwen3-VL (Pearson r, 300 imgs) | Human vs Gemma-4 (Pearson r, 300 imgs) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All raters | 10, 10 | 0.586 | 0.520 | 0.539 | 0.921 | 0.765 | 0.772 |
| Excl. outliers | 9, 9 | 0.663 | 0.590 | 0.597 | 0.930 | 0.761 | 0.780 |

## 5. Caveats

- 10 raters in Round 1 and 10 in Round 2; per-bias-type breakdowns still have wide CIs and should be read as directional.
- Round 1 had 13 form responses; 10 were randomly retained (seed 42, kept rows [1, 2, 4, 5, 6, 9, 10, 11, 12, 13]) so both rounds contribute the same number of raters.
- Round 1 and Round 2 use disjoint case samples drawn from the same lean-stereotype pool. Raters are not paired across rounds -- the form collects no per-rater identifier -- so per-rater analyses are reported per round only.
- Only seed 1 was rated for each case.
- VLM scores are stored in each round's manifest (`vlm_qwen_score`, `vlm_gemma_score`) and were generated by Qwen3-VL-30B and Gemma-4 respectively.
