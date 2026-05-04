# Human Evaluation Summary (Round 1 + Round 2)

- **Round 1**: 50 cases, 12 raters.
- **Round 2**: 50 cases, 12 raters.
- **Total ratings**: 1200 KG-validity + 3600 image ratings. Raters are anonymous form respondents and are not paired across rounds (the form collects only a Timestamp), so per-rater pooling across rounds is omitted.

Each case asks one KG-validity question (Yes / No / Unsure) and three image-rating questions (0-5) for the stereotype-trigger, neutral, and anti-stereotype-trigger images. Auto-evaluation scores come from Qwen3-VL and Gemma-4, recorded in each round's manifest.

## 1. KG validity -- Is this a real societal stereotype?

| Set | n ratings | % Yes | % Unsure | % No |
| --- | --- | --- | --- | --- |
| Round 1 | 600 | 80.5 | 15.5 | 4.0 |
| Round 2 | 600 | 71.3 | 18.2 | 10.5 |
| Combined | 1200 | 75.9 | 16.8 | 7.2 |

| Set | Cases | All-rater Yes | Majority Yes (>= half) | Unanimous (any label) |
| --- | --- | --- | --- | --- |
| Round 1 | 50 | 8 | 44 | 8 |
| Round 2 | 50 | 13 | 38 | 13 |
| Combined | 100 | 21 | 82 | 21 |

Across both rounds, **75.9%** of KG-validity ratings answered "Yes", **16.8%** "Unsure", and **7.2%** "No".

**KG validity by bias type (combined):**

| bias_type | n | pct_yes | pct_unsure | pct_no |
| --- | --- | --- | --- | --- |
| age | 48 | 47.9 | 22.9 | 29.2 |
| disability | 36 | 55.6 | 16.7 | 27.8 |
| gender | 276 | 80.8 | 14.5 | 4.7 |
| nationality | 24 | 50.0 | 41.7 | 8.3 |
| profession | 336 | 83.3 | 11.6 | 5.1 |
| race | 252 | 70.6 | 25.4 | 4.0 |
| race-color | 72 | 70.8 | 20.8 | 8.3 |
| religion | 72 | 88.9 | 9.7 | 1.4 |
| sexual-orientation | 12 | 100.0 | 0.0 | 0.0 |
| socioeconomic | 72 | 66.7 | 13.9 | 19.4 |

## 2. Image ratings (0-5) by condition

| Set | Condition | Mean | Std | n |
| --- | --- | --- | --- | --- |
| Round 1 | stereotype_trigger | 3.868 | 1.339 | 600 |
| Round 1 | neutral | 3.135 | 1.649 | 600 |
| Round 1 | anti_stereotype_trigger | 1.777 | 1.668 | 600 |
| Round 2 | stereotype_trigger | 3.683 | 1.392 | 600 |
| Round 2 | neutral | 3.198 | 1.574 | 600 |
| Round 2 | anti_stereotype_trigger | 1.422 | 1.530 | 600 |
| Combined | stereotype_trigger | 3.776 | 1.368 | 1200 |
| Combined | neutral | 3.167 | 1.611 | 1200 |
| Combined | anti_stereotype_trigger | 1.599 | 1.610 | 1200 |

**Bias amplification and total separation:**

| Set | Bias amplification (S - N) | Total separation (S - A) |
| --- | --- | --- |
| Round 1 | 0.733 | 2.092 |
| Round 2 | 0.485 | 2.262 |
| Combined | 0.609 | 2.177 |

**Mean human rating by bias type x condition (combined):**

| bias_type | stereotype_trigger | neutral | anti_stereotype_trigger |
| --- | --- | --- | --- |
| age | 3.69 | 2.04 | 1.48 |
| disability | 2.75 | 2.11 | 0.89 |
| gender | 4.13 | 3.55 | 1.27 |
| nationality | 2.67 | 1.62 | 2.33 |
| profession | 3.7 | 3.28 | 1.56 |
| race | 3.77 | 3.23 | 1.75 |
| race-color | 4.25 | 2.85 | 2.04 |
| religion | 3.93 | 3.65 | 2.38 |
| sexual-orientation | 4.17 | 4.17 | 0.75 |
| socioeconomic | 3.06 | 2.42 | 1.65 |

## 3. Inter-rater reliability (image ratings)

| Set | Raters | Images | Pairwise Pearson r (mean) | Pairwise weighted kappa (mean) | ICC(2,1) | ICC(2,k) |
| --- | --- | --- | --- | --- | --- | --- |
| Round 1 | 12 | 150 | 0.641 | 0.571 | 0.578 | 0.943 |
| Round 2 | 12 | 150 | 0.652 | 0.590 | 0.593 | 0.946 |

Each round's raters are independent. R1 and R2 are reported separately; per-rater pooling across rounds is omitted because the two forms collect no per-rater identifier.

## 4. Human vs auto-evaluation (Qwen3-VL, Gemma-4)

| Set | Images | Pearson r (Qwen3-VL) | Spearman rho (Qwen3-VL) | MAE (Qwen3-VL) | Pearson r (Gemma-4) | Spearman rho (Gemma-4) | MAE (Gemma-4) | Cases S>A human/qwen/gemma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 1 | 150 | 0.744 | 0.773 | 0.984 | 0.822 | 0.791 | 0.888 | 47/47/40 of 50 |
| Round 2 | 150 | 0.772 | 0.767 | 0.953 | 0.741 | 0.748 | 1.089 | 42/46/39 of 50 |
| Combined | 300 | 0.758 | 0.768 | 0.969 | 0.781 | 0.776 | 0.988 | 89/93/79 of 100 |

**Per-rater Pearson r with VLMs (per round, raters not paired across rounds):**

| Round | Rater | Images rated | Pearson r vs Qwen3-VL | Pearson r vs Gemma-4 |
| --- | --- | --- | --- | --- |
| Round 1 | r1_rater_1 | 150 | 0.682 | 0.781 |
| Round 1 | r1_rater_10 | 150 | 0.606 | 0.599 |
| Round 1 | r1_rater_12 | 150 | 0.419 | 0.571 |
| Round 1 | r1_rater_13 | 150 | 0.563 | 0.608 |
| Round 1 | r1_rater_2 | 150 | 0.622 | 0.689 |
| Round 1 | r1_rater_3 | 150 | 0.732 | 0.763 |
| Round 1 | r1_rater_4 | 150 | 0.619 | 0.690 |
| Round 1 | r1_rater_5 | 150 | 0.594 | 0.681 |
| Round 1 | r1_rater_6 | 150 | 0.690 | 0.729 |
| Round 1 | r1_rater_7 | 150 | 0.582 | 0.669 |
| Round 1 | r1_rater_8 | 150 | 0.487 | 0.557 |
| Round 1 | r1_rater_9 | 150 | 0.721 | 0.708 |
| Round 2 | r2_rater_1 | 150 | 0.672 | 0.688 |
| Round 2 | r2_rater_10 | 150 | 0.572 | 0.537 |
| Round 2 | r2_rater_11 | 150 | 0.533 | 0.549 |
| Round 2 | r2_rater_12 | 150 | 0.554 | 0.489 |
| Round 2 | r2_rater_13 | 150 | 0.606 | 0.656 |
| Round 2 | r2_rater_2 | 150 | 0.719 | 0.640 |
| Round 2 | r2_rater_3 | 150 | 0.641 | 0.546 |
| Round 2 | r2_rater_4 | 150 | 0.578 | 0.582 |
| Round 2 | r2_rater_5 | 150 | 0.611 | 0.524 |
| Round 2 | r2_rater_6 | 150 | 0.707 | 0.720 |
| Round 2 | r2_rater_7 | 150 | 0.712 | 0.719 |
| Round 2 | r2_rater_8 | 150 | 0.709 | 0.688 |

## 4b. Overall agreement (humans + Qwen3-VL + Gemma-4, per round)

Treats each round's human raters together with Qwen3-VL and Gemma-4 as a single rater pool over that round's 150 images (50 cases x 3 conditions). Rounds are reported separately because human raters are not paired across rounds.

| Set | Annotators | Images | Mean pairwise Pearson r | Mean pairwise Spearman rho | Mean pairwise weighted kappa | ICC(2,1) | ICC(2,k) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Round 1 | 12 humans + 2 VLMs | 150 | 0.642 | 0.637 | 0.579 | 0.587 | 0.952 |
| Round 2 | 12 humans + 2 VLMs | 150 | 0.645 | 0.651 | 0.587 | 0.591 | 0.953 |

**Full pairwise Pearson r matrix -- Round 1:**

|  | r1_rater_1 | r1_rater_10 | r1_rater_12 | r1_rater_13 | r1_rater_2 | r1_rater_3 | r1_rater_4 | r1_rater_5 | r1_rater_6 | r1_rater_7 | r1_rater_8 | r1_rater_9 | qwen3vl | gemma4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r1_rater_1 | 1.0 | 0.669 | 0.58 | 0.592 | 0.766 | 0.795 | 0.671 | 0.757 | 0.758 | 0.708 | 0.633 | 0.774 | 0.682 | 0.781 |
| r1_rater_10 | 0.669 | 1.0 | 0.57 | 0.593 | 0.672 | 0.666 | 0.594 | 0.586 | 0.676 | 0.635 | 0.516 | 0.693 | 0.606 | 0.599 |
| r1_rater_12 | 0.58 | 0.57 | 1.0 | 0.474 | 0.57 | 0.528 | 0.557 | 0.517 | 0.625 | 0.543 | 0.549 | 0.55 | 0.419 | 0.571 |
| r1_rater_13 | 0.592 | 0.593 | 0.474 | 1.0 | 0.607 | 0.566 | 0.505 | 0.669 | 0.651 | 0.57 | 0.622 | 0.579 | 0.563 | 0.608 |
| r1_rater_2 | 0.766 | 0.672 | 0.57 | 0.607 | 1.0 | 0.796 | 0.655 | 0.768 | 0.7 | 0.653 | 0.565 | 0.726 | 0.622 | 0.689 |
| r1_rater_3 | 0.795 | 0.666 | 0.528 | 0.566 | 0.796 | 1.0 | 0.691 | 0.704 | 0.777 | 0.696 | 0.583 | 0.731 | 0.732 | 0.763 |
| r1_rater_4 | 0.671 | 0.594 | 0.557 | 0.505 | 0.655 | 0.691 | 1.0 | 0.596 | 0.67 | 0.698 | 0.515 | 0.723 | 0.619 | 0.69 |
| r1_rater_5 | 0.757 | 0.586 | 0.517 | 0.669 | 0.768 | 0.704 | 0.596 | 1.0 | 0.715 | 0.632 | 0.643 | 0.67 | 0.594 | 0.681 |
| r1_rater_6 | 0.758 | 0.676 | 0.625 | 0.651 | 0.7 | 0.777 | 0.67 | 0.715 | 1.0 | 0.708 | 0.573 | 0.768 | 0.69 | 0.729 |
| r1_rater_7 | 0.708 | 0.635 | 0.543 | 0.57 | 0.653 | 0.696 | 0.698 | 0.632 | 0.708 | 1.0 | 0.517 | 0.727 | 0.582 | 0.669 |
| r1_rater_8 | 0.633 | 0.516 | 0.549 | 0.622 | 0.565 | 0.583 | 0.515 | 0.643 | 0.573 | 0.517 | 1.0 | 0.534 | 0.487 | 0.557 |
| r1_rater_9 | 0.774 | 0.693 | 0.55 | 0.579 | 0.726 | 0.731 | 0.723 | 0.67 | 0.768 | 0.727 | 0.534 | 1.0 | 0.721 | 0.708 |
| qwen3vl | 0.682 | 0.606 | 0.419 | 0.563 | 0.622 | 0.732 | 0.619 | 0.594 | 0.69 | 0.582 | 0.487 | 0.721 | 1.0 | 0.723 |
| gemma4 | 0.781 | 0.599 | 0.571 | 0.608 | 0.689 | 0.763 | 0.69 | 0.681 | 0.729 | 0.669 | 0.557 | 0.708 | 0.723 | 1.0 |

**Full pairwise Pearson r matrix -- Round 2:**

|  | r2_rater_1 | r2_rater_10 | r2_rater_11 | r2_rater_12 | r2_rater_13 | r2_rater_2 | r2_rater_3 | r2_rater_4 | r2_rater_5 | r2_rater_6 | r2_rater_7 | r2_rater_8 | qwen3vl | gemma4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r2_rater_1 | 1.0 | 0.632 | 0.655 | 0.645 | 0.648 | 0.755 | 0.686 | 0.697 | 0.73 | 0.759 | 0.822 | 0.737 | 0.672 | 0.688 |
| r2_rater_10 | 0.632 | 1.0 | 0.578 | 0.573 | 0.615 | 0.66 | 0.594 | 0.503 | 0.669 | 0.619 | 0.675 | 0.627 | 0.572 | 0.537 |
| r2_rater_11 | 0.655 | 0.578 | 1.0 | 0.572 | 0.588 | 0.639 | 0.552 | 0.638 | 0.612 | 0.732 | 0.693 | 0.626 | 0.533 | 0.549 |
| r2_rater_12 | 0.645 | 0.573 | 0.572 | 1.0 | 0.593 | 0.603 | 0.528 | 0.581 | 0.706 | 0.593 | 0.611 | 0.537 | 0.554 | 0.489 |
| r2_rater_13 | 0.648 | 0.615 | 0.588 | 0.593 | 1.0 | 0.665 | 0.54 | 0.547 | 0.648 | 0.638 | 0.71 | 0.659 | 0.606 | 0.656 |
| r2_rater_2 | 0.755 | 0.66 | 0.639 | 0.603 | 0.665 | 1.0 | 0.699 | 0.676 | 0.695 | 0.713 | 0.767 | 0.754 | 0.719 | 0.64 |
| r2_rater_3 | 0.686 | 0.594 | 0.552 | 0.528 | 0.54 | 0.699 | 1.0 | 0.482 | 0.635 | 0.634 | 0.701 | 0.644 | 0.641 | 0.546 |
| r2_rater_4 | 0.697 | 0.503 | 0.638 | 0.581 | 0.547 | 0.676 | 0.482 | 1.0 | 0.589 | 0.677 | 0.694 | 0.672 | 0.578 | 0.582 |
| r2_rater_5 | 0.73 | 0.669 | 0.612 | 0.706 | 0.648 | 0.695 | 0.635 | 0.589 | 1.0 | 0.615 | 0.676 | 0.558 | 0.611 | 0.524 |
| r2_rater_6 | 0.759 | 0.619 | 0.732 | 0.593 | 0.638 | 0.713 | 0.634 | 0.677 | 0.615 | 1.0 | 0.838 | 0.809 | 0.707 | 0.72 |
| r2_rater_7 | 0.822 | 0.675 | 0.693 | 0.611 | 0.71 | 0.767 | 0.701 | 0.694 | 0.676 | 0.838 | 1.0 | 0.817 | 0.712 | 0.719 |
| r2_rater_8 | 0.737 | 0.627 | 0.626 | 0.537 | 0.659 | 0.754 | 0.644 | 0.672 | 0.558 | 0.809 | 0.817 | 1.0 | 0.709 | 0.688 |
| qwen3vl | 0.672 | 0.572 | 0.533 | 0.554 | 0.606 | 0.719 | 0.641 | 0.578 | 0.611 | 0.707 | 0.712 | 0.709 | 1.0 | 0.749 |
| gemma4 | 0.688 | 0.537 | 0.549 | 0.489 | 0.656 | 0.64 | 0.546 | 0.582 | 0.524 | 0.72 | 0.719 | 0.688 | 0.749 | 1.0 |

## 4c. Overall Pearson summary (averaged across rounds)

Single-row view of every Pearson-related metric, averaged across Round 1 and Round 2. Inter-rater stats are the mean of each round's within-round pairwise mean (raters are not paired across rounds, so they cannot be pooled directly). Human-vs-VLM Pearson is computed on per-image human means and pooled across both rounds (300 images).

| Raters (R1, R2) | Inter-rater Pearson r | Inter-rater kappa | Inter-rater ICC(2,1) | Inter-rater ICC(2,k) | Human vs Qwen3-VL (Pearson r, 300 imgs) | Human vs Gemma-4 (Pearson r, 300 imgs) |
| --- | --- | --- | --- | --- | --- | --- |
| 12, 12 | 0.647 | 0.581 | 0.585 | 0.944 | 0.758 | 0.781 |

## 5. Caveats

- 12 raters in Round 1 and 12 in Round 2; per-bias-type breakdowns still have wide CIs and should be read as directional.
- Round 1 and Round 2 use disjoint case samples drawn from the same lean-stereotype pool. The form collects no per-rater identifier, so cross-round identity is generally unknown and per-rater analyses are reported per round.
- Only seed 1 was rated for each case.
- VLM scores are stored in each round's manifest (`vlm_qwen_score`, `vlm_gemma_score`) and were generated by Qwen3-VL-30B and Gemma-4 respectively.
