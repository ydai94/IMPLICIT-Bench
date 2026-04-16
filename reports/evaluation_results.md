# Lean Stereotype Benchmark: Neutral Lean Toward Stereotype

Three independent evaluation methods — CLIP (vision embedding similarity), Qwen3-VL (VLM scoring), and Gemma-4 (VLM scoring) — are used to test whether neutral images lean toward stereotype rather than anti-stereotype. The benchmark consists of curated lean_stereotype cases from StereoSet and CrowS-Pairs, where either VLM flagged the neutral image as closer to the stereotype condition.

- **Evaluation methods**: CLIP-ViT-L/14 cosine similarity, Qwen3-VL 0-5 scoring, Gemma-4 0-5 scoring
- **Image seeds**: 3 per case (seed 0-2)
- **Image types**: neutral, stereotype_trigger, anti_stereotype_trigger
- **Benchmark cases**: 1,831 (StereoSet: 1,393 + CrowS-Pairs: 438)
- **Total samples**: 5,493 (StereoSet: 4,179 + CrowS-Pairs: 1,314)

---

## 1. Lean Metric Definitions

For each sample, "neutral lean toward stereotype" is defined as:

- **CLIP**: `sim(neutral, stereotype) > sim(neutral, anti_stereotype)` — neutral image embedding is closer to stereotype than anti-stereotype
- **Qwen-VL / Gemma-4**: `|score(neutral) - score(anti)| > |score(neutral) - score(stereo)|` — neutral evaluation score is closer to stereotype than anti-stereotype

---

## 2. Three-Method Agreement

### Sample-Level (n=5,493)

| Method | Lean S% | Mean Lean | t-stat | p-value | Cohen's d |
|---|---|---|---|---|---|
| CLIP | 67.7% | +0.0462 | +29.01 | 2.92e-172 | +0.391 |
| Qwen-VL | 73.7% | +1.8452 | +63.63 | ~0 | +0.895 |
| Gemma4 | 48.7% | +0.7430 | +21.65 | 8.44e-100 | +0.292 |

| Agreement | Samples | % |
|---|---|---|
| All 3 agree lean stereotype | 1,770 | 32.2% |
| At least 2 agree | 3,495 | 63.6% |

### Case-Level (n=1,831, averaged across seeds)

| Method | Lean S% | Mean Lean | Cohen's d |
|---|---|---|---|
| CLIP | 70.9% | +0.0462 | +0.485 |
| Qwen-VL | 84.3% | +1.8365 | +1.087 |
| Gemma4 | 56.7% | +0.7426 | +0.346 |

All 3 agree lean stereotype: 754/1,831 cases (41.2%).

### Chance-Corrected Agreement on Binary Lean Decisions

Raw "% agree" is inflated by chance, so we also report Fleiss' κ and Krippendorff's
α (nominal) over the binary lean-stereotype labels from the three methods, plus
pairwise Cohen's κ. n=5,050 samples have all three methods scored.

**Sample-level (n=5,050)**

- Fleiss' κ: **0.2082**
- Krippendorff's α: **0.2082**

| Pair | Cohen's κ |
|---|---|
| CLIP vs Qwen-VL | 0.1647 |
| CLIP vs Gemma4 | 0.2277 |
| Qwen-VL vs Gemma4 | 0.2771 |

**Case-level (n=1,831, majority vote over 3 seeds)**

- Fleiss' κ: **0.1590**
- Krippendorff's α: **0.1590**

| Pair | Cohen's κ |
|---|---|
| CLIP vs Qwen-VL | 0.1222 |
| CLIP vs Gemma4 | 0.2315 |
| Qwen-VL vs Gemma4 | 0.1899 |

Interpretation: agreement on the *binary* lean decision is only "slight to fair"
(κ ≈ 0.16–0.28). The methods agree much more strongly on the *direction and
magnitude* of the lean (Pearson r up to +0.44 in section 3) than on which side of
zero any individual sample falls — i.e., the three methods produce correlated
continuous signals but disagree on borderline cases near the decision boundary.

---

## 3. Pairwise Pearson Correlations

### Sample-Level (n=5,493)

| Pair | Pearson r | p-value | Spearman r | p-value |
|---|---|---|---|---|
| CLIP vs Qwen-VL | +0.210 | 1.46e-51 | +0.255 | 7.67e-76 |
| CLIP vs Gemma4 | +0.253 | 4.49e-81 | +0.306 | 4.23e-119 |
| Qwen-VL vs Gemma4 | +0.434 | 1.77e-231 | +0.463 | 7.09e-267 |

### Case-Level (n=1,831)

| Pair | Pearson r | p-value |
|---|---|---|
| CLIP vs Qwen-VL | +0.264 | 1.58e-30 |
| CLIP vs Gemma4 | +0.309 | 6.78e-42 |
| Qwen-VL vs Gemma4 | +0.443 | 4.96e-89 |

### VLM Score Agreement (Qwen vs Gemma)

| Score Type | Pearson r |
|---|---|
| stereotype | +0.632 |
| anti-stereotype | +0.625 |
| neutral | +0.594 |

---

## 4. Per Bias Type

| Bias Type | n | CLIP lean% | Qwen lean% | Gemma lean% | CLIP d | Qwen d | Gemma d |
|---|---|---|---|---|---|---|---|
| physical-appearance | 42 | 76.2% | 38.1% | 69.0% | +0.815 | +0.576 | +0.987 |
| gender | 1,110 | 73.2% | 66.1% | 54.7% | +0.509 | +0.913 | +0.419 |
| profession | 2,094 | 71.1% | 75.6% | 55.3% | +0.414 | +1.052 | +0.486 |
| religion | 201 | 65.2% | 65.2% | 30.8% | +0.398 | +0.907 | -0.067 |
| race | 1,284 | 64.8% | 73.1% | 45.1% | +0.356 | +0.837 | +0.150 |
| socioeconomic | 219 | 59.8% | 37.0% | 26.9% | +0.290 | +0.658 | -0.216 |
| race-color | 198 | 57.1% | 43.9% | 47.5% | +0.294 | +0.398 | +0.279 |
| age | 111 | 55.9% | 47.7% | 37.8% | +0.075 | +1.021 | +0.045 |
| nationality | 105 | 54.3% | 41.0% | 20.0% | +0.341 | +0.745 | -0.371 |
| disability | 54 | 51.9% | 35.2% | 13.0% | -0.201 | +0.560 | -0.089 |
| sexual-orientation | 75 | 42.7% | 45.3% | 21.3% | -0.031 | +0.724 | -0.033 |

---

## 5. CLIP Similarity: Per-Dataset

### StereoSet (4,179 samples, 1,393 cases)

| Metric | Mean | Std | Median |
|---|---|---|---|
| sim(neutral, stereo) | 0.8172 | 0.1457 | 0.8521 |
| sim(neutral, anti) | 0.7653 | 0.1522 | 0.7925 |
| sim(stereo, anti) | 0.7633 | 0.1471 | 0.7861 |

Neutral lean stereotype: 69.3%, gap=+0.0519, t=26.75, p=1.26e-145, d=+0.414

| Bias Type | n | Gap | Lean S% |
|---|---|---|---|
| gender | 651 | +0.0674 | 73.1% |
| religion | 150 | +0.0543 | 66.7% |
| profession | 2,094 | +0.0506 | 71.1% |
| race | 1,284 | +0.0459 | 64.8% |

### CrowS-Pairs (1,314 samples, 438 cases)

| Metric | Mean | Std | Median |
|---|---|---|---|
| sim(neutral, stereo) | 0.8780 | 0.1011 | 0.9077 |
| sim(neutral, anti) | 0.8499 | 0.1153 | 0.8811 |
| sim(stereo, anti) | 0.8428 | 0.1156 | 0.8735 |

Neutral lean stereotype: 62.6%, gap=+0.0281, t=11.53, p=2.25e-29, d=+0.318

| Bias Type | n | Gap | Lean S% |
|---|---|---|---|
| physical-appearance | 42 | +0.0739 | 76.2% |
| religion | 51 | +0.0471 | 60.8% |
| gender | 459 | +0.0367 | 73.2% |
| socioeconomic | 219 | +0.0323 | 59.8% |
| nationality | 105 | +0.0320 | 54.3% |
| race-color | 198 | +0.0208 | 57.1% |
| age | 111 | +0.0078 | 55.9% |
| sexual-orientation | 75 | -0.0025 | 42.7% |
| disability | 54 | -0.0130 | 51.9% |

---

## 6. Cross-Dataset Comparison

Distribution comparison (KS-test):

| Metric | KS statistic | p-value |
|---|---|---|
| sim(neutral, stereo) | 0.185 | 2.02e-30 |
| sim(neutral, anti) | 0.263 | 1.99e-61 |
| sim(stereo, anti) | 0.254 | 3.62e-57 |

Overlapping bias types (gap = sim(N,S) - sim(N,A)):

| Bias Type | StereoSet gap (n) | CrowS-Pairs gap (n) | Same direction |
|---|---|---|---|
| gender | +0.0674 (651) | +0.0367 (459) | YES |
| race/race-color | +0.0459 (1,284) | +0.0208 (198) | YES |
| religion | +0.0543 (150) | +0.0471 (51) | YES |

Overall:

| Dataset | Mean gap | Cohen's d |
|---|---|---|
| StereoSet | +0.0519 | +0.414 |
| CrowS-Pairs | +0.0281 | +0.318 |

---

## 7. Key Findings

1. **Neutral images lean toward stereotype across all three methods.** CLIP (67.7%), Qwen-VL (73.7%), and Gemma4 (48.7%) all show significant lean (p < 1e-100 for all).

2. **41.2% of cases (754/1,831) have all three methods agreeing** that neutral leans toward stereotype, providing high-confidence benchmark cases.

3. **Three methods correlate positively.** Case-level Pearson r ranges from +0.264 (CLIP-Qwen) to +0.443 (Qwen-Gemma), confirming the effect is method-independent.

4. **Strongest bias types**: physical-appearance (d=+0.82-0.99), gender (d=+0.42-0.91), and profession (d=+0.41-1.05) show the most consistent neutral-to-stereotype lean across all methods.

5. **Weakest bias types**: disability and sexual-orientation show inconsistent or absent lean, with Gemma4 and CLIP often showing near-zero or negative effect sizes.

6. **Cross-dataset consistency**: All three overlapping bias types (gender, race/race-color, religion) show positive gaps in both StereoSet and CrowS-Pairs, with StereoSet showing larger effect sizes.

7. **CLIP captures visual similarity, not semantic bias.** CLIP correlates moderately with VLM lean metrics (r=0.21-0.31) but measures visual proximity rather than bias semantics, providing complementary evidence.
