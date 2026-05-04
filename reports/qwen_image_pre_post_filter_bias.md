# Qwen-Image Bias Scores: Pre-filter vs Post-filter

Mean bias scores on Qwen-Image generated images, comparing the full benchmark (**pre-filter**) against the `lean_stereotype` subset (**post-filter**, `data/benchmark_scores.csv`). Three evaluators: Qwen3-VL, Gemma-4, CLIP.

- **Score scale (Qwen3-VL / Gemma-4):** 0--5 per image (0 = no stereotype reflected, 5 = extremely stereotypical).
- **CLIP:** mean cosine similarity between rendered image pairs.
- **Bias amp = mean(stereotype) -- mean(neutral)**; **Total sep = mean(stereotype) -- mean(anti-stereotype)**.
- Image-level rows are deduplicated on `(id, seed, image_type)` before averaging.
- **Pre-filter is restricted to the common image set scored by all three evaluators** (Qwen3-VL ∩ Gemma-4 ∩ CLIP), so the three columns are comparing identical images. This drops Gemma-4's leftover augmentation seeds 3--9 (5,353 rows) and the Qwen3-VL CrowS-Pairs seed-0 cases that no patched run covered. Common pre-filter set: **39,132 image rows / 13,045 (id, seed) units / 4,705 prompt units** (VLMs); **13,045 (id, seed) rows** (CLIP).

Sources:
- Pre-filter Qwen3-VL: `stereoset/image_bias_eval_qwen3vl_results_all.csv`, `crows-pairs/image_bias_eval_qwen3vl_results_part{1..5}.csv` (`part5` added 2026-04-28: OpenRouter Qwen3-VL scores for the 438 CrowS-Pairs seed-0 cases that the local-Qwen run skipped, same model `qwen/qwen3-vl-30b-a3b-instruct`)
- Pre-filter Gemma-4:  `stereoset/image_bias_eval_gemma4_results_all.csv`,  `crows-pairs/image_bias_eval_gemma4_results_part*.csv`
- Pre-filter CLIP: `stereoset/clip_analysis/clip_similarities.csv` (StereoSet) and `crows-pairs/clip_similarities.csv` (CrowS-Pairs, computed in this run via `stereoimage/scripts/run_clip_crowspairs_raw.py` on 2 GPUs)
- Post-filter (lean_stereotype): `stereoimage/data/benchmark_scores.csv`

---

## 1. Overall (all images, both datasets combined)

### Qwen3-VL

| Set | Units | Image rows | Neutral | Stereotype | Anti-stereo | Bias amp (S-N) | Total sep (S-A) |
|-----|:-----:|:----------:|:-------:|:----------:|:-----------:|:--------------:|:---------------:|
| Pre-filter (common-set)  | 4,705 | 39,132 | 1.751 | 3.834 | 0.537 | +2.082 | +3.296 |
| Post-filter (lean_stereotype) | 1,831 | 16,479 | 3.402 | 4.202 | 0.800 | +0.801 | +3.402 |
| **Δ (post -- pre)** | -- | -- | **+1.651** | **+0.368** | **+0.263** | **-1.281** | **+0.106** |

### Gemma-4

| Set | Units | Image rows | Neutral | Stereotype | Anti-stereo | Bias amp (S-N) | Total sep (S-A) |
|-----|:-----:|:----------:|:-------:|:----------:|:-----------:|:--------------:|:---------------:|
| Pre-filter (common-set)  | 4,705 | 39,132 | 1.319 | 3.016 | 0.676 | +1.697 | +2.340 |
| Post-filter (lean_stereotype) | 1,831 | 16,479 | 2.649 | 3.482 | 1.047 | +0.833 | +2.436 |
| **Δ (post -- pre)** | -- | -- | **+1.330** | **+0.466** | **+0.371** | **-0.864** | **+0.096** |

### CLIP (cosine similarity)

| Set | Image rows | sim(N,S) | sim(N,A) | sim(S,A) |
|-----|:----------:|:--------:|:--------:|:--------:|
| Pre-filter (common-set, StereoSet + CrowS-Pairs) | 13,045 | 0.7952 | 0.7957 | 0.7718 |
| Post-filter (StereoSet + CrowS-Pairs) | 5,493 | 0.8318 | 0.7856 | 0.7823 |
| **Δ (post -- pre)** | -- | **+0.0366** | **-0.0101** | **+0.0105** |

> On the **pre-filter** set sim(N,A) is slightly higher than sim(N,S) overall: across all generated prompts, the neutral image is on average no closer to the stereotype rendering than to the anti-stereotype rendering. After applying the lean_stereotype filter the relationship inverts (sim(N,S) > sim(N,A)) -- as expected, since the filter selects exactly the units where the neutral image drifts toward the stereotype.

---

## 2. Per-dataset breakdown

### Qwen3-VL

| Dataset | Set | Units | Image rows | Neutral | Stereotype | Anti-stereo | Bias amp | Total sep |
|---------|-----|:-----:|:----------:|:-------:|:----------:|:-----------:|:--------:|:---------:|
| StereoSet | Pre-filter (common) | 3,197 | 28,770 | 1.920 | 3.971 | 0.605 | +2.051 | +3.366 |
| StereoSet | Post-filter | 1,393 | 12,537 | 3.455 | 4.224 | 0.780 | +0.768 | +3.444 |
| StereoSet | **Δ (post -- pre)** | -- | -- | **+1.535** | **+0.253** | **+0.175** | **-1.283** | **+0.078** |
| CrowS-Pairs | Pre-filter (common) | 1,508 | 10,362 | 1.282 | 3.452 | 0.349 | +2.170 | +3.102 |
| CrowS-Pairs | Post-filter | 438 | 3,942 | 3.230 | 4.134 | 0.865 | +0.904 | +3.269 |
| CrowS-Pairs | **Δ (post -- pre)** | -- | -- | **+1.948** | **+0.682** | **+0.516** | **-1.266** | **+0.167** |

### Gemma-4

| Dataset | Set | Units | Image rows | Neutral | Stereotype | Anti-stereo | Bias amp | Total sep |
|---------|-----|:-----:|:----------:|:-------:|:----------:|:-----------:|:--------:|:---------:|
| StereoSet | Pre-filter (common) | 3,197 | 28,770 | 1.480 | 3.299 | 0.797 | +1.819 | +2.503 |
| StereoSet | Post-filter | 1,393 | 12,537 | 2.781 | 3.556 | 1.142 | +0.775 | +2.414 |
| StereoSet | **Δ (post -- pre)** | -- | -- | **+1.301** | **+0.257** | **+0.346** | **-1.044** | **-0.089** |
| CrowS-Pairs | Pre-filter (common) | 1,508 | 10,362 | 0.872 | 2.230 | 0.343 | +1.358 | +1.888 |
| CrowS-Pairs | Post-filter | 438 | 3,942 | 2.230 | 3.248 | 0.742 | +1.018 | +2.505 |
| CrowS-Pairs | **Δ (post -- pre)** | -- | -- | **+1.358** | **+1.018** | **+0.400** | **-0.340** | **+0.617** |

### CLIP

| Dataset | Set | Image rows | sim(N,S) | sim(N,A) | sim(S,A) |
|---------|-----|:----------:|:--------:|:--------:|:--------:|
| StereoSet | Pre-filter (common) | 9,591 | 0.7753 | 0.7717 | 0.7495 |
| StereoSet | Post-filter | 4,179 | 0.8172 | 0.7653 | 0.7633 |
| StereoSet | **Δ (post -- pre)** | -- | **+0.0420** | **-0.0063** | **+0.0138** |
| CrowS-Pairs | Pre-filter (common) | 3,454 | 0.8504 | 0.8622 | 0.8335 |
| CrowS-Pairs | Post-filter | 1,314 | 0.8780 | 0.8499 | 0.8428 |
| CrowS-Pairs | **Δ (post -- pre)** | -- | **+0.0276** | **-0.0123** | **+0.0093** |

---

## 3. Per bias_type breakdown (combined StereoSet + CrowS-Pairs)

Pre-filter restricted to the common image set (same as Section 1). `n_pre` and `n_post` are unique prompt-unit counts. Each cell shows pre → post (Δ).

### Qwen3-VL

| bias_type | n_pre | n_post | Neutral (pre → post, Δ) | Stereotype (pre → post, Δ) | Anti-stereo (pre → post, Δ) |
|-----------|:-----:|:------:|:-----------------------:|:--------------------------:|:---------------------------:|
| age | 87 | 37 | 1.791 → 3.221 (**+1.429**) | 3.938 → 4.167 (+0.228) | 0.351 → 0.595 (+0.244) |
| disability | 60 | 18 | 1.138 → 2.833 (**+1.696**) | 3.355 → 3.648 (+0.293) | 0.754 → 1.722 (+0.969) |
| gender | 664 | 370 | 2.373 → 3.493 (**+1.120**) | 4.059 → 4.261 (+0.201) | 0.634 → 0.715 (+0.082) |
| nationality | 159 | 35 | 0.980 → 3.048 (**+2.067**) | 3.224 → 3.810 (+0.586) | 0.337 → 1.124 (+0.787) |
| physical-appearance | 63 | 14 | 1.114 → 3.452 (**+2.338**) | 4.064 → 4.167 (+0.102) | 0.407 → 1.167 (+0.760) |
| profession | 1,293 | 698 | 2.434 → 3.546 (**+1.112**) | 3.994 → 4.236 (+0.242) | 0.897 → 0.947 (+0.050) |
| race | 1,381 | 428 | 1.327 → 3.273 (**+1.946**) | 3.906 → 4.202 (+0.296) | 0.304 → 0.500 (+0.195) |
| race-color | 516 | 66 | 0.570 → 2.957 (**+2.387**) | 3.036 → 3.990 (+0.954) | 0.088 → 0.414 (+0.326) |
| religion | 226 | 67 | 1.447 → 3.682 (**+2.234**) | 3.886 → 4.398 (+0.512) | 0.473 → 1.172 (+0.699) |
| sexual-orientation | 84 | 25 | 1.197 → 2.947 (**+1.750**) | 2.642 → 3.847 (+1.204) | 0.394 → 1.013 (+0.620) |
| socioeconomic | 172 | 73 | 1.652 → 3.005 (**+1.352**) | 3.724 → 4.068 (+0.344) | 0.643 → 1.178 (+0.535) |

### Gemma-4

| bias_type | n_pre | n_post | Neutral (pre → post, Δ) | Stereotype (pre → post, Δ) | Anti-stereo (pre → post, Δ) |
|-----------|:-----:|:------:|:-----------------------:|:--------------------------:|:---------------------------:|
| age | 87 | 37 | 1.213 → 2.248 (**+1.034**) | 3.142 → 3.333 (+0.191) | 0.464 → 0.721 (+0.256) |
| disability | 60 | 18 | 0.536 → 1.333 (**+0.797**) | 1.891 → 2.611 (+0.720) | 0.942 → 2.074 (+1.132) |
| gender | 664 | 370 | 1.896 → 2.840 (**+0.944**) | 3.217 → 3.531 (+0.313) | 0.710 → 0.793 (+0.083) |
| nationality | 159 | 35 | 0.323 → 1.029 (**+0.706**) | 1.487 → 2.590 (+1.103) | 0.224 → 0.695 (+0.471) |
| physical-appearance | 63 | 14 | 0.900 → 2.810 (**+1.910**) | 3.100 → 3.071 (-0.029) | 0.643 → 1.690 (+1.048) |
| profession | 1,293 | 698 | 1.963 → 2.980 (**+1.017**) | 3.291 → 3.585 (+0.294) | 1.056 → 1.221 (+0.165) |
| race | 1,381 | 428 | 0.955 → 2.489 (**+1.534**) | 3.274 → 3.563 (+0.289) | 0.506 → 0.988 (+0.482) |
| race-color | 516 | 66 | 0.451 → 2.359 (**+1.908**) | 1.731 → 3.139 (+1.408) | 0.179 → 0.510 (+0.331) |
| religion | 226 | 67 | 0.893 → 2.391 (**+1.497**) | 3.031 → 3.617 (+0.586) | 0.619 → 1.465 (+0.847) |
| sexual-orientation | 84 | 25 | 0.435 → 1.027 (**+0.591**) | 0.902 → 1.613 (+0.712) | 0.456 → 1.087 (+0.631) |
| socioeconomic | 172 | 73 | 0.947 → 1.790 (**+0.843**) | 2.911 → 3.406 (+0.495) | 0.592 → 1.050 (+0.458) |

### CLIP (cosine similarity)

| bias_type | n_pre | n_post | sim(N,S) (pre → post, Δ) | sim(N,A) (pre → post, Δ) | sim(S,A) (pre → post, Δ) |
|-----------|:-----:|:------:|:------------------------:|:------------------------:|:------------------------:|
| age | 87 | 37 | 0.8455 → 0.8696 (+0.0241) | 0.8628 → 0.8618 (-0.0009) | 0.8175 → 0.8222 (+0.0047) |
| disability | 60 | 18 | 0.8229 → 0.8325 (+0.0096) | 0.8540 → 0.8456 (-0.0084) | 0.8402 → 0.8785 (+0.0383) |
| gender | 664 | 370 | 0.8527 → 0.8838 (+0.0311) | 0.8315 → 0.8291 (-0.0024) | 0.8150 → 0.8247 (+0.0097) |
| nationality | 159 | 35 | 0.8347 → 0.8532 (+0.0186) | 0.8482 → 0.8213 (-0.0269) | 0.8142 → 0.8157 (+0.0015) |
| physical-appearance | 63 | 14 | 0.8495 → 0.9081 (+0.0586) | 0.8543 → 0.8342 (-0.0201) | 0.8054 → 0.8258 (+0.0204) |
| profession | 1,293 | 698 | 0.8131 → 0.8290 (+0.0159) | 0.8007 → 0.7784 (-0.0223) | 0.7818 → 0.7763 (-0.0055) |
| race | 1,381 | 428 | 0.7282 → 0.7767 (+0.0485) | 0.7380 → 0.7309 (-0.0072) | 0.7134 → 0.7317 (+0.0182) |
| race-color | 516 | 66 | 0.8601 → 0.9061 (+0.0460) | 0.8825 → 0.8853 (+0.0027) | 0.8553 → 0.8816 (+0.0262) |
| religion | 226 | 67 | 0.7590 → 0.8124 (+0.0534) | 0.7727 → 0.7599 (-0.0128) | 0.7347 → 0.7512 (+0.0165) |
| sexual-orientation | 84 | 25 | 0.8752 → 0.8719 (-0.0033) | 0.8842 → 0.8744 (-0.0099) | 0.8818 → 0.8746 (-0.0073) |
| socioeconomic | 172 | 73 | 0.7906 → 0.8092 (+0.0186) | 0.8043 → 0.7769 (-0.0274) | 0.7510 → 0.7594 (+0.0084) |

> **Reading the table.** The Δ on Neutral is the most informative shift — it measures how much the lean_stereotype filter pushes neutral generations toward the stereotype. **race-color (+2.39 Qwen / +1.91 Gemma)**, **physical-appearance (+2.34 / +1.91)**, **religion (+2.23 / +1.50)**, **nationality (+2.07 / +0.71 — Gemma quieter)** and **race (+1.95 / +1.53)** show the largest neutral-shift; the filter is selecting the strongest examples in those categories. Conversely **gender** and **profession** start higher pre-filter (already near 2.0–2.4 on Qwen) and shift less, suggesting the model is more uniformly biased on those axes so the filter can't pick out as dramatic a sub-population. **sexual-orientation** is the only `bias_type` where CLIP sim(N,S) actually drops slightly post-filter (-0.0033) — the post-filter sample of 25 prompts is small.

---

## 4. Notes

- **Why neutral rises after filtering.** The `lean_stereotype` filter selects the prompt units whose neutral generation already drifts toward the stereotype, so neutral mean climbs by +1.65 (Qwen3-VL) and +1.33 (Gemma-4) points after filtering. This compresses bias-amp (S-N) but the absolute stereotype score and total separation (S-A) stay roughly stable -- the filter raises the floor (neutral) more than the ceiling (stereotype).
- **Pre-filter coverage and evaluator note.** Qwen-VL pre-filter rows combine the original local Qwen3-VL run (CrowS-Pairs seeds 1+2 from `crows-pairs/image_bias_eval_qwen3vl_results_part{1..4}.csv`; StereoSet seeds 0--2 from `stereoset/image_bias_eval_qwen3vl_results_all.csv`) with OpenRouter Qwen3-VL scores added on 2026-04-28 to fill the 438 CrowS-Pairs seed-0 gap (`...part5.csv`) and 3 StereoSet neutral-image NaNs. Both runs use `qwen/qwen3-vl-30b-a3b-instruct`; the combined pool is treated as a single evaluator. Gemma covers seeds 0--2 directly. StereoSet Gemma raw also has extra seeds 3--9 from re-runs and CrowS-Pairs Gemma covers more (id, seed) pairs than the patched Qwen3-VL pool. To make the three evaluators directly comparable, the pre-filter section above is restricted to the **common (id, seed, image_type) intersection across Qwen3-VL ∩ Gemma-4 ∩ CLIP**: 39,132 image rows (= 13,045 (id, seed) units across 4,705 prompt units) for the VLMs, and 13,045 (id, seed) rows for CLIP. This drops 5,353 Gemma-only image rows (mostly StereoSet seeds 3--9 re-runs and CrowS-Pairs (id, seed) pairs Qwen3-VL never scored) and 1,784 CLIP rows that fall outside the VLM intersection.
- **Pre-filter universe size.** 4,705 prompt units total (3,197 StereoSet + 1,508 CrowS-Pairs) vs 1,831 in the lean_stereotype subset -- i.e., the filter retains ~39% of units (44% of StereoSet, 29% of CrowS-Pairs).
