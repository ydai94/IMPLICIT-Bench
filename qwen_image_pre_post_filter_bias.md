# Qwen-Image Bias Scores: Pre-filter vs Post-filter

Mean bias scores on Qwen-Image generated images, comparing the full benchmark (**pre-filter**) against the `lean_stereotype` subset (**post-filter**, `data/merged_all.csv`). Three evaluators: Qwen3-VL, Gemma-4, CLIP.

- **Score scale (Qwen3-VL / Gemma-4):** 0--5 per image (0 = no stereotype reflected, 5 = extremely stereotypical).
- **CLIP:** mean cosine similarity between rendered image pairs.
- **Bias amp = mean(stereotype) -- mean(neutral)**; **Total sep = mean(stereotype) -- mean(anti-stereotype)**.
- Image-level rows are deduplicated on `(id, seed, image_type)` before averaging.

Sources:
- Pre-filter Qwen3-VL: `stereoset/image_bias_eval_qwen3vl_results_all.csv`, `crows-pairs/image_bias_eval_qwen3vl_results_part{1..5}.csv` (`part5` added 2026-04-28: OpenRouter Qwen3-VL scores for the 438 CrowS-Pairs seed-0 cases that the local-Qwen run skipped, same model `qwen/qwen3-vl-30b-a3b-instruct`)
- Pre-filter Gemma-4:  `stereoset/image_bias_eval_gemma4_results_all.csv`,  `crows-pairs/image_bias_eval_gemma4_results_part*.csv`
- Pre-filter CLIP: `stereoset/clip_analysis/clip_similarities.csv` (StereoSet) and `crows-pairs/clip_similarities.csv` (CrowS-Pairs, computed in this run via `stereoimage/scripts/run_clip_crowspairs_raw.py` on 2 GPUs)
- Post-filter (lean_stereotype): `stereoimage/data/merged_all.csv`

---

## 1. Overall (all images, both datasets combined)

### Qwen3-VL

| Set | Units | Image rows | Neutral | Stereotype | Anti-stereo | Bias amp (S-N) | Total sep (S-A) |
|-----|:-----:|:----------:|:-------:|:----------:|:-----------:|:--------------:|:---------------:|
| Pre-filter (all)  | 4,705 | 40,854 | 1.756 | 3.817 | 0.542 | +2.061 | +3.275 |
| Post-filter (lean_stereotype) | 1,831 | 16,479 | 3.402 | 4.202 | 0.800 | +0.801 | +3.402 |
| **Δ (post -- pre)** | -- | -- | **+1.646** | **+0.385** | **+0.258** | **-1.260** | **+0.127** |

### Gemma-4

| Set | Units | Image rows | Neutral | Stereotype | Anti-stereo | Bias amp (S-N) | Total sep (S-A) |
|-----|:-----:|:----------:|:-------:|:----------:|:-----------:|:--------------:|:---------------:|
| Pre-filter (all)  | 4,705 | 44,487 | 1.256 | 2.940 | 0.653 | +1.684 | +2.287 |
| Post-filter (lean_stereotype) | 1,831 | 16,479 | 2.649 | 3.482 | 1.047 | +0.833 | +2.436 |
| **Δ (post -- pre)** | -- | -- | **+1.394** | **+0.542** | **+0.394** | **-0.852** | **+0.149** |

### CLIP (cosine similarity)

| Set | Image rows | sim(N,S) | sim(N,A) | sim(S,A) |
|-----|:----------:|:--------:|:--------:|:--------:|
| Pre-filter (StereoSet + CrowS-Pairs) | 14,829 | 0.7978 | 0.8005 | 0.7751 |
| Post-filter (StereoSet + CrowS-Pairs) | 5,493 | 0.8318 | 0.7856 | 0.7823 |
| **Δ (post -- pre)** | -- | **+0.0340** | **-0.0149** | **+0.0072** |

> On the **pre-filter** set sim(N,A) is slightly higher than sim(N,S) overall: across all generated prompts, the neutral image is on average no closer to the stereotype rendering than to the anti-stereotype rendering. After applying the lean_stereotype filter the relationship inverts (sim(N,S) > sim(N,A)) -- as expected, since the filter selects exactly the units where the neutral image drifts toward the stereotype.

---

## 2. Per-dataset breakdown

### Qwen3-VL

| Dataset | Set | Units | Neutral | Stereotype | Anti-stereo | Bias amp | Total sep |
|---------|-----|:-----:|:-------:|:----------:|:-----------:|:--------:|:---------:|
| StereoSet | Pre-filter | 3,197 | 1.926 | 3.947 | 0.611 | +2.021 | +3.337 |
| StereoSet | Post-filter | 1,393 | 3.455 | 4.224 | 0.780 | +0.768 | +3.444 |
| CrowS-Pairs | Pre-filter | 1,508 | 1.268 | 3.442 | 0.345 | +2.175 | +3.097 |
| CrowS-Pairs | Post-filter | 438 | 3.230 | 4.134 | 0.865 | +0.904 | +3.269 |

### Gemma-4

| Dataset | Set | Units | Neutral | Stereotype | Anti-stereo | Bias amp | Total sep |
|---------|-----|:-----:|:-------:|:----------:|:-----------:|:--------:|:---------:|
| StereoSet | Pre-filter | 3,197 | 1.503 | 3.317 | 0.813 | +1.814 | +2.504 |
| StereoSet | Post-filter | 1,393 | 2.781 | 3.556 | 1.142 | +0.775 | +2.414 |
| CrowS-Pairs | Pre-filter | 1,508 | 0.692 | 2.081 | 0.287 | +1.389 | +1.794 |
| CrowS-Pairs | Post-filter | 438 | 2.230 | 3.248 | 0.742 | +1.018 | +2.505 |

### CLIP

| Dataset | Set | Image rows | sim(N,S) | sim(N,A) | sim(S,A) |
|---------|-----|:----------:|:--------:|:--------:|:--------:|
| StereoSet | Pre-filter | 9,591 | 0.7762 | 0.7726 | 0.7498 |
| StereoSet | Post-filter | 4,179 | 0.8172 | 0.7653 | 0.7633 |
| CrowS-Pairs | Pre-filter | 4,524 | 0.8469 | 0.8640 | 0.8328 |
| CrowS-Pairs | Post-filter | 1,314 | 0.8780 | 0.8499 | 0.8428 |

---

## 3. Notes

- **Why neutral rises after filtering.** The `lean_stereotype` filter selects the prompt units whose neutral generation already drifts toward the stereotype, so neutral mean climbs by +1.65 (Qwen3-VL) and +1.39 (Gemma-4) points after filtering. This compresses bias-amp (S-N) but the absolute stereotype score and total separation (S-A) stay roughly stable -- the filter raises the floor (neutral) more than the ceiling (stereotype).
- **Pre-filter coverage and evaluator note.** Qwen-VL pre-filter rows now combine the original local Qwen3-VL run (CrowS-Pairs seeds 1+2 from `crows-pairs/image_bias_eval_qwen3vl_results_part{1..4}.csv`; StereoSet seeds 0--2 from `stereoset/image_bias_eval_qwen3vl_results_all.csv`) with OpenRouter Qwen3-VL scores added on 2026-04-28 to fill the 438 CrowS-Pairs seed-0 gap (`...part5.csv`) and 3 StereoSet neutral-image NaNs. Both runs use `qwen/qwen3-vl-30b-a3b-instruct`; the combined pool is treated as a single evaluator. Gemma covers 0--2 directly. StereoSet Gemma raw has extra seeds 3--9 from re-runs (kept after dedup). Raw CLIP for CrowS-Pairs was computed in this run (1,508 units × 3 seeds = 4,524 triplets).
- **Pre-filter universe size.** 4,705 prompt units total (3,197 StereoSet + 1,508 CrowS-Pairs) vs 1,831 in the lean_stereotype subset -- i.e., the filter retains ~39% of units (44% of StereoSet, 29% of CrowS-Pairs).
