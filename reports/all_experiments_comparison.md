# Stereotype Bias + Image–Prompt Alignment Comparison Across All 12 Experiments

**Date:** 2026-04-21
**Dataset:** `data/benchmark_prompts.csv` (1,831 cases × 3 seeds = 5,493 images per experiment)
**Evaluators (both Qwen3-VL-30B-A3B-Instruct via OpenRouter):**
- **Bias score**: 0–5 scale of how strongly the image reflects the stereotype (lower = better).
- **Alignment**: binary yes/no — does the image correctly depict `prompt_neutral`? (higher = better).

**Coverage:** 12 experiments × 5,493 images × 2 judges = 131,832 VLM evaluations.

> **Framing (benchmark paper).** This is a benchmark with two experimental settings:
> - **Realistic** — the deployable pipeline uses *extracted* KG. Exp 1, 2, 4, 5, 8, 10 are the methods under this setting.
> - **Ideal upper bound** — *ground-truth* KG annotations, available only in the benchmark. Exp 3, 6, 7, 9, 11 are the methods under this setting.
>
> The primary method comparisons are **within a KG source** (extracted vs extracted, GT vs GT). Extracted-vs-GT cross-KG comparisons are treated as a **diagnostic** of pipeline headroom (§12), not as head-to-head method evaluations. GT is the ceiling the realistic setup is measured against.
>
> **Headline:** under realistic KG, Exp 2 (LLM rewrite) ≈ Exp 5 (tail SV) win at composite ~30.1–30.2 — only ~2 points above baseline. Under GT KG, Exp 11 (GT-pair SV) hits 40.6. The ~10-point gap is the single largest source of benchmark headroom and comes from KG-quality alone.

---

## 1. Experiment Legend

| ID | Name | Family | KG source | Intervention |
|----|------|--------|-----------|-------------|
| 0 | baseline | none | — | raw prompt |
| 1 | llm_rewrite_no_kg | prompt rewrite | — | LLM rewrite without KG |
| 2 | extracted_kg_llm_rewrite | prompt rewrite | extracted | LLM rewrite guided by extracted KG |
| 3 | gt_kg_llm_rewrite | prompt rewrite | GT | LLM rewrite guided by GT KG |
| 4 | extracted_kg_full_triple_sv | steering vector | extracted | full-triple SV |
| 5 | extracted_kg_tail_sv | steering vector | extracted | tail-only SV |
| 6 | gt_kg_full_triple_sv | steering vector | GT | full-triple SV |
| 7 | gt_kg_tail_sv | steering vector | GT | tail-only SV |
| 8 | extracted_kg_llm_pair_sv | steering vector | extracted | LLM-generated anti/stereo pair SV |
| 9 | gt_kg_llm_pair_sv | steering vector | GT | LLM-generated anti/stereo pair SV |
| 10 | extracted_kg_gt_pair_sv | steering vector | extracted | GT anti/stereo pair SV |
| 11 | gt_kg_gt_pair_sv | steering vector | GT | GT anti/stereo pair SV |

---

## 2. Joint Results — ranked by composite score

**Composite** = `aligned_rate × (1 − bias_mean/5) × 100`
Rewards methods that both keep the image on-prompt *and* drive bias toward 0. Baseline = 28.17.

| Rank | Exp | Name | Bias mean | Bias Δ vs base | Aligned % | Align Δ vs base | **Composite** |
|---:|---:|------|------:|---------:|--------:|----------:|----:|
| 1 | **11** | gt_kg_gt_pair_sv | 1.992 | −41.3% | 67.5% | −20.0 pts | **40.61** |
| 2 | **7** | gt_kg_tail_sv | 2.180 | −35.7% | 71.0% | −16.5 pts | **40.07** |
| 3 | **3** | gt_kg_llm_rewrite | 2.597 | −23.4% | 80.4% | −7.2 pts | **38.62** |
| 4 | 6 | gt_kg_full_triple_sv | 2.794 | −17.6% | 74.7% | −12.9 pts | 32.94 |
| 5 | 1 | llm_rewrite_no_kg | 3.009 | −11.3% | 76.4% | −11.1 pts | 30.41 |
| 6 | 5 | extracted_kg_tail_sv | 2.802 | −17.4% | 68.8% | −18.8 pts | 30.24 |
| 7 | 2 | extracted_kg_llm_rewrite | 2.965 | −12.6% | 73.9% | −13.7 pts | 30.06 |
| — | 0 | **baseline** | 3.391 | 0.0% | **87.5%** | 0.0 pts | **28.17** |
| 8 | 10 | extracted_kg_gt_pair_sv | 2.792 | −17.7% | 62.9% | −24.6 pts | 27.80 |
| 9 | 4 | extracted_kg_full_triple_sv | 3.204 | −5.5% | 74.6% | −12.9 pts | 26.81 |
| 10 | 9 | gt_kg_llm_pair_sv | 1.011 | **−70.2%** | **23.2%** | **−64.3 pts** | 18.52 |
| 11 | 8 | extracted_kg_llm_pair_sv | 1.683 | −50.4% | **21.7%** | **−65.9 pts** | 14.37 |

**Headline reversal:** the two LLM-pair-SV methods (Exp 8 & 9), previously ranked #1 and #2 on bias alone, drop to the **bottom two** on the composite score. Their bias-reduction comes at the cost of the VLM no longer finding the prompted subject in the image.

---

## 3. Bias on *aligned-only* images — the real debiasing signal

For each method, we filter to only the images the alignment judge said *correctly depict the neutral prompt*, then compute mean bias on that subset. This isolates genuine debiasing from "bias disappeared because the subject did."

| Exp | Name | n (aligned) | Bias on aligned | Bias on all | Hidden cost |
|----:|------|----:|-----:|-----:|-----:|
| **9** | gt_kg_llm_pair_sv | 1,275 | **1.822** | 1.011 | +0.811 |
| **11** | gt_kg_gt_pair_sv | **3,708** | **2.266** | 1.992 | +0.274 |
| 7 | gt_kg_tail_sv | 3,902 | 2.515 | 2.180 | +0.335 |
| 8 | extracted_kg_llm_pair_sv | 1,190 | 2.655 | 1.683 | +0.971 |
| 3 | gt_kg_llm_rewrite | 4,414 | 2.763 | 2.597 | +0.165 |
| 6 | gt_kg_full_triple_sv | 4,101 | 3.029 | 2.794 | +0.235 |
| 5 | extracted_kg_tail_sv | 3,778 | 3.130 | 2.802 | +0.329 |
| 10 | extracted_kg_gt_pair_sv | 3,457 | 3.140 | 2.792 | +0.348 |
| 2 | extracted_kg_llm_rewrite | 4,058 | 3.225 | 2.965 | +0.260 |
| 1 | llm_rewrite_no_kg | 4,196 | 3.262 | 3.009 | +0.253 |
| 4 | extracted_kg_full_triple_sv | 4,099 | 3.408 | 3.204 | +0.205 |
| 0 | baseline | 4,808 | 3.436 | 3.391 | +0.045 |

- **Exp 9 is still the lowest-bias method even on aligned-only images** (1.82), but it only produces **1,275 usable images out of 5,493** (23%). Exp 11 produces nearly **3× as many usable images** (3,708) at a bias of 2.27 — only 0.44 points higher.
- Every method's bias increases when we restrict to aligned images (positive "hidden cost"). The size of the gap is a tell: **Exp 8 (+0.97) and Exp 9 (+0.81) were hiding the most behind content destruction**, while Exp 3 (+0.17), Exp 4 (+0.21), and Exp 6 (+0.24) are essentially honest.
- **Ordering by bias-on-aligned:** 9 < 11 < 7 < 8 < 3 < 6 < 5 < 10 < 2 < 1 < 4 < 0. Here the three GT-KG steering methods (11, 7, 9) again dominate.

---

## 4. Alignment flip analysis (per image vs baseline)

For each shared `(case_id, seed)`, compare whether the alignment judge's verdict flipped. Baseline aligned 87.5% of images; each method perturbs that.

| Exp | Name | Kept aligned | **Lost alignment** | Recovered | Stayed misaligned |
|---:|------|------:|------:|------:|------:|
| 3 | gt_kg_llm_rewrite | 75.1% | **12.5%** | 5.3% | 7.2% |
| 1 | llm_rewrite_no_kg | 71.3% | 16.3% | 5.1% | 7.4% |
| 6 | gt_kg_full_triple_sv | 70.7% | 16.8% | 3.9% | 8.5% |
| 4 | extracted_kg_full_triple_sv | 70.6% | 16.9% | 4.0% | 8.4% |
| 2 | extracted_kg_llm_rewrite | 68.9% | 18.7% | 5.0% | 7.5% |
| 7 | gt_kg_tail_sv | 67.3% | 20.3% | 3.8% | 8.7% |
| 5 | extracted_kg_tail_sv | 65.2% | 22.3% | 3.6% | 8.9% |
| 11 | gt_kg_gt_pair_sv | 63.8% | 23.7% | 3.7% | 8.8% |
| 10 | extracted_kg_gt_pair_sv | 59.4% | 28.2% | 3.6% | 8.9% |
| 9 | gt_kg_llm_pair_sv | 22.1% | **65.5%** | 1.2% | 11.3% |
| 8 | extracted_kg_llm_pair_sv | 20.7% | **66.9%** | 1.0% | 11.5% |

- **Losses dominate recoveries across every method** (3–5× for most, 50–65× for pair-SV). No method reliably *fixes* baseline misalignment; they just add new misalignments at different rates.
- **LLM rewrite with GT KG (Exp 3)** is by far the gentlest intervention — only 12.5% of baseline-aligned images lose alignment.
- **LLM-pair SV (Exp 8 & 9)** shatters alignment on ~2/3 of images. This is a **qualitative** gap, not a quantitative one — they are operating in a different regime.

---

## 5. Score distributions (bias)

| Exp | 0 | 1 | 2 | 3 | 4 | 5 |
|----:|---:|---:|---:|---:|---:|---:|
| 0 baseline | 5.0 | 2.6 | 13.1 | 16.4 | **53.5** | 9.5 |
| 1 llm_rewrite_no_kg | 14.5 | 3.7 | 17.7 | 10.0 | 38.6 | 15.5 |
| 2 extracted_kg_llm_rewrite | 14.2 | 6.2 | 17.8 | 8.1 | 37.7 | 15.9 |
| 3 gt_kg_llm_rewrite | 18.7 | 11.4 | 18.7 | 7.1 | 30.9 | 13.2 |
| 4 extracted_kg_full_triple_sv | 10.7 | 4.1 | 16.7 | 8.4 | 43.0 | 17.2 |
| 5 extracted_kg_tail_sv | 18.7 | 6.5 | 15.8 | 8.6 | 35.8 | 14.7 |
| 6 gt_kg_full_triple_sv | 16.2 | 7.8 | 19.1 | 7.9 | 35.2 | 13.8 |
| 7 gt_kg_tail_sv | 27.9 | 11.0 | 19.7 | 7.1 | 25.5 | 8.9 |
| **8 extracted_kg_llm_pair_sv** | **45.8** | 8.2 | 13.0 | 5.8 | 19.0 | 8.1 |
| **9 gt_kg_llm_pair_sv** | **57.7** | 14.0 | 13.5 | 2.9 | 8.5 | **3.5** |
| 10 extracted_kg_gt_pair_sv | 18.3 | 6.3 | 17.3 | 8.3 | 35.7 | 14.2 |
| 11 gt_kg_gt_pair_sv | 31.5 | 12.1 | 19.2 | 6.5 | 24.2 | 6.5 |

Exp 9's 57.7% score-0 mass now looks suspicious in light of §3 — many of those zeros are "no stereotype because no subject."

---

## 6. Alignment-rate by bias type

| Bias type | n | base | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|-----------|---:|-----:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| age | 111 | 94.6 | 78.4 | 70.3 | 89.2 | 75.7 | 78.4 | 77.5 | 84.7 | 19.8 | 15.3 | 66.7 | 79.3 |
| disability | 54 | 98.1 | 87.0 | 81.5 | 77.8 | 81.5 | 88.9 | 50.0 | 87.0 | 20.4 | 9.3 | 61.1 | 81.5 |
| gender | 1110 | 88.7 | 72.9 | 72.2 | 80.3 | 70.5 | 68.6 | 75.0 | 73.5 | 13.7 | 18.6 | 63.4 | 69.0 |
| nationality | 105 | 100.0 | 85.7 | 61.0 | 95.2 | 81.0 | 84.8 | 76.2 | 90.5 | 7.6 | 11.4 | 65.7 | 93.3 |
| physical-appearance | 42 | 92.9 | 69.0 | 76.2 | 88.1 | 59.5 | 47.6 | 71.4 | 90.5 | 2.4 | 0.0 | 42.9 | 83.3 |
| profession | 2094 | 87.0 | 82.5 | 76.6 | 81.1 | 79.0 | 68.5 | 77.5 | 71.8 | 27.0 | 30.4 | 64.3 | 65.8 |
| race | 1284 | 81.6 | 67.7 | 68.7 | 73.1 | 69.2 | 59.7 | 69.8 | 56.5 | 22.5 | 23.1 | 56.4 | 59.7 |
| race-color | 198 | 97.0 | 91.9 | 81.8 | 96.5 | 78.3 | 83.3 | 86.4 | 89.4 | 26.8 | 21.7 | 72.7 | 90.9 |
| religion | 201 | 90.5 | 59.7 | 72.6 | 78.6 | 77.1 | 77.6 | 65.2 | 70.6 | 25.4 | 14.9 | 72.1 | 66.7 |
| sexual-orientation | 75 | 94.7 | 70.7 | 68.0 | 93.3 | 81.3 | 76.0 | 89.3 | 96.0 | 9.3 | 24.0 | 66.7 | 88.0 |
| socioeconomic | 219 | 94.1 | 83.1 | 88.6 | 85.8 | 74.9 | 89.0 | 72.1 | 88.1 | 13.7 | 4.6 | 68.5 | 69.9 |

- **physical-appearance under Exp 9: 0% aligned.** The LLM-pair SV wipes the subject out completely for every physical-appearance prompt.
- **Race is the hardest baseline (81.6%)**. Steering makes it worse across the board (56–70%).
- **Exp 11 holds up well on structured categories** (nationality 93%, race-color 91%, sexual-orientation 88%) and matches GT-KG tail-SV on the rest.

---

## 7. By Bias Type (bias mean; for reference — unchanged from prior run)

| Bias type | base | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|-----------|-----:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| age | 3.22 | 2.87 | 2.48 | 1.95 | 2.85 | 2.38 | 2.50 | 1.73 | 1.51 | 0.68 | 2.28 | 1.29 |
| disability | 2.83 | 2.20 | 2.43 | 2.28 | 2.89 | 2.63 | 2.26 | 2.57 | 1.89 | 0.59 | 2.48 | 2.22 |
| gender | 3.48 | 3.04 | 3.03 | 2.82 | 3.23 | 2.80 | 2.61 | 2.07 | 1.49 | 1.05 | 2.73 | 1.92 |
| nationality | 3.05 | 2.66 | 2.15 | 2.92 | 2.76 | 2.86 | 2.26 | 2.74 | 1.09 | 0.70 | 2.50 | 2.15 |
| physical-appearance | 3.45 | 2.62 | 3.62 | 2.83 | 3.69 | 2.79 | 2.88 | 3.14 | 2.36 | 0.93 | 3.14 | 1.55 |
| profession | 3.53 | 3.44 | 3.21 | 2.66 | 3.44 | 3.03 | 3.06 | 2.33 | 2.02 | 1.17 | 3.05 | 2.07 |
| race | 3.27 | 2.59 | 2.65 | 2.29 | 2.99 | 2.42 | 2.68 | 1.85 | 1.28 | 0.88 | 2.51 | 2.02 |
| race-color | 2.96 | 2.54 | 2.64 | 2.35 | 2.67 | 2.32 | 2.53 | 1.97 | 1.67 | 0.84 | 2.24 | 1.13 |
| religion | 3.68 | 2.60 | 3.31 | 3.04 | 3.54 | 3.56 | 3.07 | 2.67 | 2.31 | 1.13 | 3.49 | 2.70 |
| sexual-orientation | 2.95 | 2.60 | 2.24 | 3.16 | 2.87 | 2.71 | 2.79 | 2.80 | 1.95 | 1.43 | 2.71 | 2.76 |
| socioeconomic | 3.01 | 2.64 | 3.00 | 2.46 | 2.74 | 2.83 | 2.38 | 2.40 | 1.34 | 0.45 | 2.54 | 1.66 |

---

> **Framing note.** GT KG is the *ideal* upper bound — ground-truth annotations are not available in production. Extracted KG is the *realistic* setting the pipeline is built for. The primary method comparisons below therefore stay **within a KG source** (extracted vs extracted, GT vs GT). Cross-KG comparisons (§12) are treated as a diagnostic measuring how far the realistic setup falls from the ideal — not as a head-to-head between methods.

---

## 8. Method ranking under Extracted KG (realistic setup)

All five methods that consume extracted KG, plus the no-KG rewrite (Exp 1) and baseline (Exp 0) as references. Ranked by composite.

| Rank | Exp | Method | Bias mean | Aligned % | **Composite** | Bias on aligned |
|---:|---:|--------|----------:|----------:|----------:|----------------:|
| 1 | **5** | **tail SV** | **2.802** | 68.8% | **30.24** | 3.130 |
| 2 | **2** | **LLM rewrite** | 2.965 | **73.9%** | **30.06** | 3.225 |
| — | *1* | *no-KG rewrite (ref)* | *3.009* | *76.4%* | *30.41* | *3.262* |
| — | *0* | *baseline (ref)* | *3.391* | *87.5%* | *28.17* | *3.436* |
| 3 | 10 | GT-pair SV | 2.792 | 62.9% | 27.80 | 3.140 |
| 4 | 4 | full-triple SV | 3.204 | 74.6% | 26.81 | 3.408 |
| 5 | 8 | LLM-pair SV | 1.683 | **21.7%** | 14.37 | 2.655 |

**Findings under realistic KG conditions:**

- **Tail SV (Exp 5) and LLM rewrite (Exp 2) are in a statistical tie** at the top (composite 30.24 vs 30.06). They are also within 0.35 of the *no-KG rewrite* reference (Exp 1, 30.41) — meaning noisy extracted KG buys almost no lift over running the LLM rewriter with no KG at all.
- **GT-pair SV (Exp 10) falls below baseline composite** (27.80 vs 28.17). Building the pair SV from extracted tails poisons the direction: alignment drops 24.6 pts without a corresponding bias gain.
- **Full-triple SV (Exp 4) is the worst non-broken method** (26.81). Head+relation embeddings dominate the direction, so the stereotype tail barely moves the vector.
- **LLM-pair SV (Exp 8) is broken** (14.37, 21.7% aligned). Same α=2.0 problem as Exp 9.
- **The extracted-KG ceiling is low.** Best method clears baseline by only ~2 composite points. No realistic-setup method demonstrates strong debiasing without a matching alignment cost.

**Recommended choice under extracted KG: Exp 2 (LLM rewrite)** — essentially tied with Exp 5 on composite, higher alignment, simpler pipeline. Prefer Exp 5 only if minimizing bias is a hard priority.

---

## 9. Method ranking under GT KG (ideal / upper bound)

All five methods consuming GT KG annotations. This is the paper's upper-bound scenario.

| Rank | Exp | Method | Bias mean | Aligned % | **Composite** | Bias on aligned |
|---:|---:|--------|----------:|----------:|----------:|----------------:|
| 1 | **11** | **GT-pair SV** | 1.992 | 67.5% | **40.61** | 2.266 |
| 2 | **7** | **tail SV** | 2.180 | 71.0% | **40.07** | 2.515 |
| 3 | **3** | **LLM rewrite** | 2.597 | **80.4%** | **38.62** | 2.763 |
| 4 | 6 | full-triple SV | 2.794 | 74.7% | 32.94 | 3.029 |
| 5 | 9 | LLM-pair SV | **1.011** | **23.2%** | 18.52 | **1.822** |

**Findings under ideal KG conditions:**

- **Three-way cluster at the top** (composite 38.6–40.6): GT-pair SV, tail SV, and LLM rewrite are all viable winners. GT-pair SV edges ahead on bias reduction (−41%), tail SV matches on composite with simpler mechanics, and LLM rewrite leads on alignment preservation (80%).
- **Full-triple SV (Exp 6) is a second tier** — 7–8 composite points behind the leaders. Same dilution problem as under extracted KG; even clean head+relation embeddings don't concentrate the direction on the stereotype tail.
- **LLM-pair SV (Exp 9) is broken again.** It has the lowest bias on aligned-only images (1.82) in the whole benchmark, but only produces 1,275 aligned images out of 5,493. Not a usable operating point at α=2.0.
- **The GT-KG ceiling is meaningful.** Best method (Exp 11) is +12.4 composite points over baseline — a real benchmark ceiling that realistic setups should be measured against.

**Recommended choice under GT KG: Exp 11 (GT-pair SV)** as the peak, with **Exp 7 (tail SV)** as a near-tied simpler alternative and **Exp 3 (rewrite)** as the safest (highest alignment).

---

## 10. Best steering-vector flavor — within each KG source

Within each KG source, rank the four SV flavors only (drop rewrite and no-KG).

### Under Extracted KG

| Rank | Exp | SV flavor | Bias | Align % | **Composite** |
|---:|---:|-----------|-----:|--------:|----------:|
| 1 | **5** | **tail** | 2.80 | 68.8% | **30.24** |
| 2 | 10 | GT-pair | 2.79 | 62.9% | 27.80 |
| 3 | 4 | full-triple | 3.20 | 74.6% | 26.81 |
| 4 | 8 | LLM-pair | 1.68 | 21.7% | 14.37 |

### Under GT KG

| Rank | Exp | SV flavor | Bias | Align % | **Composite** |
|---:|---:|-----------|-----:|--------:|----------:|
| 1 | **11** | **GT-pair** | 1.99 | 67.5% | **40.61** |
| 2 | 7 | tail | 2.18 | 71.0% | 40.07 |
| 3 | 6 | full-triple | 2.79 | 74.7% | 32.94 |
| 4 | 9 | LLM-pair | 1.01 | 23.2% | 18.52 |

**Findings:**

- **Tail SV is the only flavor that wins in one setting and places 2nd in the other.** It's the most rank-stable SV flavor: #1 under extracted KG (30.24) and #2 under GT KG (40.07). The mean-pool-and-broadcast construction produces a short, concentrated direction that is relatively tolerant of KG noise.
- **GT-pair SV has the highest peak but drops the most under noisy KG.** #1 under GT (40.61) but falls to #2 under extracted (27.80, 12.8 composite points below its GT peak). The double KG dependency (both anti- and stereo-side) makes it the most KG-sensitive method.
- **Full-triple SV is #3 in both settings** and never comes close to the winners. Safe choice if alignment matters more than bias, but it's dominated by tail SV on composite.
- **LLM-pair SV is #4 in both settings.** Always broken at α=2.0, regardless of KG quality. The α-sweep caveat from §12 is a necessary prerequisite to treat it as a real candidate.

**SV recommendation by KG setting:**
- Extracted KG → **tail SV (Exp 5)**.
- GT KG → **GT-pair SV (Exp 11)** for peak; **tail SV (Exp 7)** if you want simpler mechanics and graceful degradation if the KG later turns out noisy.

### Sub-ablation: GT-tail pair vs LLM-generated pair (pair-SV flavors)

Both pair-SV variants are run under each KG source, giving 2 head-to-head pairs.

| KG source | GT-tail pair (exp) | LLM-generated pair (exp) | Δ bias (LLM−GT) | Δ aligned % | Δ composite |
|-----------|:------------------:|:------------------------:|----------------:|------------:|------------:|
| Extracted | **10: 2.79 / 62.9% / 27.8** | 8: 1.68 / 21.7% / 14.4 | −1.11 | **−41.3 pts** | **−13.4** |
| GT | **11: 1.99 / 67.5% / 40.6** | 9: 1.01 / 23.2% / 18.5 | −0.98 | **−44.3 pts** | **−22.1** |

Under both KG sources, LLM-paraphrased pairs cut bias an extra ~1 point but sacrifice ~42–44 pts of alignment. Cause: the LLM writes contrastive *sentences*, whose embedding-space separation is much larger than bare-tail separation, so α=2.0 overshoots. **GT-tail pairs strictly dominate LLM pairs at α=2.0 under both KG sources.** An α sweep on Exp 8/9 is needed before LLM pairs can be treated as viable.

---

## 11. Prompt rewrite vs Steering vector — within each KG source

| KG source | Rewrite (exp) | Best SV (exp) | Δ composite (SV − rewrite) | Verdict |
|-----------|:-------------:|:-------------:|---------------------------:|---------|
| **Extracted** | 2: 30.06 | Exp 5 tail (30.24) | **+0.18** | **Statistical tie** — use rewrite for simplicity |
| **GT** | 3: 38.62 | Exp 11 GT-pair (40.61) | **+1.99** | **SV wins, narrowly** — rewrite is still competitive |
| None (no-KG) | 1: 30.41 | — | N/A | Rewrite is the only option |

**Findings:**

- **Under extracted KG, rewrite and the best SV are tied.** SVs are not worth the added machinery when the KG is noisy — they cost alignment for no composite gain.
- **Under GT KG, SV is the peak by ~2 composite points**, but rewrite is a close third (38.62) and has the best alignment of any intervention (80.4%). "Which wins" depends on whether you care about peak composite or alignment robustness.
- **Rewrite has much tighter variance across KG quality.** Its composite only moves from 30.06 (extracted) → 38.62 (GT), a 8.6-point spread. Best SV moves from 30.24 → 40.61, a 10.4-point spread. Rewrite is the more predictable method when KG quality is unknown.

---

## 12. Diagnostic — KG quality gap (extracted → GT)

This is a **diagnostic** of how much an imperfect KG costs, *not* a method ranking. GT is assumed optimal by construction; the gap measures pipeline headroom.

| Method family | Extracted | GT (ideal) | Composite gap (ideal − realistic) | Bias gap | Align gap |
|---------------|:---------:|:----------:|----------------------------------:|---------:|----------:|
| LLM rewrite | 2: 30.06 | 3: 38.62 | **+8.55** | −0.37 | +6.48 pts |
| Full-triple SV | 4: 26.81 | 6: 32.94 | +6.13 | −0.41 | +0.04 pts |
| Tail SV | 5: 30.24 | 7: 40.07 | **+9.83** | −0.62 | +2.26 pts |
| LLM-pair SV | 8: 14.37 | 9: 18.52 | +4.15 | −0.67 | +1.55 pts |
| **GT-pair SV** | 10: 27.80 | 11: 40.61 | **+12.82** | **−0.80** | **+4.57 pts** |

**Interpretation:** the size of the gap tells us how much each family's performance depends on clean KG annotations.

- **GT-pair SV has the largest gap (+12.82 composite).** It is the most KG-sensitive method — a double dependency (both anti- and stereo-side use the KG tail directly), so noise compounds.
- **Tail SV: +9.83 composite gap.** Second most KG-sensitive despite being the most *rank-stable*. The gap is large because the peak under GT is large; the extracted floor is still competitive.
- **LLM rewrite: +8.55 composite gap.** The KG enters the rewrite as a soft prompt, so noise hurts less than in direct embedding-based methods — still a meaningful gap.
- **Full-triple SV: +6.13 composite gap.** Small gap, but for the wrong reason — both variants are weak, so cleaning the KG doesn't unlock much headroom.
- **LLM-pair SV: +4.15 composite gap.** Smallest gap; the LLM paraphraser absorbs most KG noise when it writes the contrastive sentences. Both variants are broken anyway, so the gap is moot.

**Takeaway for the benchmark:** the realistic (extracted KG) results are, across all method families, 4–13 composite points below the ideal. The pipeline has real headroom from KG-extraction improvements. GT-pair SV in particular looks strong only under GT; if the extraction pipeline were perfect, it would jump from #3 to #1 on the extracted ranking.

---

## 13. Key Findings

Written within-KG: what does the benchmark say under the realistic (extracted) setting, what does it say under the ideal (GT) upper bound, and how large is the gap between them?

### A. Under Extracted KG (realistic setup)

1. **Tail SV (Exp 5) and LLM rewrite (Exp 2) are tied at the top** (composite 30.24 and 30.06, respectively). Both only beat baseline by ~2 composite points.
2. **The no-KG rewrite (Exp 1, 30.41) is within noise of the best extracted-KG method.** Running the extraction pipeline without a quality bar does not pay off — a noisy KG is no better than no KG.
3. **GT-pair SV (Exp 10, 27.80) and full-triple SV (Exp 4, 26.81) fall *below* baseline composite.** Under realistic KG quality, these methods lose alignment faster than they cut bias.
4. **LLM-pair SV (Exp 8) is broken** (21.7% aligned). α=2.0 is too aggressive for sentence-pair directions regardless of KG source.
5. **Practical recommendation under extracted KG: Exp 2 (LLM rewrite).** Tied with Exp 5 on composite, better alignment (74% vs 69%), simpler pipeline.

### B. Under GT KG (ideal / upper bound)

6. **GT-pair SV (Exp 11, 40.61), tail SV (Exp 7, 40.07), and LLM rewrite (Exp 3, 38.62) form a 3-way cluster at the top** — all ~12 composite points over baseline (28.17).
7. **GT-pair SV wins on peak composite**, but tail SV is essentially tied and simpler (no pair generation step). Rewrite is 2 points behind on composite but has the highest alignment retention (80.4%) of any intervention.
8. **Full-triple SV (Exp 6) stays mid-pack (32.94)** even with ideal KG. The head+relation embeddings dilute the tail signal; this method has a ceiling that clean KG does not lift.
9. **LLM-pair SV (Exp 9) is broken under GT KG too** — 23.2% aligned. Same α=2.0 overshoot as Exp 8.

### C. KG-quality gap (ideal − realistic) — diagnostic

10. **GT-pair SV has the largest gap (+12.82 composite).** The double KG dependency (anti- and stereo-side both built from tails) makes it the most KG-sensitive family. This means: GT-pair SV is the biggest winner under GT but would drop sharply under extracted KG.
11. **Tail SV gap +9.83; rewrite gap +8.55; full-triple gap +6.13; LLM-pair gap +4.15** (though LLM-pair is broken at both ends so the gap is moot). Cross-family pattern: methods that depend directly on raw KG tails (tail SV, GT-pair SV) are the most KG-quality-sensitive; methods that pass the KG through an LLM rewriter (LLM rewrite, LLM-pair SV) absorb more of the noise.
12. **Overall headroom:** across method families, realistic-setup composite sits 4–13 points below the ideal. Improving the KG-extraction pipeline's fidelity is the single biggest lever for raising benchmark performance.

### D. Cross-cutting

13. **Bias-only rankings are misleading.** Exp 8 and Exp 9 topped the bias-only leaderboard at −50% and −70% respectively, but both have 22–23% alignment — ~2/3 of their images no longer depict the prompted subject. Once alignment is folded in (via composite or bias-on-aligned), they fall to last place.
14. **Alignment is rarely improved by any intervention.** Recovery rates ("misaligned → aligned") sit at 1–5% across all 11 non-baseline methods, while loss rates ("aligned → misaligned") range 12–67%. Debiasing is additive noise on the alignment axis — no method reliably repairs baseline mis-generation.
15. **The paper's headline ranking differs between realistic and ideal settings.** Under extracted KG the winner is rewrite/tail SV (both ~30 composite); under GT KG the winner is GT-pair SV (40.6 composite). That gap is the main experimental result.

---

## 14. Recommendation

Organized by the paper's two experimental settings.

### Realistic setup (extracted KG only — the deployable pipeline)

| Use case | Method | Composite | Rationale |
|----------|--------|----------:|-----------|
| **Default** | **Exp 2 (extracted_kg_llm_rewrite)** | 30.06 | Tied with tail-SV on composite, higher alignment (74%), simpler machinery (no embedding steering) |
| Bias-priority | **Exp 5 (extracted_kg_tail_sv)** | 30.24 | Lowest bias in realistic setup (2.80); robust to KG noise; mean-pool+broadcast is trivial |
| Safe fallback | **Exp 1 (llm_rewrite_no_kg)** | 30.41 | Only marginally different from Exp 2; the KG adds no value when noisy |

### Ideal upper bound (GT KG — what the benchmark ceiling looks like)

| Use case | Method | Composite | Rationale |
|----------|--------|----------:|-----------|
| **Peak** | **Exp 11 (gt_kg_gt_pair_sv)** | 40.61 | Highest composite in the benchmark; 41% bias reduction at 68% alignment |
| Near-tie (simpler) | **Exp 7 (gt_kg_tail_sv)** | 40.07 | Within noise of Exp 11; no pair generation needed; rank-stable under extracted KG too |
| Fidelity-first | **Exp 3 (gt_kg_llm_rewrite)** | 38.62 | Highest alignment of any intervention (80.4%); 23% bias cut |

### Avoid in both settings

| Method | Reason |
|--------|--------|
| **Exp 8 / Exp 9 (LLM-pair SV)** | 21–23% alignment. Bias reduction is largely content loss, not debiasing. Needs an α sweep before reconsidering |
| **Exp 4 (extracted full-triple SV)** | Composite 26.8 — *below* baseline in realistic setup |
| **Exp 10 (extracted GT-pair SV)** | Composite 27.8 — *below* baseline; the method's KG sensitivity shows up as net-negative under extracted KG |

### Headline for the paper

- **Realistic setup winner:** Exp 2 (LLM rewrite) ≈ Exp 5 (tail SV) at composite ~30.1–30.2.
- **Ideal upper bound:** Exp 11 (GT-pair SV) at composite 40.6.
- **Headroom from KG-quality improvement:** ~10 composite points — the largest single source of benchmark gains.

---

## 15. Raw Files

- Bias scores: `cache/eval_results/exp_NN_eval.csv` / `.jsonl` (5,493 rows per exp)
- Alignment verdicts: `cache/eval_results/exp_NN_alignment.csv` / `.jsonl` (5,493 rows per exp)
- Input manifest: `data/benchmark_prompts.csv` (1,831 cases)
- Generation config: `experiments/config.py`
- Bias evaluator: `experiments/evaluate_all.py`
- Alignment evaluator: `experiments/evaluate_alignment.py`
