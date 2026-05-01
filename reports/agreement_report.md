# LLM Prompt Labeling: Agreement Report

## Matching Approach Results

Each LLM saw all 3 prompts together (shuffled) and assigned labels.

### Per-Model Accuracy

| Model | Row Accuracy | Prompt Accuracy (N) | Prompt Accuracy (S) | Prompt Accuracy (A) | Overall Prompt Acc |
|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 0.8568 | 0.9869 | 0.8617 | 0.8617 | 0.9034 |
| Qwen3-30B | 0.5375 | 0.7781 | 0.6005 | 0.5879 | 0.6555 |
| Gemma4-26B | 0.7321 | 0.9049 | 0.7622 | 0.7671 | 0.8114 |
| Llama4-Maverick | 0.7386 | 0.9427 | 0.7510 | 0.7650 | 0.8196 |

### Confusion Matrices

#### Claude Sonnet 4.6

| Pred \ True | neutral | stereotype | anti-stereotype |
|---|---|---|---|
| neutral | 1805 | 12 | 12 |
| stereotype | 12 | 1576 | 241 |
| anti-stereotype | 12 | 241 | 1576 |

#### Qwen3-30B

| Pred \ True | neutral | stereotype | anti-stereotype |
|---|---|---|---|
| neutral | 1420 | 151 | 254 |
| stereotype | 231 | 1096 | 498 |
| anti-stereotype | 174 | 578 | 1073 |

#### Gemma4-26B

| Pred \ True | neutral | stereotype | anti-stereotype |
|---|---|---|---|
| neutral | 1655 | 89 | 85 |
| stereotype | 94 | 1394 | 341 |
| anti-stereotype | 80 | 346 | 1403 |

#### Llama4-Maverick

| Pred \ True | neutral | stereotype | anti-stereotype |
|---|---|---|---|
| neutral | 1677 | 56 | 46 |
| stereotype | 71 | 1336 | 372 |
| anti-stereotype | 31 | 387 | 1361 |


### Inter-Model Agreement

Units with all 4 models responding: 5325 / 5493

**Fleiss' Kappa**: 0.6539

**Krippendorff's Alpha**: 0.6539

**Pairwise Cohen's Kappa**:

| Model A | Model B | Cohen's Kappa |
|---|---|---|
| Claude Sonnet 4.6 | Qwen3-30B | 0.5014 |
| Claude Sonnet 4.6 | Gemma4-26B | 0.7611 |
| Claude Sonnet 4.6 | Llama4-Maverick | 0.7693 |
| Qwen3-30B | Gemma4-26B | 0.5800 |
| Qwen3-30B | Llama4-Maverick | 0.5659 |
| Gemma4-26B | Llama4-Maverick | 0.7459 |


---

## Independent Approach Results

Each LLM classified each prompt individually without seeing siblings.

### Per-Model Accuracy

| Model | Neutral Acc | Stereotype Acc | Anti-Stereo Acc | Overall Prompt Acc |
|---|---|---|---|---|
| Claude Sonnet 4.6 | 0.8869 | 0.7914 | 0.7733 | 0.8172 |
| Qwen3-30B | 0.8580 | 0.5434 | 0.6264 | 0.6760 |
| Gemma4-26B | 0.8706 | 0.6865 | 0.6226 | 0.7266 |
| Llama4-Maverick | 0.7406 | 0.6499 | 0.6537 | 0.6814 |

### Confusion Matrices

#### Claude Sonnet 4.6

| Pred \ True | neutral | stereotype | anti-stereotype |
|---|---|---|---|
| neutral | 1624 | 147 | 60 |
| stereotype | 227 | 1449 | 155 |
| anti-stereotype | 246 | 169 | 1416 |

#### Qwen3-30B

| Pred \ True | neutral | stereotype | anti-stereotype |
|---|---|---|---|
| neutral | 1571 | 217 | 43 |
| stereotype | 633 | 995 | 203 |
| anti-stereotype | 377 | 307 | 1147 |

#### Gemma4-26B

| Pred \ True | neutral | stereotype | anti-stereotype |
|---|---|---|---|
| neutral | 1594 | 159 | 78 |
| stereotype | 517 | 1257 | 57 |
| anti-stereotype | 458 | 233 | 1140 |

#### Llama4-Maverick

| Pred \ True | neutral | stereotype | anti-stereotype |
|---|---|---|---|
| neutral | 1356 | 321 | 154 |
| stereotype | 497 | 1190 | 144 |
| anti-stereotype | 355 | 279 | 1197 |


### Inter-Model Agreement

Units with all 4 models responding: 5493 / 5493

**Fleiss' Kappa**: 0.6429

**Krippendorff's Alpha**: 0.6429

**Pairwise Cohen's Kappa**:

| Model A | Model B | Cohen's Kappa |
|---|---|---|
| Claude Sonnet 4.6 | Qwen3-30B | 0.5868 |
| Claude Sonnet 4.6 | Gemma4-26B | 0.6842 |
| Claude Sonnet 4.6 | Llama4-Maverick | 0.6285 |
| Qwen3-30B | Gemma4-26B | 0.6535 |
| Qwen3-30B | Llama4-Maverick | 0.6373 |
| Gemma4-26B | Llama4-Maverick | 0.6703 |


---

## Approach Comparison

| Model | Matching Acc | Independent Acc |
|---|---|---|
| Claude Sonnet 4.6 | 0.8568 | 0.8172 |
| Qwen3-30B | 0.5375 | 0.6760 |
| Gemma4-26B | 0.7321 | 0.7266 |
| Llama4-Maverick | 0.7386 | 0.6814 |

---

## Agreement Excluding Qwen3-30B

Qwen3-30B has notably lower row accuracy in the matching approach (~0.54), so we
recompute inter-model agreement using only Claude Sonnet 4.6, Gemma4-26B, and
Llama4-Maverick.

### Matching Approach (3 raters)

Units with all 3 models responding: 5334 / 5493

**Fleiss' Kappa**: 0.7588

**Krippendorff's Alpha**: 0.7588

**Pairwise Cohen's Kappa**:

| Model A | Model B | Cohen's Kappa |
|---|---|---|
| Claude Sonnet 4.6 | Gemma4-26B | 0.7615 |
| Claude Sonnet 4.6 | Llama4-Maverick | 0.7691 |
| Gemma4-26B | Llama4-Maverick | 0.7458 |

### Independent Approach (3 raters)

Units with all 3 models responding: 5493 / 5493

**Fleiss' Kappa**: 0.6603

**Krippendorff's Alpha**: 0.6603

**Pairwise Cohen's Kappa**:

| Model A | Model B | Cohen's Kappa |
|---|---|---|
| Claude Sonnet 4.6 | Gemma4-26B | 0.6842 |
| Claude Sonnet 4.6 | Llama4-Maverick | 0.6285 |
| Gemma4-26B | Llama4-Maverick | 0.6703 |

### Notes

- Dropping Qwen3 raises matching Fleiss' κ from 0.6539 → 0.7588 (+0.10), confirming
  Qwen3 is an outlier in the matching task.
- For the independent task, Fleiss' κ barely moves (0.6429 → 0.6603); the remaining
  three models disagree among themselves nearly as much as they disagree with Qwen3,
  so the lower agreement reflects task difficulty rather than a single weak rater.