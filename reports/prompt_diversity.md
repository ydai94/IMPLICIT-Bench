# Prompt Diversity Analysis

Source: `data/benchmark_prompts.csv` -- 1831 prompt units (StereoSet 1393 + CrowS-Pairs 438). Each unit contributes three prompts (neutral, stereotype-trigger, anti-stereotype-trigger), so the prompt corpus has 5493 entries.

Tokenisation is whitespace + punctuation (regex `[A-Za-z][A-Za-z\-']*`), lowercased. "Content tokens" drop a small built-in stopword list and tokens of length 1.

## 1. Token-length and lexical diversity, per arm

| Arm | Mean length | Std | Min | p25 | p50 | p75 | Max | Total tokens | Unique tokens | TTR | Unique content | TTR (content) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 8.67 | 2.55 | 2 | 7 | 9 | 10 | 19 | 15877 | 2549 | 0.1605 | 2453 | 0.2591 |
| stereotype-trigger | 9.80 | 3.15 | 3 | 8 | 10 | 12 | 24 | 17944 | 3279 | 0.1827 | 3172 | 0.2898 |
| anti-stereotype-trigger | 9.94 | 3.32 | 3 | 8 | 10 | 12 | 27 | 18209 | 3359 | 0.1845 | 3251 | 0.2935 |
| all arms | 9.47 | 3.08 | 2 | 7 | 9 | 11 | 27 | 52030 | 4179 | 0.0803 | 4068 | 0.1292 |

TTR is the type-token ratio (unique tokens / total tokens). For comparison: a fully template-instantiated dataset of the form `"a {adj} {noun} doing {verb}"` would converge to TTR << 0.05; natural language English text typically sits between 0.05 and 0.20 depending on length. The IMPLICIT-Bench prompts are LLM-generated from KG triples, not template-instantiated.

## 2. Top content tokens (all arms pooled, top 50)

| # | Token | Count | # | Token | Count |
| --- | --- | --- | --- | --- | --- |
| 1 | person | 684 | 26 | nurse | 98 |
| 2 | man | 547 | 27 | beside | 95 |
| 3 | people | 497 | 28 | chess | 94 |
| 4 | described | 494 | 29 | mathematician | 90 |
| 5 | woman | 325 | 30 | researcher | 90 |
| 6 | standing | 269 | 31 | physicist | 90 |
| 7 | describing | 260 | 32 | historian | 88 |
| 8 | room | 205 | 33 | manager | 88 |
| 9 | living | 185 | 34 | commander | 87 |
| 10 | work | 165 | 35 | job | 87 |
| 11 | table | 152 | 36 | speaking | 86 |
| 12 | office | 141 | 37 | poor | 86 |
| 13 | player | 134 | 38 | terms | 83 |
| 14 | kitchen | 130 | 39 | school | 83 |
| 15 | male | 126 | 40 | black | 83 |
| 16 | looking | 125 | 41 | street | 82 |
| 17 | portrait | 124 | 42 | child | 82 |
| 18 | working | 115 | 43 | chemist | 81 |
| 19 | psychologist | 108 | 44 | shown | 80 |
| 20 | white | 108 | 45 | sitting | 79 |
| 21 | performing | 106 | 46 | wearing | 79 |
| 22 | interacting | 105 | 47 | social | 78 |
| 23 | mother | 104 | 48 | about | 78 |
| 24 | personal | 99 | 49 | grandfather | 76 |
| 25 | mommy | 98 | 50 | new | 76 |

## 3. Scene / object vocabulary (prompt_neutral, top 50 nouns)

Approximate noun-like vocabulary on the neutral arm only -- captures the variety of subjects and settings the benchmark depicts before any stereotype trigger is applied. Heuristic: lowercase content tokens not in a small list of common adjectives/verbs/adverbs. This is a lower bound; some adjectives slip through.

| # | Token | Count | # | Token | Count |
| --- | --- | --- | --- | --- | --- |
| 1 | person | 364 | 26 | nurse | 31 |
| 2 | people | 166 | 27 | mathematician | 30 |
| 3 | man | 110 | 28 | neutral | 30 |
| 4 | standing | 92 | 29 | researcher | 30 |
| 5 | room | 68 | 30 | physicist | 30 |
| 6 | living | 62 | 31 | personality | 30 |
| 7 | personal | 51 | 32 | historian | 30 |
| 8 | portrait | 50 | 33 | child | 30 |
| 9 | table | 50 | 34 | street | 29 |
| 10 | terms | 47 | 35 | commander | 29 |
| 11 | woman | 46 | 36 | about | 29 |
| 12 | office | 46 | 37 | manager | 29 |
| 13 | player | 44 | 38 | professional | 28 |
| 14 | kitchen | 43 | 39 | job | 28 |
| 15 | interacting | 38 | 40 | speaking | 28 |
| 16 | performing | 36 | 41 | natural | 27 |
| 17 | psychologist | 36 | 42 | thoughtful | 27 |
| 18 | mother | 34 | 43 | chemist | 27 |
| 19 | mommy | 32 | 44 | demeanor | 26 |
| 20 | appearance | 31 | 45 | simple | 26 |
| 21 | social | 31 | 46 | evaluated | 26 |
| 22 | chess | 31 | 47 | school | 26 |
| 23 | beside | 31 | 48 | sitting | 25 |
| 24 | toward | 31 | 49 | conditions | 25 |
| 25 | condition | 31 | 50 | desk | 25 |

Unique noun-like content tokens on `prompt_neutral`: **2389** across 8706 occurrences (content TTR = 0.2744). For a 1,831-prompt corpus this is a strong indicator that scenes are not template-instantiated.

## 4. Per-bias-type breakdown (all three arms pooled)

| Bias type | Units | Prompts | Mean length | p50 | Total tokens | Unique tokens | TTR | TTR (content) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| profession | 698 | 2094 | 9.06 | 8 | 18982 | 2049 | 0.1079 | 0.1739 |
| race | 428 | 1284 | 8.29 | 8 | 10645 | 1453 | 0.1365 | 0.1985 |
| gender | 370 | 1110 | 9.97 | 10 | 11072 | 1416 | 0.1279 | 0.2057 |
| socioeconomic | 73 | 219 | 11.86 | 12 | 2598 | 380 | 0.1463 | 0.2220 |
| religion | 67 | 201 | 9.63 | 9 | 1936 | 413 | 0.2133 | 0.3086 |
| race-color | 66 | 198 | 11.21 | 11 | 2220 | 304 | 0.1369 | 0.2056 |
| age | 37 | 111 | 11.59 | 12 | 1287 | 211 | 0.1639 | 0.2444 |
| nationality | 35 | 105 | 12.07 | 12 | 1267 | 217 | 0.1713 | 0.2571 |
| sexual-orientation | 25 | 75 | 11.63 | 12 | 872 | 138 | 0.1583 | 0.2362 |
| disability | 18 | 54 | 12.74 | 12 | 688 | 143 | 0.2078 | 0.3041 |
| physical-appearance | 14 | 42 | 11.02 | 11 | 463 | 96 | 0.2073 | 0.2888 |

TTR is computed within each bias-type sub-corpus, so smaller bias types (e.g. disability, physical-appearance) naturally have higher TTR -- fewer total tokens, less repetition. The point is that no category collapses to template-like values (TTR << 0.05).

## 5. Source breakdown (StereoSet vs CrowS-Pairs)

| Source | Units | Mean length | Std | p50 | Unique tokens | TTR |
| --- | --- | --- | --- | --- | --- | --- |
| CrowS-Pairs | 438 | 11.43 | 1.87 | 11 | 1349 | 0.0898 |
| StereoSet | 1393 | 8.86 | 3.13 | 8 | 3455 | 0.0934 |

