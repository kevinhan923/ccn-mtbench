# §7 analysis (metric = xcomet_d, Δ = clean − noisy)

Systems: aya-101, aya-expanse-8b, deepseek, gemini, gemma3-12b, gemma3-27b, gemma3-4b, gemmax2-9b, google-translate, gpt-4o, hunyuan-mt-7b, nllb-3.3b, qwen3-0.6b, qwen3-1.7b, qwen3-14b, qwen3-32b, qwen3-4b, qwen3-8b

## RQ1 — key system pairs (Δ_A − Δ_B; CI excluding 0 = robustness gap is real)

| A | B | diff | 95% CI | p_boot |
|---|---|---|---|---|
| nllb-3.3b | qwen3-8b | 3.27 | [0.71, 5.75] | 0.015 |
| qwen3-8b | gpt-4o | 1.55 | [0.40, 2.72] | 0.005 |
| nllb-3.3b | gpt-4o | 4.81 | [2.39, 7.25] | 0.000 |
| qwen3-8b | qwen3-32b | 1.08 | [-0.11, 2.35] | 0.072 |

## RQ2 — regression Δ ~ category + z(length) + z(edit_distance)

Category coefficients are vs the baseline category; `excludes_zero=yes` means the 95% bootstrap CI is one-sided (effect is credible).
Cross-category *paired* comparison is deliberately NOT done — categories fall on different sentence sets; the covariate-controlled regression is the RQ2 evidence.

| model | term | estimate | 95% CI | excl. 0 |
|---|---|---|---|---|
| aya-101 | intercept | 16.55 | [6.82, 29.19] | yes |
| aya-101 | cat=PYA | -6.74 | [-20.62, 5.22] |  |
| aya-101 | cat=NEO | -8.63 | [-21.71, 1.54] |  |
| aya-101 | cat=MIX | -15.35 | [-29.04, -3.64] | yes |
| aya-101 | length_z | -2.05 | [-3.51, -0.55] | yes |
| aya-101 | editdist_z | 1.42 | [-0.67, 3.70] |  |
| aya-expanse-8b | intercept | 15.66 | [6.32, 26.22] | yes |
| aya-expanse-8b | cat=PYA | -1.98 | [-19.56, 13.74] |  |
| aya-expanse-8b | cat=NEO | -4.78 | [-15.75, 5.01] |  |
| aya-expanse-8b | cat=MIX | -10.81 | [-23.03, -0.10] | yes |
| aya-expanse-8b | length_z | -5.69 | [-8.12, -3.82] | yes |
| aya-expanse-8b | editdist_z | 1.92 | [-0.16, 3.84] |  |
| deepseek | intercept | 11.37 | [6.99, 16.01] | yes |
| deepseek | cat=PYA | -6.08 | [-12.51, 0.80] |  |
| deepseek | cat=NEO | -5.80 | [-10.58, -1.21] | yes |
| deepseek | cat=MIX | -7.93 | [-13.49, -2.09] | yes |
| deepseek | length_z | -4.18 | [-5.76, -3.00] | yes |
| deepseek | editdist_z | 2.37 | [0.90, 3.77] | yes |
| gemini | intercept | 10.18 | [3.80, 17.38] | yes |
| gemini | cat=PYA | -1.10 | [-10.32, 6.52] |  |
| gemini | cat=NEO | -4.48 | [-12.08, 2.18] |  |
| gemini | cat=MIX | -5.06 | [-13.62, 2.57] |  |
| gemini | length_z | -3.65 | [-5.22, -2.38] | yes |
| gemini | editdist_z | 1.99 | [0.47, 3.44] | yes |
| gemma3-12b | intercept | 9.86 | [3.08, 17.15] | yes |
| gemma3-12b | cat=PYA | -0.25 | [-11.42, 11.40] |  |
| gemma3-12b | cat=NEO | -1.63 | [-9.07, 5.31] |  |
| gemma3-12b | cat=MIX | -6.11 | [-14.16, 1.58] |  |
| gemma3-12b | length_z | -3.21 | [-5.01, -1.67] | yes |
| gemma3-12b | editdist_z | 2.77 | [0.90, 4.41] | yes |
| gemma3-27b | intercept | 16.11 | [9.92, 22.65] | yes |
| gemma3-27b | cat=PYA | -3.65 | [-15.51, 10.23] |  |
| gemma3-27b | cat=NEO | -6.46 | [-13.29, -0.12] | yes |
| gemma3-27b | cat=MIX | -14.92 | [-24.49, -6.15] | yes |
| gemma3-27b | length_z | -3.74 | [-6.13, -1.97] | yes |
| gemma3-27b | editdist_z | 4.47 | [2.40, 6.67] | yes |
| gemma3-4b | intercept | 13.96 | [5.51, 21.98] | yes |
| gemma3-4b | cat=PYA | -4.81 | [-16.70, 7.59] |  |
| gemma3-4b | cat=NEO | -4.80 | [-13.30, 3.63] |  |
| gemma3-4b | cat=MIX | -12.02 | [-21.64, -1.99] | yes |
| gemma3-4b | length_z | -3.74 | [-6.18, -1.90] | yes |
| gemma3-4b | editdist_z | 3.58 | [1.45, 5.64] | yes |
| gemmax2-9b | intercept | 16.33 | [10.76, 22.47] | yes |
| gemmax2-9b | cat=PYA | -9.87 | [-17.12, -2.13] | yes |
| gemmax2-9b | cat=NEO | -9.08 | [-15.25, -3.03] | yes |
| gemmax2-9b | cat=MIX | -12.63 | [-19.97, -5.63] | yes |
| gemmax2-9b | length_z | -3.35 | [-4.97, -2.18] | yes |
| gemmax2-9b | editdist_z | 3.06 | [1.47, 4.87] | yes |
| google-translate | intercept | 14.70 | [8.86, 21.44] | yes |
| google-translate | cat=PYA | -9.20 | [-17.29, -1.23] | yes |
| google-translate | cat=NEO | -7.76 | [-14.33, -1.71] | yes |
| google-translate | cat=MIX | -11.21 | [-18.64, -3.89] | yes |
| google-translate | length_z | -2.96 | [-4.73, -1.50] | yes |
| google-translate | editdist_z | 3.38 | [2.06, 4.94] | yes |
| gpt-4o | intercept | 12.31 | [6.74, 19.40] | yes |
| gpt-4o | cat=PYA | -2.84 | [-10.63, 4.75] |  |
| gpt-4o | cat=NEO | -6.13 | [-13.03, -0.26] | yes |
| gpt-4o | cat=MIX | -8.81 | [-16.40, -2.03] | yes |
| gpt-4o | length_z | -3.69 | [-5.25, -2.55] | yes |
| gpt-4o | editdist_z | 2.71 | [1.47, 3.99] | yes |
| hunyuan-mt-7b | intercept | 11.50 | [5.53, 18.29] | yes |
| hunyuan-mt-7b | cat=PYA | 0.17 | [-9.00, 10.41] |  |
| hunyuan-mt-7b | cat=NEO | -3.73 | [-10.58, 2.58] |  |
| hunyuan-mt-7b | cat=MIX | -6.44 | [-14.18, 0.90] |  |
| hunyuan-mt-7b | length_z | -4.60 | [-6.20, -3.23] | yes |
| hunyuan-mt-7b | editdist_z | 2.23 | [0.50, 4.04] | yes |
| nllb-3.3b | intercept | 11.51 | [-4.34, 25.49] |  |
| nllb-3.3b | cat=PYA | 0.85 | [-14.55, 19.90] |  |
| nllb-3.3b | cat=NEO | 1.26 | [-13.32, 17.94] |  |
| nllb-3.3b | cat=MIX | -8.08 | [-23.46, 9.55] |  |
| nllb-3.3b | length_z | -3.95 | [-5.97, -2.07] | yes |
| nllb-3.3b | editdist_z | 4.26 | [0.94, 7.54] | yes |
| qwen3-0.6b | intercept | 11.50 | [4.16, 19.33] | yes |
| qwen3-0.6b | cat=PYA | -7.43 | [-16.87, 1.03] |  |
| qwen3-0.6b | cat=NEO | -4.08 | [-12.02, 4.06] |  |
| qwen3-0.6b | cat=MIX | -6.36 | [-16.05, 3.33] |  |
| qwen3-0.6b | length_z | -3.72 | [-5.73, -2.20] | yes |
| qwen3-0.6b | editdist_z | 2.52 | [0.48, 4.75] | yes |
| qwen3-1.7b | intercept | 14.25 | [8.24, 21.80] | yes |
| qwen3-1.7b | cat=PYA | -6.96 | [-15.38, 1.06] |  |
| qwen3-1.7b | cat=NEO | -7.70 | [-15.57, -1.64] | yes |
| qwen3-1.7b | cat=MIX | -12.00 | [-20.08, -4.22] | yes |
| qwen3-1.7b | length_z | -3.28 | [-4.95, -1.73] | yes |
| qwen3-1.7b | editdist_z | 3.25 | [1.56, 5.14] | yes |
| qwen3-14b | intercept | 14.57 | [9.48, 20.91] | yes |
| qwen3-14b | cat=PYA | -7.06 | [-15.80, 3.24] |  |
| qwen3-14b | cat=NEO | -7.20 | [-13.88, -1.87] | yes |
| qwen3-14b | cat=MIX | -10.65 | [-17.80, -4.21] | yes |
| qwen3-14b | length_z | -3.22 | [-5.10, -1.78] | yes |
| qwen3-14b | editdist_z | 3.46 | [1.89, 5.14] | yes |
| qwen3-32b | intercept | 15.25 | [8.41, 23.10] | yes |
| qwen3-32b | cat=PYA | -8.16 | [-17.20, 0.07] |  |
| qwen3-32b | cat=NEO | -8.64 | [-16.16, -1.74] | yes |
| qwen3-32b | cat=MIX | -11.05 | [-19.48, -3.90] | yes |
| qwen3-32b | length_z | -3.53 | [-5.25, -2.13] | yes |
| qwen3-32b | editdist_z | 3.12 | [1.62, 4.54] | yes |
| qwen3-4b | intercept | 13.18 | [6.27, 21.32] | yes |
| qwen3-4b | cat=PYA | -4.15 | [-14.00, 5.41] |  |
| qwen3-4b | cat=NEO | -4.81 | [-12.86, 2.37] |  |
| qwen3-4b | cat=MIX | -11.50 | [-20.80, -3.46] | yes |
| qwen3-4b | length_z | -3.15 | [-5.16, -1.50] | yes |
| qwen3-4b | editdist_z | 4.07 | [1.95, 6.20] | yes |
| qwen3-8b | intercept | 12.28 | [5.86, 19.63] | yes |
| qwen3-8b | cat=PYA | -3.70 | [-12.88, 5.93] |  |
| qwen3-8b | cat=NEO | -3.75 | [-11.25, 2.97] |  |
| qwen3-8b | cat=MIX | -9.25 | [-17.51, -1.73] | yes |
| qwen3-8b | length_z | -3.05 | [-4.99, -1.55] | yes |
| qwen3-8b | editdist_z | 3.86 | [2.22, 5.63] | yes |

## Twin-edit magnitude by category (Limitations covariate)

| category | n | mean edit dist | mean len |
|---|---|---|---|
| HOM | 17 | 1.88 | 12.59 |
| PYA | 15 | 3.73 | 13.53 |
| NEO | 191 | 4.69 | 15.51 |
| MIX | 44 | 6.11 | 21.09 |
| all | 267 | 4.69 | 16.13 |

## §7.5 metric-failure — 2091 items flagged for human check (see flagged_<model>.tsv)

## §7.3 contamination (table A vs B, gap = Δ_A − Δ_B; positive = A inflated)

| model | category | Δ_A | Δ_B | gap |
|---|---|---|---|---|
| aya-101 | HOM | 15.86 | 11.72 | 4.14 |
| aya-101 | PYA | 9.85 | 3.93 | 5.92 |
| aya-101 | NEO | 8.04 | 9.27 | -1.23 |
| aya-101 | MIX | 0.93 | -1.34 | 2.27 |
| aya-101 | all | 7.47 | 5.60 | 1.86 |
| aya-expanse-8b | HOM | 15.72 | 12.64 | 3.09 |
| aya-expanse-8b | PYA | 14.47 | 3.65 | 10.83 |
| aya-expanse-8b | NEO | 11.22 | 10.05 | 1.18 |
| aya-expanse-8b | MIX | 3.09 | 1.75 | 1.33 |
| aya-expanse-8b | all | 10.35 | 6.86 | 3.49 |
| deepseek | HOM | 10.47 | 5.81 | 4.67 |
| deepseek | PYA | 5.55 | 1.61 | 3.94 |
| deepseek | NEO | 5.82 | 3.20 | 2.62 |
| deepseek | MIX | 2.63 | -0.75 | 3.38 |
| deepseek | all | 5.58 | 2.35 | 3.23 |
| gemini | HOM | 9.47 | 7.15 | 2.32 |
| gemini | PYA | 9.33 | 1.65 | 7.68 |
| gemini | NEO | 5.92 | 4.69 | 1.23 |
| gemini | MIX | 4.36 | -0.87 | 5.23 |
| gemini | all | 6.08 | 3.01 | 3.07 |
| gemma3-12b | HOM | 8.23 | 9.77 | -1.54 |
| gemma3-12b | PYA | 9.49 | 1.61 | 7.88 |
| gemma3-12b | NEO | 8.42 | 6.01 | 2.41 |
| gemma3-12b | MIX | 3.60 | -0.05 | 3.64 |
| gemma3-12b | all | 7.67 | 4.21 | 3.46 |
| gemma3-27b | HOM | 12.99 | 10.82 | 2.17 |
| gemma3-27b | PYA | 11.90 | 5.37 | 6.53 |
| gemma3-27b | NEO | 9.87 | 8.55 | 1.33 |
| gemma3-27b | MIX | 1.64 | 0.25 | 1.39 |
| gemma3-27b | all | 8.83 | 5.99 | 2.84 |
| gemma3-4b | HOM | 11.73 | 8.67 | 3.05 |
| gemma3-4b | PYA | 8.89 | 1.87 | 7.03 |
| gemma3-4b | NEO | 9.38 | 7.37 | 2.01 |
| gemma3-4b | MIX | 1.95 | 1.05 | 0.89 |
| gemma3-4b | all | 8.28 | 4.64 | 3.64 |
| gemmax2-9b | HOM | 14.47 | 10.40 | 4.07 |
| gemmax2-9b | PYA | 6.28 | 4.20 | 2.08 |
| gemmax2-9b | NEO | 7.46 | 7.57 | -0.12 |
| gemmax2-9b | MIX | 3.63 | -0.09 | 3.73 |
| gemmax2-9b | all | 7.21 | 5.30 | 1.91 |
| google-translate | HOM | 12.38 | 10.31 | 2.07 |
| google-translate | PYA | 5.11 | 3.09 | 2.02 |
| google-translate | NEO | 7.11 | 5.55 | 1.56 |
| google-translate | MIX | 3.76 | -0.03 | 3.79 |
| google-translate | all | 6.78 | 4.57 | 2.21 |
| gpt-4o | HOM | 10.91 | 7.96 | 2.95 |
| gpt-4o | PYA | 9.49 | 2.16 | 7.33 |
| gpt-4o | NEO | 6.40 | 4.45 | 1.95 |
| gpt-4o | MIX | 3.09 | -0.45 | 3.54 |
| gpt-4o | all | 6.31 | 3.39 | 2.92 |
| hunyuan-mt-7b | HOM | 10.89 | 11.19 | -0.30 |
| hunyuan-mt-7b | PYA | 12.08 | 3.70 | 8.38 |
| hunyuan-mt-7b | NEO | 8.05 | 6.57 | 1.48 |
| hunyuan-mt-7b | MIX | 3.98 | 0.16 | 3.82 |
| hunyuan-mt-7b | all | 7.79 | 5.22 | 2.57 |
| nllb-3.3b | HOM | 8.68 | 11.08 | -2.40 |
| nllb-3.3b | PYA | 11.93 | 0.04 | 11.89 |
| nllb-3.3b | NEO | 13.00 | 6.52 | 6.48 |
| nllb-3.3b | MIX | 3.67 | -0.54 | 4.20 |
| nllb-3.3b | all | 11.13 | 4.17 | 6.96 |
| qwen3-0.6b | HOM | 10.30 | 7.94 | 2.36 |
| qwen3-0.6b | PYA | 4.16 | 1.72 | 2.45 |
| qwen3-0.6b | NEO | 7.65 | 5.77 | 1.87 |
| qwen3-0.6b | MIX | 4.63 | -1.74 | 6.36 |
| qwen3-0.6b | all | 7.12 | 3.23 | 3.90 |
| qwen3-1.7b | HOM | 12.18 | 9.86 | 2.32 |
| qwen3-1.7b | PYA | 7.02 | 0.99 | 6.04 |
| qwen3-1.7b | NEO | 6.74 | 5.44 | 1.30 |
| qwen3-1.7b | MIX | 2.31 | 0.69 | 1.62 |
| qwen3-1.7b | all | 6.37 | 4.18 | 2.20 |
| qwen3-14b | HOM | 12.27 | 9.52 | 2.76 |
| qwen3-14b | PYA | 7.16 | 3.34 | 3.82 |
| qwen3-14b | NEO | 7.56 | 5.82 | 1.74 |
| qwen3-14b | MIX | 4.11 | -0.85 | 4.95 |
| qwen3-14b | all | 7.27 | 4.25 | 3.02 |
| qwen3-32b | HOM | 13.38 | 9.62 | 3.77 |
| qwen3-32b | PYA | 6.93 | 3.65 | 3.28 |
| qwen3-32b | NEO | 6.81 | 6.09 | 0.73 |
| qwen3-32b | MIX | 4.07 | -1.27 | 5.34 |
| qwen3-32b | all | 6.79 | 4.28 | 2.51 |
| qwen3-4b | HOM | 10.26 | 10.09 | 0.17 |
| qwen3-4b | PYA | 8.45 | 2.71 | 5.74 |
| qwen3-4b | NEO | 8.55 | 6.12 | 2.43 |
| qwen3-4b | MIX | 2.20 | -0.46 | 2.66 |
| qwen3-4b | all | 7.61 | 4.44 | 3.17 |
| qwen3-8b | HOM | 9.53 | 9.78 | -0.24 |
| qwen3-8b | PYA | 8.06 | 3.14 | 4.91 |
| qwen3-8b | NEO | 8.70 | 5.86 | 2.85 |
| qwen3-8b | MIX | 3.50 | -1.56 | 5.06 |
| qwen3-8b | all | 7.86 | 4.07 | 3.79 |
