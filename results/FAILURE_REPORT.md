# LHF1 failure report: V-JEPA spectral audit

**Verdict: killed at Experiment −1 (H3 no-go after null calibration).**

## What was tested
Whether V-JEPA2 puts frame-order (temporal) information in its top-variance principal components, the ones a naive SOAP-style "remove the common components" step would delete. If time lives there, removing them trades off temporal information and there is a paper; if not, there is no trade-off.
Setup: V-JEPA2 ViT-L (`vjepa2-vitl-fpc64-256`), final layer, 200 Kinetics-400 stand-in clips, frozen, CPU, seed 0. Per-PC temporal Semantic-Invariance (SI_temporal) on patch tokens, compared with 200 random directions (null).

## Kill rule (from PLAN.md, H3)
"< 10 % of removed components have high/order-carrying SI_temporal" → stop (Exp −1 kill criterion). The plan's first pass used a fixed 0.7 threshold; Block 2 replaced it with a null-calibrated threshold (5th percentile of the random-direction null).

## Numbers (from `results/e_minus_1/vjepa2_vitl_final_cpu/summary.json` and `results/block2/summary.json`)
| Quantity | Value |
|---|---|
| Mean SI_temporal, top-50 PCs | 0.640 (min 0.583, max 0.821) |
| Mean SI_temporal, PCs 51–200 | 0.587 |
| Random-direction null | mean 0.611, std 0.014, p5 0.594, p95 0.639 |
| Fixed 0.7 threshold: top-50 fraction below 0.7 | 0.92 (46/50) → first-pass verdict "GO" |
| **Null-calibrated: top-50 fraction below null p5 (order-carrying)** | **0.04 (95% CI 0.02–0.08)** → below 10%, no-go |
| Same, PCs 51–200 | 0.69 (95% CI 0.62–0.79) |
| Clean-vs-shuffled probe acc, all features | 0.972 ± 0.012 |
| Probe, top-50 PCs only | 0.885 ± 0.022 |
| Probe, top-50 variance PCs removed | 0.980 ± 0.010 (random 50 removed: 0.969 ± 0.001) |

## Why it was killed
The first-pass "GO" came from an uncalibrated 0.7 threshold that random directions also fall under. Once calibrated against the null, only 4% of the top-50 PCs carry frame order, while most order-carrying directions sit in the low-variance tail (PCs 51–200). Removing the top variance PCs does not hurt a clean-vs-shuffled order probe (0.980 vs 0.972). So naive SOAP removal does not destroy temporal information: no trade-off, no paper (PLAN.md failure mode 2: "time lives in the tail").
Caveat: K400 stand-in clips, one checkpoint, final layer only, one seed.

## Lesson
Calibrate any threshold against a random-direction null and check where the signal lives (one PCA + probe) before planning a trade-off paper.
