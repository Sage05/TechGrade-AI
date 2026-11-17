# Experiment 019 — Validation-Only Scorer Calibration

## Objective
Determine whether validation-fitted bias correction or affine calibration improves numerical scoring on held-out questions.

## Methodology
- Five question-level random-state splits were evaluated.
- The existing concept-aware feature extractor was used.
- The categorical classifier was trained on the outer training split.
- Ridge alpha was selected using GroupKFold within outer training.
- Calibration parameters were fitted on validation predictions only.
- Test labels and scores were not used to fit calibration parameters.
- Calibrated test scores were clipped to the 0–10 interval.
- The global training mean is an uncalibrated baseline.

## Mean metrics across splits
| dataset | method | variant | repetitions | MAE_mean | MAE_std | RMSE_mean | RMSE_std | mean_prediction_error_mean | Spearman_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DBMS | Predicted-category mean | raw | 5 | 1.476 | 0.727 | 2.385 | 1.206 | -0.458 | 0.728 |
| DBMS | Probability-weighted scorer | raw | 5 | 1.646 | 0.473 | 2.391 | 0.723 | -0.550 | 0.740 |
| DBMS | Predicted-category mean | bias_correction | 5 | 1.730 | 0.574 | 2.638 | 1.038 | -0.572 | 0.704 |
| DBMS | Probability-weighted scorer | bias_correction | 5 | 1.744 | 0.426 | 2.505 | 0.718 | -0.496 | 0.740 |
| DBMS | Ridge regression | bias_correction | 5 | 1.776 | 0.207 | 2.446 | 0.346 | -0.244 | 0.817 |
| DBMS | Probability-weighted scorer | affine_calibration | 5 | 1.798 | 0.379 | 2.489 | 0.578 | -0.327 | 0.740 |
| DBMS | Ridge regression | raw | 5 | 1.802 | 0.195 | 2.318 | 0.328 | 0.027 | 0.813 |
| DBMS | Ridge regression | affine_calibration | 5 | 1.812 | 0.193 | 2.441 | 0.327 | -0.119 | 0.813 |
| DBMS | Predicted-category mean | affine_calibration | 5 | 1.898 | 0.574 | 2.446 | 0.776 | -0.217 | 0.728 |
| DBMS | Global mean baseline | raw | 5 | 3.351 | 0.156 | 3.687 | 0.111 | -0.052 | — |
| OS | Predicted-category mean | raw | 5 | 1.260 | 0.523 | 2.384 | 0.819 | -0.296 | 0.767 |
| OS | Predicted-category mean | affine_calibration | 5 | 1.265 | 0.551 | 2.327 | 0.768 | -0.293 | 0.767 |
| OS | Predicted-category mean | bias_correction | 5 | 1.277 | 0.564 | 2.384 | 0.827 | -0.293 | 0.767 |
| OS | Probability-weighted scorer | raw | 5 | 1.375 | 0.354 | 2.253 | 0.580 | -0.307 | 0.768 |
| OS | Ridge regression | bias_correction | 5 | 1.384 | 0.197 | 1.819 | 0.157 | -0.189 | 0.858 |
| OS | Probability-weighted scorer | bias_correction | 5 | 1.389 | 0.327 | 2.285 | 0.572 | -0.357 | 0.768 |
| OS | Probability-weighted scorer | affine_calibration | 5 | 1.397 | 0.361 | 2.299 | 0.615 | -0.351 | 0.766 |
| OS | Ridge regression | affine_calibration | 5 | 1.448 | 0.217 | 1.884 | 0.138 | -0.210 | 0.865 |
| OS | Ridge regression | raw | 5 | 1.738 | 0.504 | 2.044 | 0.571 | 0.612 | 0.868 |
| OS | Global mean baseline | raw | 5 | 3.435 | 0.079 | 3.771 | 0.061 | 0.039 | — |

## Interpretation cautions
The validation partitions contain only a small number of unique questions. Calibration parameters can therefore vary considerably across splits. Repeated splits reuse questions and are not independent external validation.
The raw results in this experiment are evaluated on the same outer test questions as the calibrated versions, enabling within-split comparisons. The test results should not be used repeatedly to tune parameters or claim definitive superiority.
Exp 019 uses inner cross-validation to select Ridge alpha, so its Ridge setup is not identical to the validation-tuned Ridge in Exp 016.

## Files generated
- `exp019_run_metrics.csv`
- `exp019_summary.csv`
- `exp019_predictions.csv`
- `exp019_calibration_parameters.csv`
- `exp019_split_manifest.csv`
- `exp019_ridge_inner_cv.csv`
- `exp019_calibration_mae_comparison.png`
