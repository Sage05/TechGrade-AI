# Experiment 020 — Final Consolidated Evaluation

## Objective
Consolidate categorical classification and numerical scoring results from the current TechGradeAI datasets.

## Evaluation design
- Datasets: Operating Systems and DBMS.
- Random states: [42, 52, 62, 72, 82].
- Splits are question-level; leakage checks passed for all runs.
- Ridge alpha was selected using GroupKFold within the training partition.
- Training-fold score means were used by category-dependent scorers.
- Calibration parameters were fitted using validation predictions only.
- All final numerical predictions were clipped to the same 0–10 interval.
- Test human scores were not used to fit models or calibrators.

## Categorical classification summary
| dataset | repetitions | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std | weighted_f1_mean | weighted_f1_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DBMS | 5 | 0.711 | 0.137 | 0.708 | 0.113 | 0.714 | 0.136 |
| OS | 5 | 0.711 | 0.120 | 0.675 | 0.132 | 0.712 | 0.118 |

## Numerical scorer summary
| dataset | method | variant | repetitions | MAE_mean | MAE_std | RMSE_mean | RMSE_std | R2_mean | Spearman_mean | mean_prediction_error_mean | score_clipped_pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DBMS | Predicted-category mean | raw | 5 | 1.476 | 0.727 | 2.385 | 1.206 | 0.485 | 0.728 | -0.458 | 0.000 |
| DBMS | Probability-weighted scorer | raw | 5 | 1.646 | 0.473 | 2.391 | 0.723 | 0.541 | 0.740 | -0.550 | 0.000 |
| DBMS | Ridge regression | raw | 5 | 1.699 | 0.245 | 2.237 | 0.353 | 0.620 | 0.813 | 0.008 | 10.819 |
| DBMS | Predicted-category mean | bias_correction | 5 | 1.730 | 0.574 | 2.638 | 1.038 | 0.415 | 0.704 | -0.572 | 16.901 |
| DBMS | Probability-weighted scorer | bias_correction | 5 | 1.744 | 0.426 | 2.505 | 0.718 | 0.501 | 0.740 | -0.496 | 2.105 |
| DBMS | Ridge regression | affine_calibration | 5 | 1.745 | 0.200 | 2.378 | 0.330 | 0.573 | 0.814 | -0.110 | 11.813 |
| DBMS | Ridge regression | bias_correction | 5 | 1.748 | 0.191 | 2.371 | 0.325 | 0.577 | 0.812 | -0.140 | 10.702 |
| DBMS | Probability-weighted scorer | affine_calibration | 5 | 1.798 | 0.379 | 2.489 | 0.578 | 0.517 | 0.740 | -0.327 | 4.444 |
| DBMS | Predicted-category mean | affine_calibration | 5 | 1.898 | 0.574 | 2.446 | 0.776 | 0.520 | 0.728 | -0.217 | 0.000 |
| DBMS | Global mean baseline | raw | 5 | 3.351 | 0.156 | 3.687 | 0.111 | -0.003 | — | -0.052 | 0.000 |
| OS | Predicted-category mean | raw | 5 | 1.260 | 0.523 | 2.384 | 0.819 | 0.561 | 0.767 | -0.296 | 0.000 |
| OS | Predicted-category mean | affine_calibration | 5 | 1.265 | 0.551 | 2.327 | 0.768 | 0.584 | 0.767 | -0.293 | 10.000 |
| OS | Predicted-category mean | bias_correction | 5 | 1.277 | 0.564 | 2.384 | 0.827 | 0.560 | 0.767 | -0.293 | 10.000 |
| OS | Ridge regression | affine_calibration | 5 | 1.328 | 0.178 | 1.753 | 0.158 | 0.783 | 0.858 | -0.146 | 20.000 |
| OS | Ridge regression | bias_correction | 5 | 1.331 | 0.177 | 1.751 | 0.155 | 0.784 | 0.858 | -0.172 | 20.000 |
| OS | Probability-weighted scorer | raw | 5 | 1.375 | 0.354 | 2.253 | 0.580 | 0.622 | 0.768 | -0.307 | 0.000 |
| OS | Probability-weighted scorer | bias_correction | 5 | 1.389 | 0.327 | 2.285 | 0.572 | 0.612 | 0.768 | -0.357 | 6.667 |
| OS | Probability-weighted scorer | affine_calibration | 5 | 1.397 | 0.361 | 2.299 | 0.615 | 0.604 | 0.766 | -0.351 | 11.111 |
| OS | Ridge regression | raw | 5 | 1.494 | 0.361 | 1.830 | 0.426 | 0.757 | 0.870 | 0.570 | 20.000 |
| OS | Global mean baseline | raw | 5 | 3.435 | 0.079 | 3.771 | 0.061 | -0.001 | — | 0.039 | 0.000 |

## Interpretation
Lower MAE and RMSE indicate smaller numerical errors. Mean prediction error is predicted score minus human score; positive values indicate overprediction and negative values indicate underprediction. Spearman correlation describes rank-order agreement.
The method rows are pre-specified comparisons. The method with the lowest observed test MAE is not automatically declared a universally best model; repeated test comparisons can introduce selection bias.

## Important limitations
The datasets contain only 14 OS questions and 15 DBMS questions. The five repeated splits reuse questions and some answer rows. The resulting standard deviations describe split variability, not uncertainty from five independent external datasets.
Calibration validation sets are small, so calibration parameters may be unstable. The results support exploratory comparisons but do not establish production readiness or generalisation to other subjects.

## Output files
- `exp020_run_metrics.csv`
- `exp020_summary.csv`
- `exp020_predictions.csv`
- `exp020_classification_metrics.csv`
- `exp020_classification_summary.csv`
- `exp020_per_class_metrics.csv`
- `exp020_calibration_parameters.csv`
- `exp020_split_manifest.csv`
- `exp020_ridge_inner_cv.csv`
- `exp020_confusion_counts_os.csv`
- `exp020_confusion_counts_dbms.csv`
- `exp020_confusion_row_percent_os.csv`
- `exp020_confusion_row_percent_dbms.csv`
- `exp020_final_mae_comparison.png`

## Final note
This experiment concludes the current evaluation phase. Further work should prioritise expanding the number and diversity of questions, validating human-score agreement, and evaluating on genuinely independent questions.
