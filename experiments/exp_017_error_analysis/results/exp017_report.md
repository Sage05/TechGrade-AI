# Experiment 017 — Error Analysis and Scorer Calibration

## Objective
Analyse the saved predictions from Experiment 016 to understand where each numerical scorer makes errors. This experiment does not retrain models, change labels, modify human scores, or automatically calibrate any predictions.

## Data and interpretation
- Input prediction rows: 183.
- Unique original answer rows: 134.
- Datasets: DBMS, OS.
- Random states observed: 5.
- Residual = predicted score − actual human score.
- Positive residual means overprediction; negative residual means underprediction.
- The same original answer can occur in multiple test splits. Repeated predictions are not independent answers or independent datasets.

## 1. Mean performance across random-state runs
The table below reports the mean and standard deviation of the metrics calculated separately for each random state. These are the more appropriate figures for comparing the repeated splits.

| dataset | method | repetitions | MAE_mean | MAE_std | RMSE_mean | RMSE_std | R2_mean | Spearman_mean | mean_prediction_error_mean | prediction_to_actual_std_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DBMS | Predicted-category mean | 5 | 1.476 | 0.727 | 2.385 | 1.206 | 0.485 | 0.728 | -0.458 | 0.937 |
| DBMS | Probability-weighted scorer | 5 | 1.646 | 0.473 | 2.391 | 0.723 | 0.541 | 0.740 | -0.550 | 0.754 |
| DBMS | Ridge regression | 5 | 1.771 | 0.282 | 2.283 | 0.366 | 0.606 | 0.796 | 0.188 | 0.810 |
| DBMS | Global mean baseline | 5 | 3.351 | 0.156 | 3.687 | 0.111 | -0.003 | — | -0.052 | 0.000 |
| OS | Predicted-category mean | 5 | 1.260 | 0.523 | 2.384 | 0.819 | 0.561 | 0.767 | -0.296 | 0.988 |
| OS | Probability-weighted scorer | 5 | 1.375 | 0.354 | 2.253 | 0.580 | 0.622 | 0.768 | -0.307 | 0.884 |
| OS | Ridge regression | 5 | 1.804 | 0.317 | 2.304 | 0.417 | 0.619 | 0.791 | 0.613 | 0.874 |
| OS | Global mean baseline | 5 | 3.435 | 0.079 | 3.771 | 0.061 | -0.001 | — | 0.039 | 0.000 |

### DBMS
- Lowest mean MAE: **Predicted-category mean** (1.476).
- Lowest mean RMSE: **Ridge regression** (2.283).
- Highest mean Spearman correlation: **Ridge regression** (0.796).
- Prediction-to-actual standard-deviation ratios are listed in the main table. Ratios below 1 indicate a narrower spread of predicted scores than actual scores; this is a descriptive compression diagnostic, not proof of calibration quality.

### OS
- Lowest mean MAE: **Predicted-category mean** (1.260).
- Lowest mean RMSE: **Probability-weighted scorer** (2.253).
- Highest mean Spearman correlation: **Ridge regression** (0.791).
- Prediction-to-actual standard-deviation ratios are listed in the main table. Ratios below 1 indicate a narrower spread of predicted scores than actual scores; this is a descriptive compression diagnostic, not proof of calibration quality.

## 2. Pooled prediction metrics
Pooled metrics combine all available test-prediction occurrences across random states. Since original answers can reappear across runs, these values are descriptive and are not independent-sample estimates.

| dataset | method | n_prediction_occurrences | unique_test_rows | MAE | RMSE | mean_prediction_error | overprediction_pct | underprediction_pct | within_1_point_pct | prediction_to_actual_std_ratio | predictions_below_zero_pct | predictions_above_10_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DBMS | Predicted-category mean | 93 | 74 | 1.485 | 2.634 | -0.469 | 37.634 | 51.613 | 75.269 | 0.948 | 0.000 | 0.000 |
| DBMS | Probability-weighted scorer | 93 | 74 | 1.653 | 2.487 | -0.559 | 49.462 | 50.538 | 54.839 | 0.763 | 0.000 | 0.000 |
| DBMS | Ridge regression | 93 | 74 | 1.776 | 2.312 | 0.183 | 52.688 | 40.860 | 31.183 | 0.815 | 0.000 | 0.000 |
| DBMS | Global mean baseline | 93 | 74 | 3.352 | 3.689 | -0.055 | 56.989 | 43.011 | 11.828 | 0.021 | 0.000 | 0.000 |
| OS | Predicted-category mean | 90 | 60 | 1.260 | 2.494 | -0.296 | 47.778 | 36.667 | 86.667 | 0.991 | 0.000 | 0.000 |
| OS | Probability-weighted scorer | 90 | 60 | 1.375 | 2.312 | -0.307 | 63.333 | 36.667 | 68.889 | 0.889 | 0.000 | 0.000 |
| OS | Ridge regression | 90 | 60 | 1.804 | 2.334 | 0.613 | 52.222 | 36.667 | 36.667 | 0.884 | 0.000 | 0.000 |
| OS | Global mean baseline | 90 | 60 | 3.435 | 3.772 | 0.039 | 52.222 | 47.778 | 13.333 | 0.008 | 0.000 | 0.000 |

## 3. Error patterns by actual categorical label
See `exp017_actual_label_error_summary.csv` for MAE, RMSE, bias and the number of evaluated occurrences for every dataset, scorer and actual label. Interpret small groups cautiously.

## 4. Error patterns by predicted categorical label
See `exp017_predicted_label_error_summary.csv` for numerical scoring errors grouped by the categorical label predicted by the classification model.

## 5. Error patterns by actual score band
See `exp017_score_band_error_summary.csv` and `exp017_error_by_actual_score_band.png`. The bias plot uses predicted score minus actual human score: values above zero indicate overprediction and values below zero indicate underprediction.

## 6. Errors by question
See `exp017_question_error_summary.csv` to identify questions where a scorer performs poorly or inconsistently. Some questions may have few student responses, so their metrics can be unstable.

## 7. Repeated errors for individual answers
`exp017_row_repeat_summary.csv` aggregates repeated predictions for each original answer and scorer. `exp017_worst_errors.csv` contains the ten highest mean absolute errors per dataset and scorer, based on the observed appearances of each answer in the test splits. Review the associated question, answer, actual label and human score before concluding that a scorer or human annotation is wrong.

## 8. Interpretation cautions
- These datasets contain a limited number of unique questions.
- Repeated random splits reuse questions and often reuse answer rows, so repetitions are not independent external validations.
- Error patterns can reflect model limitations, variation in answer difficulty, score-label alignment, rubric ambiguity, or human annotation decisions.
- This experiment does not automatically alter labels, human scores, model outputs, or calibration settings.
- Predictions outside the 0–10 range are reported rather than silently clipped.

## Generated files
- `exp017_residuals_long.csv`
- `exp017_pooled_method_metrics.csv`
- `exp017_run_level_metrics.csv`
- `exp017_run_level_summary.csv`
- `exp017_actual_label_error_summary.csv`
- `exp017_predicted_label_error_summary.csv`
- `exp017_score_band_error_summary.csv`
- `exp017_question_error_summary.csv`
- `exp017_row_repeat_summary.csv`
- `exp017_worst_errors.csv`
- `exp017_mae_by_method.png`
- `exp017_error_by_actual_score_band.png`
