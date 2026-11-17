# TechGrade AI — Experiment 015

## Learned Regression Scorer

### Objective

Evaluate whether direct regression from the existing 36-feature representation predicts human-assigned scores more accurately than deterministic categorical baselines.

### Models compared

1. Global training-score mean.
2. Predicted-category training-score mean.
3. Probability-weighted categorical scorer.
4. StandardScaler followed by Ridge Regression.

### Validation protocol

Ridge alpha is selected using validation MAE. Candidate values are [0.1, 1.0, 10.0, 100.0]. The held-out test set is not used to select alpha.

### Validation tuning results

| dataset | alpha | validation_rows | validation_MAE | validation_RMSE |
| --- | --- | --- | --- | --- |
| OS | 0.1000 | 12 | 1.4938 | 2.1695 |
| OS | 1.0000 | 12 | 1.3241 | 1.6915 |
| OS | 10.0000 | 12 | 1.4915 | 1.7937 |
| OS | 100.0000 | 12 | 2.0373 | 2.4019 |
| DBMS | 0.1000 | 18 | 2.0483 | 2.7806 |
| DBMS | 1.0000 | 18 | 1.9564 | 2.6497 |
| DBMS | 10.0000 | 18 | 1.9168 | 2.5623 |
| DBMS | 100.0000 | 18 | 2.0568 | 2.5758 |

### Held-out test results

| dataset | method | selected_ridge_alpha | test_rows | MAE | RMSE | R2 | Spearman | mean_prediction_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OS | Global mean baseline | NA | 18 | 3.4372 | 3.7551 | -0.0002 | NA | 0.0463 |
| OS | Predicted-category mean | NA | 18 | 1.7500 | 3.0870 | 0.3241 | 0.6148 | -0.1883 |
| OS | Probability-weighted scorer | NA | 18 | 1.7199 | 2.6980 | 0.4837 | 0.6705 | -0.4654 |
| OS | Ridge regression | 1.0000 | 18 | 1.8889 | 2.7216 | 0.4746 | 0.7256 | 0.5077 |
| DBMS | Global mean baseline | NA | 19 | 3.4455 | 3.7618 | -0.0045 | NA | -0.2531 |
| DBMS | Predicted-category mean | NA | 19 | 1.5052 | 2.5664 | 0.5325 | 0.7461 | -0.5308 |
| DBMS | Probability-weighted scorer | NA | 19 | 1.8151 | 2.4452 | 0.5756 | 0.8290 | -0.8117 |
| DBMS | Ridge regression | 10.0000 | 19 | 1.6819 | 2.3119 | 0.6206 | 0.8783 | -0.5642 |

### Metric interpretation

- Lower MAE means smaller average absolute score error.
- Lower RMSE means smaller squared-error penalty on large mistakes.
- Higher R-squared indicates better performance relative to predicting the test-set mean.
- Spearman measures rank association, not numerical agreement.
- Negative R-squared is possible.
- Numerical predictions are clipped to the 0–10 scoring scale.

### Limitations

- Only a small number of unique questions are available.
- Validation contains few questions, so selected alpha may be unstable.
- Results use one question-level split per subject.
- The predicted category probabilities may not be calibrated.
- Human scores may contain annotation noise or evaluator subjectivity.
- Results are exploratory and do not establish production readiness.

### Next step

Compare Exp 014 and Exp 015 on the same held-out questions. Repeat the evaluation across multiple question-level splits before choosing a final scoring method.
