# TechGrade AI — Experiment 014

## Deterministic Numerical Scoring Baseline

### Objective

Test whether Exp 008 categorical predictions can be converted into 0–10 numerical scores using category means estimated from the training data.

### Scoring methods

1. **Global mean baseline:** predicts the training-set mean human score for every held-out answer.
2. **Predicted-category mean:** uses the training-set mean for the most likely predicted category.
3. **Probability-weighted scorer:** computes the weighted average of training-set category means using Exp 008 class probabilities.

### Training-only rule

Human scores in the held-out test set are used only to calculate evaluation metrics. They are not used to construct category means or numerical predictions.

### Category score mappings

| dataset | label | training_count | training_mean_score |
| --- | --- | --- | --- |
| OS | Correct | 18 | 8.7500 |
| OS | Partially Correct | 9 | 4.5556 |
| OS | Incorrect | 9 | 0.9444 |
| OS | Irrelevant | 9 | 0.0000 |
| OS | Contradictory | 9 | 0.6111 |
| DBMS | Correct | 18 | 8.8889 |
| DBMS | Partially Correct | 9 | 4.2222 |
| DBMS | Incorrect | 13 | 1.4231 |
| DBMS | Irrelevant | 6 | 0.0000 |
| DBMS | Contradictory | 9 | 0.6111 |

### Held-out test results

| dataset | method | test_rows | MAE | RMSE | R2 | Spearman | mean_prediction_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OS | Global mean baseline | 18 | 3.4372 | 3.7551 | -0.0002 | NA | 0.0463 |
| OS | Predicted-category mean | 18 | 1.7500 | 3.0870 | 0.3241 | 0.6148 | -0.1883 |
| OS | Probability-weighted scorer | 18 | 1.7199 | 2.6980 | 0.4837 | 0.6705 | -0.4654 |
| DBMS | Global mean baseline | 19 | 3.4455 | 3.7618 | -0.0045 | NA | -0.2531 |
| DBMS | Predicted-category mean | 19 | 1.5052 | 2.5664 | 0.5325 | 0.7461 | -0.5308 |
| DBMS | Probability-weighted scorer | 19 | 1.8151 | 2.4452 | 0.5756 | 0.8290 | -0.8117 |

### Interpretation guidance

- Lower MAE and RMSE indicate smaller prediction errors.
- Higher R-squared indicates that more test-score variance is explained relative to predicting the test-set mean.
- Spearman correlation measures rank association, not numerical agreement.
- Negative R-squared is possible and means predictions perform worse than the test-set-mean reference under that metric.
- Category probabilities from the current Logistic Regression model are not guaranteed to be calibrated. The probability-weighted result is a baseline, not a validated final scoring method.

### Limitations

- The datasets contain only a small number of unique questions.
- The results come from one question-level split per subject.
- The validation split is not used to tune this deterministic scoring rule.
- Category means may be unstable where training counts are small.
- Numerical scores may capture distinctions that are not represented by the five categorical labels.
- Results do not establish production-level grading accuracy.

### Next step

Compare this deterministic baseline with a learned regression model in Experiment 015 using the same question-level evaluation discipline.
