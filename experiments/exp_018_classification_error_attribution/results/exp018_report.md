# Experiment 018 — Classification Error Attribution

## Purpose
Compare the current predicted-category mean with a diagnostic oracle-category mean. The oracle uses the true test label to select a score mean calculated only from that split's training fold. It is not a deployable scorer.

## Data and caveats
- Test prediction occurrences: 183.
- Unique answer rows across test runs: 134.
- Datasets: DBMS, OS.
- Random states: 42, 52, 62, 72, 82.
- Median absolute difference between saved predicted-category means and training-mean lookup: 0.000000.
- Repeated test appearances reuse some answers and are not independent samples.
- Oracle scores are a diagnostic only; never report them as ordinary model test scores.

## 1. Classification metrics (pooled rows and per split)
| dataset | random_state | prediction_occurrences | unique_test_rows | accuracy | macro_f1 | weighted_f1 | classification_error_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DBMS | POOLED_ALL_RUNS | 93 | 74 | 0.710 | 0.706 | 0.714 | 29.032 |
| DBMS | 42 | 19 | 19 | 0.684 | 0.699 | 0.696 | 31.579 |
| DBMS | 52 | 18 | 18 | 0.667 | 0.668 | 0.663 | 33.333 |
| DBMS | 62 | 19 | 19 | 0.789 | 0.783 | 0.794 | 21.053 |
| DBMS | 72 | 19 | 19 | 0.526 | 0.549 | 0.528 | 47.368 |
| DBMS | 82 | 18 | 18 | 0.889 | 0.843 | 0.889 | 11.111 |
| OS | POOLED_ALL_RUNS | 90 | 60 | 0.711 | 0.681 | 0.717 | 28.889 |
| OS | 42 | 18 | 18 | 0.611 | 0.578 | 0.621 | 38.889 |
| OS | 52 | 18 | 18 | 0.833 | 0.828 | 0.841 | 16.667 |
| OS | 62 | 18 | 18 | 0.556 | 0.502 | 0.557 | 44.444 |
| OS | 72 | 18 | 18 | 0.778 | 0.755 | 0.781 | 22.222 |
| OS | 82 | 18 | 18 | 0.778 | 0.710 | 0.758 | 22.222 |

## 2. Current versus oracle-category score mapping
| dataset | scorer | prediction_occurrences | unique_test_rows | MAE | RMSE | mean_prediction_error | within_1_point_pct | within_2_points_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DBMS | Predicted-category mean (current) | 93 | 74 | 1.485 | 2.634 | -0.469 | 75.269 | 80.645 |
| DBMS | Oracle-category mean (diagnostic only) | 93 | 74 | 0.522 | 0.633 | -0.012 | 94.624 | 100.000 |
| OS | Predicted-category mean (current) | 90 | 60 | 1.260 | 2.494 | -0.296 | 86.667 | 87.778 |
| OS | Oracle-category mean (diagnostic only) | 90 | 60 | 0.491 | 0.603 | 0.039 | 98.889 | 100.000 |

## 3. Attribution by classification correctness
| dataset | classification_status | prediction_occurrences | unique_test_rows | current_category_mean_MAE | oracle_category_mean_MAE | mean_absolute_error_reduction_with_oracle | oracle_improvement_occurrence_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DBMS | Incorrect category | 27 | 23 | 3.839 | 0.519 | 3.320 | 92.593 |
| DBMS | Correct category | 66 | 54 | 0.523 | 0.523 | 0.000 | 0.000 |
| OS | Incorrect category | 26 | 19 | 3.171 | 0.511 | 2.660 | 84.615 |
| OS | Correct category | 64 | 45 | 0.484 | 0.484 | 0.000 | 0.000 |

A positive MAE reduction means the oracle diagnostic reduced absolute error. This is not a causal estimate: the oracle uses the true test category by design, and score differences also reflect the relationship between labels and human scores.

## Key output files
- `exp018_classification_metrics.csv`: accuracy and F1 for each split and pooled results.
- `exp018_per_class_report.csv`: precision, recall and F1 by class.
- `exp018_confusion_counts_*.csv` and `exp018_confusion_row_percent_*.csv`: confusion matrices.
- `exp018_predicted_vs_oracle_score_metrics.csv`: current vs diagnostic scoring metrics.
- `exp018_error_by_classification_correctness.csv`: score errors for correct/incorrect classifications.
- `exp018_label_transition_errors.csv`: error impact for actual-to-predicted category pairs.
- `exp018_answer_level_attribution.csv`: row-level diagnostic.
- `exp018_misclassifications_oracle_helps_most.csv` / `exp018_misclassifications_oracle_helps_least.csv`: examples to inspect.
- `exp018_confusion_matrices.png`: confusion-matrix plot.

## Limitations
There are few unique questions; repeated random splits are not independent external validation. No source data, human annotations, labels, or models were modified.
