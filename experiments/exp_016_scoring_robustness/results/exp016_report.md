# TechGrade AI — Experiment 016

## Scoring Robustness and Comparative Evaluation

### Objective

Compare deterministic scoring baselines and Ridge Regression across repeated question-level splits.

### Experimental protocol

- Random states: [42, 52, 62, 72, 82]
- Question-level train/validation/test separation.
- TF-IDF is fitted on training text only.
- Score mappings use training scores only.
- Ridge alpha is selected using validation MAE.
- Test scores are reserved for final evaluation.
- All scoring methods use the same test rows within each run.

### Mean and standard deviation across repetitions

| dataset | method | repetitions | MAE_mean | MAE_std | RMSE_mean | RMSE_std | R2_mean | R2_std | Spearman_mean | Spearman_std | mean_prediction_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DBMS | Global mean baseline | 5 | 3.3507 | 0.1559 | 3.6866 | 0.1111 | -0.0031 | 0.0020 | NA | NA | -0.0522 |
| DBMS | Predicted-category mean | 5 | 1.4765 | 0.7269 | 2.3849 | 1.2062 | 0.4848 | 0.4648 | 0.7278 | 0.2023 | -0.4583 |
| DBMS | Probability-weighted scorer | 5 | 1.6457 | 0.4729 | 2.3907 | 0.7227 | 0.5413 | 0.3020 | 0.7401 | 0.1947 | -0.5499 |
| DBMS | Ridge regression | 5 | 1.7712 | 0.2820 | 2.2833 | 0.3659 | 0.6062 | 0.1395 | 0.7955 | 0.1159 | 0.1884 |
| OS | Global mean baseline | 5 | 3.4354 | 0.0791 | 3.7714 | 0.0607 | -0.0008 | 0.0013 | NA | NA | 0.0389 |
| OS | Predicted-category mean | 5 | 1.2599 | 0.5231 | 2.3840 | 0.8193 | 0.5606 | 0.2564 | 0.7671 | 0.1419 | -0.2963 |
| OS | Probability-weighted scorer | 5 | 1.3746 | 0.3543 | 2.2530 | 0.5801 | 0.6219 | 0.1646 | 0.7678 | 0.1060 | -0.3070 |
| OS | Ridge regression | 5 | 1.8041 | 0.3171 | 2.3041 | 0.4170 | 0.6186 | 0.1227 | 0.7906 | 0.0414 | 0.6134 |

### Ridge validation tuning results

| dataset | random_state | alpha | validation_rows | validation_questions | validation_MAE | validation_RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| OS | 42 | 0.1000 | 12 | 2 | 1.4938 | 2.1695 |
| OS | 42 | 1.0000 | 12 | 2 | 1.3241 | 1.6915 |
| OS | 42 | 10.0000 | 12 | 2 | 1.4915 | 1.7937 |
| OS | 42 | 100.0000 | 12 | 2 | 2.0373 | 2.4019 |
| OS | 52 | 0.1000 | 12 | 2 | 1.0441 | 1.4468 |
| OS | 52 | 1.0000 | 12 | 2 | 0.8488 | 1.0454 |
| OS | 52 | 10.0000 | 12 | 2 | 1.3465 | 1.6334 |
| OS | 52 | 100.0000 | 12 | 2 | 1.7463 | 2.0851 |
| OS | 62 | 0.1000 | 12 | 2 | 3.4439 | 4.8517 |
| OS | 62 | 1.0000 | 12 | 2 | 3.1257 | 4.6466 |
| OS | 62 | 10.0000 | 12 | 2 | 1.9634 | 2.9810 |
| OS | 62 | 100.0000 | 12 | 2 | 1.6418 | 2.2823 |
| OS | 72 | 0.1000 | 12 | 2 | 0.7935 | 1.1911 |
| OS | 72 | 1.0000 | 12 | 2 | 0.8456 | 1.0275 |
| OS | 72 | 10.0000 | 12 | 2 | 1.2082 | 1.4925 |
| OS | 72 | 100.0000 | 12 | 2 | 1.6693 | 1.8536 |
| OS | 82 | 0.1000 | 12 | 2 | 1.1475 | 1.4500 |
| OS | 82 | 1.0000 | 12 | 2 | 0.8844 | 1.2580 |
| OS | 82 | 10.0000 | 12 | 2 | 0.8999 | 1.2722 |
| OS | 82 | 100.0000 | 12 | 2 | 1.1600 | 1.4827 |
| DBMS | 42 | 0.1000 | 18 | 3 | 2.0483 | 2.7806 |
| DBMS | 42 | 1.0000 | 18 | 3 | 1.9564 | 2.6497 |
| DBMS | 42 | 10.0000 | 18 | 3 | 1.9168 | 2.5623 |
| DBMS | 42 | 100.0000 | 18 | 3 | 2.0568 | 2.5758 |
| DBMS | 52 | 0.1000 | 18 | 3 | 2.0984 | 2.8249 |
| DBMS | 52 | 1.0000 | 18 | 3 | 1.6301 | 2.1731 |
| DBMS | 52 | 10.0000 | 18 | 3 | 1.3380 | 1.7932 |
| DBMS | 52 | 100.0000 | 18 | 3 | 1.7890 | 2.0432 |
| DBMS | 62 | 0.1000 | 18 | 3 | 1.8344 | 2.6740 |
| DBMS | 62 | 1.0000 | 18 | 3 | 1.9281 | 2.6571 |
| DBMS | 62 | 10.0000 | 18 | 3 | 1.7058 | 2.2852 |
| DBMS | 62 | 100.0000 | 18 | 3 | 1.5439 | 1.9604 |
| DBMS | 72 | 0.1000 | 18 | 3 | 2.1131 | 2.8038 |
| DBMS | 72 | 1.0000 | 18 | 3 | 2.0255 | 2.5987 |
| DBMS | 72 | 10.0000 | 18 | 3 | 2.0246 | 2.4532 |
| DBMS | 72 | 100.0000 | 18 | 3 | 2.2585 | 2.5370 |
| DBMS | 82 | 0.1000 | 19 | 3 | 2.3728 | 3.2344 |
| DBMS | 82 | 1.0000 | 19 | 3 | 1.7677 | 2.3913 |
| DBMS | 82 | 10.0000 | 19 | 3 | 1.7646 | 2.3409 |
| DBMS | 82 | 100.0000 | 19 | 3 | 1.9513 | 2.3715 |

### Per-split test metrics

| dataset | random_state | method | test_rows | MAE | RMSE | R2 | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OS | 42 | Global mean baseline | 18 | 3.4372 | 3.7551 | -0.0002 | NA |
| OS | 42 | Predicted-category mean | 18 | 1.7500 | 3.0870 | 0.3241 | 0.6148 |
| OS | 42 | Probability-weighted scorer | 18 | 1.7199 | 2.6980 | 0.4837 | 0.6705 |
| OS | 42 | Ridge regression | 18 | 1.8889 | 2.7216 | 0.4746 | 0.7256 |
| OS | 52 | Global mean baseline | 18 | 3.3056 | 3.6904 | -0.0001 | NA |
| OS | 52 | Predicted-category mean | 18 | 1.0633 | 2.2871 | 0.6159 | 0.8844 |
| OS | 52 | Probability-weighted scorer | 18 | 1.1243 | 2.2813 | 0.6178 | 0.8511 |
| OS | 52 | Ridge regression | 18 | 1.2480 | 1.6552 | 0.7988 | 0.8293 |
| OS | 62 | Global mean baseline | 18 | 3.4722 | 3.8003 | -0.0002 | NA |
| OS | 62 | Predicted-category mean | 18 | 1.8611 | 3.2005 | 0.2906 | 0.6215 |
| OS | 62 | Probability-weighted scorer | 18 | 1.7769 | 2.7265 | 0.4851 | 0.6405 |
| OS | 62 | Ridge regression | 18 | 2.0454 | 2.4598 | 0.5810 | 0.7984 |
| OS | 72 | Global mean baseline | 18 | 3.4444 | 3.7572 | -0.0006 | NA |
| OS | 72 | Predicted-category mean | 18 | 0.9769 | 2.1742 | 0.6649 | 0.8023 |
| OS | 72 | Probability-weighted scorer | 18 | 1.2579 | 2.2669 | 0.6357 | 0.8041 |
| OS | 72 | Ridge regression | 18 | 1.8973 | 2.1507 | 0.6721 | 0.7784 |
| OS | 82 | Global mean baseline | 18 | 3.5175 | 3.8542 | -0.0031 | NA |
| OS | 82 | Predicted-category mean | 18 | 0.6481 | 1.1711 | 0.9074 | 0.9126 |
| OS | 82 | Probability-weighted scorer | 18 | 0.9940 | 1.2924 | 0.8872 | 0.8728 |
| OS | 82 | Ridge regression | 18 | 1.9411 | 2.5334 | 0.5666 | 0.8212 |
| DBMS | 42 | Global mean baseline | 19 | 3.4455 | 3.7618 | -0.0045 | NA |
| DBMS | 42 | Predicted-category mean | 19 | 1.5052 | 2.5664 | 0.5325 | 0.7461 |
| DBMS | 42 | Probability-weighted scorer | 19 | 1.8151 | 2.4452 | 0.5756 | 0.8290 |
| DBMS | 42 | Ridge regression | 19 | 1.6819 | 2.3119 | 0.6206 | 0.8783 |
| DBMS | 52 | Global mean baseline | 18 | 3.1190 | 3.5337 | -0.0006 | NA |
| DBMS | 52 | Predicted-category mean | 18 | 1.5106 | 2.2976 | 0.5770 | 0.7023 |
| DBMS | 52 | Probability-weighted scorer | 18 | 1.4864 | 2.1891 | 0.6160 | 0.7641 |
| DBMS | 52 | Ridge regression | 18 | 1.4283 | 1.8787 | 0.7172 | 0.8166 |
| DBMS | 62 | Global mean baseline | 19 | 3.4981 | 3.7913 | -0.0052 | NA |
| DBMS | 62 | Predicted-category mean | 19 | 1.1602 | 2.2733 | 0.6386 | 0.8635 |
| DBMS | 62 | Probability-weighted scorer | 19 | 1.3890 | 2.1757 | 0.6690 | 0.8285 |
| DBMS | 62 | Ridge regression | 19 | 1.8468 | 2.1541 | 0.6755 | 0.8807 |
| DBMS | 72 | Global mean baseline | 19 | 3.2656 | 3.6062 | -0.0014 | NA |
| DBMS | 72 | Predicted-category mean | 19 | 2.5980 | 4.0915 | -0.2891 | 0.4028 |
| DBMS | 72 | Probability-weighted scorer | 19 | 2.3791 | 3.5550 | 0.0268 | 0.3997 |
| DBMS | 72 | Ridge regression | 19 | 2.1978 | 2.8726 | 0.3646 | 0.5977 |
| DBMS | 82 | Global mean baseline | 18 | 3.4253 | 3.7402 | -0.0034 | NA |
| DBMS | 82 | Predicted-category mean | 18 | 0.6085 | 0.6956 | 0.9653 | 0.9244 |
| DBMS | 82 | Probability-weighted scorer | 18 | 1.1590 | 1.5883 | 0.8191 | 0.8794 |
| DBMS | 82 | Ridge regression | 18 | 1.7013 | 2.1993 | 0.6531 | 0.8042 |

### Interpretation

- Lower MAE and RMSE are better.
- Lower standard deviation suggests more consistent performance across the tested splits, but does not prove future stability.
- Higher R-squared and Spearman generally indicate better fit and ranking, respectively.
- Mean prediction error near zero indicates less average bias.
- The small number of unique questions limits the reliability of both mean metrics and standard deviations.

### Limitations

- Five splits provide preliminary evidence, not definitive statistical validation.
- Overlapping test questions between repeated splits mean the five runs are not fully independent samples.
- Validation sets contain few unique questions, so selected Ridge alpha can be unstable.
- Category probabilities may not be calibrated.
- Results do not establish production readiness.

### Recommended decision rule

Prefer a scoring method only when its error metrics and split-to-split stability are meaningfully competitive. Do not choose a final scorer from one metric or one subject alone.
