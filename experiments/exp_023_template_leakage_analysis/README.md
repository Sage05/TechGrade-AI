# Experiment 023: Synthetic Template Leakage and Generalisation Analysis

## 1. Objective

Investigate whether the perfect classification accuracy reported in Experiment 022 reflects genuine technical-answer evaluation or linguistic shortcuts introduced during synthetic dataset generation.

## 2. Datasets

Two Synthetic v2 datasets were evaluated:

| Property | OS | DBMS |
|---|---:|---:|
| Total samples | 1,000 | 1,000 |
| Unique question IDs | 33 | 30 |
| Unique question texts | 1,000 | 1,000 |
| Unique student answers | 211 | 43 |
| Classes | 5 | 5 |
| Samples per class | 200 | 200 |

Both datasets are synthetic and have not been independently human-validated.

## 3. Methodology

Five diagnostic analyses were conducted:

1. Dataset duplication and question-ID analysis.
2. Question-grouped five-fold cross-validation using majority-class, question-only, and answer-only baselines.
3. Evaluation on previously unseen exact answer strings and cross-domain transfer.
4. Inspection of label-associated TF-IDF word features.
5. Evaluation of a classifier using manually selected linguistic cues.

All classifiers in Experiments 023's baseline and transfer analyses were separate from the 36-feature TechGradeAI hybrid grader.

## 4. Results

### 4.1 Dataset Audit

OS contained 211 unique student answers, while DBMS contained only 43.

All 30 DBMS question IDs were associated with exactly one classification label. In contrast, all 33 OS question IDs contained multiple labels.

Each dataset contained 1,000 unique question texts despite having substantially fewer question IDs.

### 4.2 Baseline Classification

Five-fold question-grouped cross-validation produced:

| Classifier | OS Accuracy | DBMS Accuracy |
|---|---:|---:|
| Majority class | 19.80% | 0.00% |
| Question only | 13.61% | 55.00% |
| Answer only | 100.00% | 100.00% |

The answer-only classifier used character-level TF-IDF features and logistic regression. It received neither reference answers nor required concepts.

### 4.3 Unseen Answers and Cross-Domain Transfer

Answer-only accuracy on previously unseen exact answer strings remained 100% across all five folds for both subjects.

Cross-domain evaluation produced:

| Training domain | Testing domain | Accuracy |
|---|---|---:|
| OS | DBMS | 100.00% |
| DBMS | OS | 99.20% |

On test answers not occurring verbatim in the source training dataset, accuracy was 100% for OS-to-DBMS and 99% for DBMS-to-OS.

These results indicate that answer-level predictability extends beyond exact-string memorisation.

### 4.4 Linguistic Shortcut Analysis

Inspection of word-level TF-IDF coefficients identified strongly label-associated phrases:

| Label | Examples |
|---|---|
| Correct | "in short", "the core idea" |
| Partially Correct | "leaves out", "important conditions" |
| Incorrect | "regardless of correctness", "ignores the relevant" |
| Contradictory | "this contradicts", "opposite behavior" |
| Irrelevant | "topic is interesting", "many different tools" |

The selected diagnostic phrases had 100% label purity in both datasets.

A classifier restricted to the 19 selected phrase indicators achieved 100% mean grouped cross-validation accuracy in both subjects.

Because these cues were selected after examining the same datasets, this result is exploratory rather than an independent validation benchmark.

### 4.5 Class Coverage

The DBMS question-grouped splits frequently lacked one or more grading classes in their test folds.

Consequently, accuracy and five-class macro F1 must be interpreted separately. A perfect accuracy score on a subset containing fewer than five classes does not demonstrate performance across all five labels.

## 5. Interpretation

The evidence strongly supports the presence of label-associated linguistic shortcuts in Synthetic v2.

The datasets permit near-perfect or perfect classification from student-answer wording alone, without requiring the technical question, reference answer, or grading concepts.

This substantially limits the interpretation of Experiment 022's perfect accuracy.

The results do not establish that TechGradeAI's underlying grading architecture is defective. They establish that Synthetic v2 is not an adequate independent benchmark for real-world grading generalisation.

## 6. Limitations

- Synthetic data was not independently human-validated.
- Exact-answer novelty does not guarantee template or semantic novelty.
- Grouping by question ID may not separate related question texts or generation templates.
- Diagnostic phrases were selected after inspecting the datasets.
- Cross-domain transfer involved two datasets that may share generation procedures.
- No independent real-student answer benchmark was evaluated.
- No controlled intervention established the causal effect of individual phrases.

## 7. Conclusion

Experiment 023 identified substantial label-associated linguistic shortcuts in the Synthetic v2 datasets.

Answer-only classifiers achieved perfect grouped cross-validation accuracy. Performance remained near-perfect across domains, and a small set of label-associated phrases was sufficient for perfect classification in exploratory tests.

Therefore, Experiment 022's results must not be interpreted as evidence of real-world technical-answer grading generalisation.

## 8. Recommendations

Future evaluation should prioritize:

1. Independently reviewed student-style answers.
2. Multiple grading categories per question.
3. Linguistic diversity without label-revealing phrases.
4. Question, answer-template, and semantic-duplicate controls.
5. Separate development and untouched final test sets.
6. Per-class precision, recall, macro F1, and confusion matrices.
7. Comparison against simple text-only baselines.

## 9. Next Experiment

Experiment 024: Benchmark Redesign and Realistic Answer Evaluation.

The next study will develop a smaller, more credible evaluation benchmark before further claims about model generalisation.

## 10. Reproducibility

Scripts:

- run_exp023.py
- run_exp023_baselines.py
- run_exp023_transfer.py
- run_exp023_interpretability.py
- run_exp023_cue_analysis.py

All experiments use the existing OS and DBMS Synthetic v2 CSV datasets.

Generated numerical results are stored locally in the `results/` directory. CSV and JSON result files are excluded from version control by the repository's `.gitignore` rules.
