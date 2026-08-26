# Experiment 010 — Robustness / Stability Study

## Objective

Determine whether the performance differences observed in the previous
experiments remain stable across multiple question-level train/test
splits.

The previous experiments used a single fixed question-level split.
Because the datasets contain only a small number of unique questions,
performance can be sensitive to which questions are placed in the test
set.

Experiment 010 therefore evaluates the leading configurations across
multiple repeated question-level holdout splits.

---

## Configurations Evaluated

### E — TF-IDF + Embeddings + NLI

The core lexical and semantic feature combination.

### F — Experiment 007A Full Hybrid

Combines:

- TF-IDF
- embeddings
- NLI
- answer-quality features

This is the current general-purpose candidate.

### D — Answer-quality Only

Uses only answer-level structural and linguistic features.

### H — Experiment 008 Full Concept-Aware

Combines:

- TF-IDF
- embeddings
- NLI
- answer-quality features
- automatically inferred concept-level evidence

---

## Evaluation Protocol

The experiment uses repeated question-level holdout evaluation.

Each repetition creates a new train/test split based on question identity.
No question is allowed to appear in both the training and test sets.

Five fixed random seeds are used:

- 11
- 22
- 33
- 44
- 55

Approximately 20% of the unique questions are held out for testing in
each repetition.

All configurations use the same split for each repetition.

---

## Metrics

Primary metric:

- Macro F1

Secondary metrics:

- Accuracy
- Weighted F1

For each configuration, the experiment reports:

- mean performance
- standard deviation
- individual split results

A lower standard deviation indicates more stable performance across
question-level splits.

---

## Data Leakage

Question-level leakage is explicitly prevented.

Human concept annotations are not used as predictive features.

Only information available from the question, model answer, student
answer, and automatically generated model features is used.

---

## Results

### Revised OS

Results will be recorded here after execution.

### Revised DBMS

Results will be recorded here after execution.

---

## Interpretation

The primary objective is not simply to find the highest score on one
split.

A configuration should be considered more robust when it:

1. Maintains strong mean Macro F1 across repeated splits.
2. Has relatively low Macro F1 variance.
3. Performs consistently across different question subsets.
4. Does not depend strongly on one particular domain or split.

---

## Conclusion

The robustness experiment will determine whether Experiment 007A's
advantage over the other feature configurations is stable.

If Experiment 007A consistently performs well across repeated splits,
this will provide stronger evidence for selecting it as the core grading
architecture.

If another configuration performs similarly while showing lower
variance, that configuration may be preferable for the eventual system.

---

## Next Step

Use the robustness results to select the architecture for the next
stage of development and identify whether additional model tuning or
data collection is more valuable than adding further feature groups.

## Results

### Revised OS

Five repeated question-level holdout splits were evaluated.

| Configuration | Mean Accuracy | Accuracy Std | Mean Macro F1 | Macro F1 Std | Mean Weighted F1 |
|---|---:|---:|---:|---:|---:|
| **Exp 008 Full Concept-Aware** | **78.89%** | 15.91% | **75.58%** | 18.35% | **78.74%** |
| Exp 007A Full Hybrid | 76.67% | 16.39% | 73.11% | 18.09% | 76.32% |
| TF-IDF + Embeddings + NLI | 73.33% | **6.09%** | 65.97% | **6.77%** | 68.95% |
| Answer-quality only | 65.56% | 12.67% | 64.19% | 11.96% | 65.92% |

Exp 008 achieved the highest mean Macro F1, while the TF-IDF +
Embeddings + NLI configuration was substantially more stable across
splits.

---

### Revised DBMS

Five repeated question-level holdout splits were evaluated.

| Configuration | Mean Accuracy | Accuracy Std | Mean Macro F1 | Macro F1 Std | Mean Weighted F1 |
|---|---:|---:|---:|---:|---:|
| **Exp 008 Full Concept-Aware** | **70.88%** | 10.57% | **70.07%** | 16.32% | **70.08%** |
| Exp 007A Full Hybrid | 65.38% | **10.02%** | 64.38% | **12.15%** | 65.01% |
| TF-IDF + Embeddings + NLI | 55.91% | 6.92% | 53.14% | **3.04%** | 55.06% |
| Answer-quality only | 54.50% | 15.40% | 52.61% | 16.01% | 54.75% |

Exp 008 again achieved the highest mean Macro F1.

---

## Cross-Dataset Robustness

The repeated-split evaluation provides stronger evidence than the
single-split experiments.

Mean Macro F1 across both datasets:

| Configuration | OS | DBMS | Cross-Dataset Mean |
|---|---:|---:|---:|
| **Exp 008 Full Concept-Aware** | 75.58% | 70.07% | **72.83%** |
| Exp 007A Full Hybrid | 73.11% | 64.38% | 68.75% |
| TF-IDF + Embeddings + NLI | 65.97% | 53.14% | 59.56% |
| Answer-quality only | 64.19% | 52.61% | 58.40% |

Exp 008 therefore produced the strongest average performance across
both subject domains.

However, its Macro F1 standard deviation was considerably higher than
the simpler TF-IDF + Embeddings + NLI configuration.

This indicates a performance-versus-stability tradeoff rather than a
uniform improvement.

---

## Key Finding

Experiment 010 changes the interpretation of the previous experiments.

Experiment 009 showed that individual feature groups were highly
domain-sensitive. Experiment 010 demonstrates that combining these
feature groups into the concept-aware architecture can produce stronger
average performance across repeated question-level splits.

The **Exp 008 Concept-Aware Hybrid** achieved the highest mean Macro F1
on both datasets:

- OS: 75.58%
- DBMS: 70.07%

It also achieved the highest combined cross-dataset mean of 72.83%.

At the same time, the model exhibited greater variation between splits.
For example, on OS it achieved Macro F1 values ranging from 49.04% to
100.00%.

The simpler TF-IDF + Embeddings + NLI configuration was considerably
more stable:

- OS Macro F1 standard deviation: 6.77%
- DBMS Macro F1 standard deviation: 3.04%

Therefore, the current results indicate a tradeoff between peak
performance and stability.

---

## Architecture Decision

Based on the current experiments, **Experiment 008 is selected as the
primary research architecture** because it provides the strongest mean
performance across both evaluated domains.

Experiment 007A remains an important simpler baseline because it uses
fewer features and provides competitive performance, particularly on
DBMS.

The TF-IDF + Embeddings + NLI configuration is retained as the stable
semantic baseline.

These results should not be interpreted as statistically conclusive
evidence that Exp 008 is superior. The datasets contain only 14–15
unique questions, and five repeated holdout splits remain a relatively
small evaluation.

Additional data and evaluation questions will be required before making
a final production-model claim.

---

## Conclusion

Experiment 010 provides evidence that the concept-aware architecture
generalizes better on average across the current OS and DBMS datasets
than the simpler alternatives.

The current ranking is:

1. **Exp 008 — Concept-Aware Hybrid**
2. **Exp 007A — Full Hybrid**
3. **TF-IDF + Embeddings + NLI**
4. **Answer-quality only**

The next development stage should therefore focus on validating and
strengthening the Exp 008 architecture rather than continuing to add
arbitrary feature groups.