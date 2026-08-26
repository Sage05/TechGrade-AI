# Experiment 009 — Feature Ablation Study

## Objective

Determine which feature groups contribute useful and generalizable
information to technical student answer grading.

This experiment evaluates individual feature groups and combinations
using the same question-level train/validation/test methodology used
in the previous experiments.

The goal is to determine whether the additional answer-quality and
concept-aware features provide consistent improvements over the core
hybrid architecture.

---

## Feature Groups

The experiment evaluates five feature groups:

- TF-IDF lexical similarity
- Sentence embedding similarity
- NLI-based semantic signals
- Answer-quality features
- Automatically inferred concept-level evidence

No human grading annotations are used as predictive features.

---

## Ablation Configurations

### A — TF-IDF Only

Uses three lexical similarity features:

- question–student similarity
- reference–student similarity
- question–reference similarity

### B — Embeddings Only

Uses semantic similarity between:

- question and student answer
- reference and student answer
- question and reference answer

### C — NLI Only

Uses entailment, contradiction, and neutral probabilities for:

- question–student answer
- reference–student answer

### D — Answer-Quality Only

Uses features describing the structure and linguistic characteristics
of the student's answer, including:

- answer length
- sentence count
- average sentence length
- lexical overlap
- reference keyword coverage
- negation frequency
- uncertainty indicators
- vocabulary diversity
- answer/reference length ratio
- answer/question length ratio

### E — TF-IDF + Embeddings + NLI

Tests the core semantic/lexical feature combination without
answer-quality or concept features.

### F — Experiment 007A Full Hybrid

Combines:

- TF-IDF
- embeddings
- NLI
- answer-quality features

This represents the current general-purpose hybrid architecture.

### G — Concept Features Only

Tests automatically inferred concept-level evidence independently.

### H — Experiment 008 Full Concept-Aware

Combines all feature groups:

- TF-IDF
- embeddings
- NLI
- answer-quality
- concept-level evidence

---

## Evaluation Protocol

The experiment uses question-level splitting to prevent answers to the
same question from appearing across different dataset partitions.

Question leakage was verified for both datasets.

The primary evaluation metric is **Macro F1**, since the task contains
five grading classes and minority-class performance is important.

Secondary metrics are:

- Accuracy
- Weighted F1

The same configurations are evaluated on both the revised OS and
revised DBMS datasets.

---

# Results

## Revised OS

| Configuration | Features | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| TF-IDF only | 3 | 50.00% | 37.43% | 40.71% |
| Embeddings only | 3 | 50.00% | 36.91% | 40.56% |
| NLI only | 6 | 55.56% | 40.24% | 46.87% |
| **Answer-quality only** | **12** | **72.22%** | **69.86%** | **71.04%** |
| TF-IDF + Embeddings + NLI | 12 | 72.22% | 62.52% | 67.25% |
| **007A Full Hybrid** | **24** | **66.67%** | **64.15%** | **66.28%** |
| Concept features only | 12 | 72.22% | 62.94% | 67.60% |
| 008 Full Concept-Aware | 36 | 72.22% | 69.52% | 73.08% |

The strongest OS configuration by Macro F1 was **Answer-quality only**
at 69.86%, closely followed by the full concept-aware configuration
at 69.52%.

---

## Revised DBMS

| Configuration | Features | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| TF-IDF only | 3 | 57.89% | 38.07% | 45.57% |
| Embeddings only | 3 | 42.11% | 41.08% | 36.82% |
| NLI only | 6 | 52.63% | 36.24% | 44.75% |
| Answer-quality only | 12 | 47.37% | 42.58% | 45.45% |
| TF-IDF + Embeddings + NLI | 12 | 63.16% | 57.91% | 61.22% |
| **007A Full Hybrid** | **24** | **73.68%** | **73.33%** | **74.56%** |
| Concept features only | 12 | 31.58% | 30.48% | 30.64% |
| 008 Full Concept-Aware | 36 | 63.16% | 55.43% | 61.41% |

The strongest DBMS configuration by Macro F1 was **Experiment 007A Full
Hybrid**, with 73.33% Macro F1 and 73.68% accuracy.

---

## Cross-Dataset Findings

The results demonstrate substantial differences between the two subject
domains.

Answer-quality-only features performed strongly on OS:

- OS Macro F1: **69.86%**

but performed substantially worse on DBMS:

- DBMS Macro F1: **42.58%**

Similarly, concept-level features showed strong domain sensitivity.

The concept-aware model achieved:

- OS Macro F1: **69.52%**
- DBMS Macro F1: **55.43%**

In contrast, Experiment 007A achieved:

- OS Macro F1: **64.15%**
- DBMS Macro F1: **73.33%**

Therefore, Experiment 007A currently provides the most consistent
cross-domain performance.

---

## Key Finding

The ablation study shows that no individual feature group is
consistently strongest across both datasets.

Answer-quality features can provide substantial value, but their
performance is domain-sensitive.

Concept-level features also provide useful information in some cases,
but their standalone performance is poor on DBMS and their inclusion
in the full concept-aware model reduces DBMS performance compared with
Experiment 007A.

The core semantic/lexical/NLI combination performs more consistently
than the individual feature groups, while the addition of
answer-quality features produces the strongest overall DBMS result.

---

## Conclusion

Experiment 009 provides evidence that **feature diversity is more
useful than relying on a single feature family** for this grading task.

The current best general-purpose architecture remains:

> **Experiment 007A — TF-IDF + Embeddings + NLI + Answer-Quality Features**

The concept-aware architecture from Experiment 008 is retained as an
experimental branch because it improves OS performance but does not
generalize consistently to DBMS.

The results also show why evaluation across multiple subject domains is
necessary. A feature configuration that performs strongly on one
dataset cannot automatically be considered a robust grading architecture.

---

## Next Step

Experiment 010 will evaluate the leading configurations across repeated
question-level splits.

The objective is to determine whether the observed differences are
stable or are partly caused by the small test sets used in the current
experiments.

The main configurations to compare are:

- Experiment 007A Full Hybrid
- TF-IDF + Embeddings + NLI
- Answer-quality only
- Experiment 008 Full Concept-Aware