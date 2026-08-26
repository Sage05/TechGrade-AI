# Experiment 006 — Hybrid Grader

## Objective

Evaluate whether combining multiple independent signals improves
classification of technical student answers over the individual
baseline models.

The hybrid classifier combines:

1. TF-IDF similarity
2. Semantic embedding similarity
3. NLI features
4. Concept-level coverage signal
5. Basic answer statistics

The target classes are:

- Correct
- Incorrect
- Partially Correct
- Irrelevant
- Contradictory

---

## Experimental Design

The dataset is divided at the question level into:

- Training
- Validation
- Test

This prevents answers to the same question from appearing in
different splits.

Question leakage is explicitly checked before training.

---

## Feature Groups

### 1. TF-IDF

The hybrid model uses:

- question ↔ student similarity
- reference ↔ student similarity
- question ↔ reference similarity

### 2. Semantic Embeddings

The hybrid model uses:

- question ↔ student similarity
- reference ↔ student similarity
- question ↔ reference similarity

The embedding model is:

    sentence-transformers/all-MiniLM-L6-v2

### 3. NLI

The hybrid model uses NLI relationships between:

- question ↔ student
- reference ↔ student

The NLI model is:

    cross-encoder/nli-deberta-v3-base

### 4. Concept Coverage

Concept coverage represents how many required concepts appear to
be satisfied by the student's answer.

### 5. Answer Statistics

Basic answer-level features include:

- word count
- character count

---

# Important Methodological Distinction

The current implementation contains a concept-coverage feature
derived from human concept annotations.

Specifically:

- missing_concepts
- incorrect_concepts

are used to calculate an upper-bound concept coverage signal.

This is NOT deployable because those annotations are unavailable
when grading a new student answer.

Therefore, this experiment is divided conceptually into two stages.

---

# Stage A — Upper-Bound Hybrid

The first run uses human concept annotations to determine the
maximum potential contribution of concept coverage.

This answers:

> How much could concept coverage improve the classifier if
> concept coverage were perfectly available?

This is a diagnostic experiment only.

It must not be presented as the final deployable TechGrade AI
architecture.

---

# Stage B — Deployable Hybrid

The second run will remove all human concept annotations from the
feature pipeline.

The deployable version may use only information available when
grading a new answer, such as:

- question
- model/reference answer
- student answer
- required concepts

This provides the fair comparison against the individual
baselines.

---

# Leakage Prevention

The following fields must NEVER be used as predictive features:

- label
- human_score
- missing_concepts
- incorrect_concepts

when evaluating the deployable system.

The target `label` is used only for supervised training and
evaluation.

---

# Classifier

The hybrid feature vector is classified using:

    Logistic Regression

with class balancing enabled.

This keeps the classifier consistent with the supervised
classification baselines used in earlier experiments.

---

# Evaluation Metrics

The experiment reports:

- Accuracy
- Macro F1
- Weighted F1
- Per-class precision
- Per-class recall
- Per-class F1

Macro F1 is particularly important because the five grading
classes are not perfectly balanced.

---

# Feature Importance

The experiment also reports the average absolute Logistic
Regression coefficient for each feature.

This provides an initial indication of which feature groups
contribute most strongly to the classifier.

Feature importance should be interpreted cautiously because
correlated features can distribute their predictive contribution
across multiple coefficients.

---

# Research Questions

### RQ1

Does combining multiple grading signals outperform the individual
TF-IDF, embedding, and NLI baselines?

### RQ2

Does concept coverage provide additional predictive value?

### RQ3

How much performance improvement is attributable to concept-level
information?

### RQ4

Which feature groups contribute most strongly to the hybrid
classifier?

---

# Expected Outcome

The purpose of this experiment is not to assume that the hybrid
model will outperform every baseline.

If the hybrid performs better, this provides evidence that the
signals are complementary.

If it does not, the results will identify which signals fail to
provide additional predictive value.

---

# Current Status

Experiment 006 is an exploratory hybrid-classification experiment.

The upper-bound concept feature is not considered deployable.

A leakage-free hybrid variant will be evaluated separately before
any final architecture is selected.

Final Exp 006 results so far
Dataset	Hybrid Accuracy	Hybrid Macro F1	Hybrid Weighted F1
OS	        61.11%	        56.72%	        60.09%
DBMS	    68.42%	        70.10%	        69.48%