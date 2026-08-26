# Experiment 008 — Concept-Aware Hybrid Grader

## Objective

Determine whether automatically inferred concept-level evidence
improves the hybrid grading model.

The experiment extends Experiment 007A.

---

## Research Question

Can automatically inferred concept-level evidence improve
classification of technical student answers without using human
concept annotations?

---

## Input Information

The model may use:

- question text
- model/reference answer
- student answer
- required concepts

The following fields are never used as predictive features:

- label
- missing_concepts
- incorrect_concepts
- human_score

---

## Concept Evidence

The existing `ConceptMatcher` is used.

For each required concept it identifies:

- student evidence
- reference evidence
- semantic similarity
- NLI entailment
- NLI contradiction
- NLI neutral

The final thresholded concept decision is not used as a feature.

Instead, continuous evidence signals are aggregated across the
required concepts.

---

## New Feature Group

Concept-level aggregate features include:

- mean semantic similarity
- maximum semantic similarity
- minimum semantic similarity
- mean entailment
- maximum entailment
- mean contradiction
- maximum contradiction
- mean neutral probability
- fraction of concepts with strong entailment
- fraction of concepts with strong contradiction
- concept similarity spread
- number of required concepts

---

## Baseline

Experiment 007A combines:

- TF-IDF
- semantic embeddings
- NLI
- answer-quality features

Experiment 008 adds automatic concept evidence to this feature set.

---

## Leakage Prevention

Human concept annotations are excluded from the feature pipeline.

In particular:

- `missing_concepts` is not used
- `incorrect_concepts` is not used

These fields remain available only for future evaluation/diagnostic
analysis.

---

## Dataset Split

Question-level splitting is used.

No question should occur in more than one of:

- training
- validation
- test

---

## Evaluation

Primary metrics:

- Test Macro F1
- Test Accuracy
- Weighted F1

Secondary analysis:

- Partially Correct F1
- Contradictory F1
- Incorrect F1
- feature importance

---

## Hypothesis

Concept-level evidence should provide additional information about
which required ideas are present, absent, or contradicted in a
student answer.

However, because the automatic concept matcher is imperfect,
concept features may also introduce noise.

The experiment therefore tests whether the aggregate concept signals
provide useful information beyond the existing hybrid features.

---

## Status

Exploratory experiment.

The model should only be considered an improvement over Experiment
007A if performance improves on both revised OS and revised DBMS
datasets without introducing annotation leakage.


---

## Results

### Revised OS Dataset

- Test Accuracy: **72.22%**
- Test Macro F1: **69.52%**
- Test Weighted F1: **73.08%**

Compared with Experiment 007A:

- Accuracy improved from **66.67% → 72.22%**
- Macro F1 improved from **64.15% → 69.52%**

The concept-aware features provided useful additional information on the OS dataset.

### Revised DBMS Dataset

- Test Accuracy: **63.16%**
- Test Macro F1: **55.43%**
- Test Weighted F1: **61.41%**

Compared with Experiment 007A:

- Accuracy decreased from **73.68% → 63.16%**
- Macro F1 decreased from **73.33% → 55.43%**

The DBMS experiment therefore shows that the automatically inferred
concept features did not generalize consistently across domains.

---

## Cross-Dataset Findings

Experiment 008 produced different results across the two subject
domains.

The concept-level features were among the more influential features
in both experiments, including:

- concept mean similarity
- concept maximum similarity
- concept minimum similarity
- concept mean contradiction
- concept maximum contradiction
- strong contradiction fraction
- strong entailment fraction
- concept mean entailment

However, their contribution was not consistently beneficial.

The OS dataset showed a substantial improvement over Experiment 007A,
while the DBMS dataset showed a substantial degradation.

In particular, the DBMS test set showed:

- Contradictory F1: **0.00**
- Incorrect F1: **0.44**
- Partially Correct F1: **0.60**

This suggests that the current automatic concept evidence pipeline may
introduce domain-sensitive noise when combined with the hybrid grading
features.

---

## Conclusion

Experiment 008 demonstrates that automatically inferred concept
evidence can provide useful grading information, but the current
concept-aware feature design does not demonstrate consistent
cross-domain improvement.

Therefore, Experiment 008 is **not adopted as the final grading
architecture**.

Experiment 007A remains the stronger general-purpose candidate at this
stage because it provides more consistent performance across both
revised OS and DBMS datasets.

The negative result is retained as an important finding rather than
discarded.

---

## Next Step

Experiment 009 will perform a feature ablation study to determine which
feature groups contribute positively to grading performance and whether
the concept-level features provide independent information or introduce
redundancy/noise.

The analysis will compare:

- TF-IDF features
- embedding features
- NLI features
- answer-quality features
- concept-level features
- combinations of these feature groups

The goal is to identify a simpler and more robust feature configuration
before final model selection.