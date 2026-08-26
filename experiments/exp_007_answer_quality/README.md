# Experiment 007A — Answer Quality Features

## Objective

Determine whether answer-level linguistic and structural
features improve the hybrid grading model.

The experiment builds on Experiment 006.

---

## Research Question

Can features derived directly from the student's answer improve
classification of difficult grading categories, particularly:

- Partially Correct
- Contradictory

without using human grading annotations as predictive features?

---

## Baseline

Experiment 006 combines:

- TF-IDF similarity
- semantic embedding similarity
- NLI features
- answer length

Experiment 007A retains these features and adds answer-quality
features.

---

## New Features

### Structural

- sentence count
- average words per sentence
- word count
- character count

### Lexical

- student/reference lexical overlap
- student/question lexical overlap
- reference keyword coverage
- unique-token ratio

### Linguistic

- negation count
- uncertainty / hedging signal count

### Length Relationships

- student/reference length ratio
- student/question length ratio

---

## Leakage Prevention

The following fields are NOT used as predictive features:

- label
- missing_concepts
- incorrect_concepts
- annotation_confidence

Only information available at grading time is used.

---

## Dataset Split

Question-level splitting is used to prevent answers to the same
question from appearing across training, validation, and test sets.

---

## Evaluation

Metrics:

- Accuracy
- Macro F1
- Weighted F1
- Per-class precision
- Per-class recall
- Per-class F1

Macro F1 is particularly important because all grading classes
should receive meaningful consideration.

---

## Comparison

The result will be compared directly against Experiment 006.

The primary comparison metrics are:

1. Test Macro F1
2. Partially Correct F1
3. Contradictory F1
4. Overall test accuracy

---

## Hypothesis

Answer-quality features may help distinguish:

- concise but correct answers
- incomplete answers
- uncertain answers
- contradictory answers
- irrelevant answers

However, these features may also introduce noise because writing
style and answer length are not inherently indicators of correctness.

---

## Status

Exploratory experiment.


Results
Dataset	Exp 006 Macro F1	Exp 007A Macro F1	Change
OS	         0.5672	             0.6415	        +0.0743
DBMS	     0.7010	             0.7333	        +0.0323

Accuracy also improves:

Dataset	Exp 006	 Exp 007A	Change
OS	     61.11%	 66.67%	   +5.56 pp
DBMS	 68.42%	 73.68%	 +5.26 pp