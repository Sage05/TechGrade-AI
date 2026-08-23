# Experiment 005 — Automatic Concept Matching

## Objective

Evaluate whether TechGrade AI can automatically determine whether
required concepts are:

- COVERED
- MISSING
- INCORRECT

from a student's answer.

This experiment evaluates several automatic concept-matching
strategies using semantic embeddings and Natural Language Inference.

---

## Dataset

The experiment was evaluated on revised sample datasets:

- OS: 84 answers, 14 questions
- DBMS: 92 answers, 15 questions

Concept-level annotations were used only as evaluation ground truth.

---

# Experiment Variants

## 005.0 — Initial Concept Matching

The initial approach combined:

- MiniLM semantic similarity
- DeBERTa NLI
- heuristic thresholds

Concept names were directly compared with student answers.

### OS result

Accuracy:

    47.14%

The model showed a strong bias toward predicting MISSING.

---

## 005.1 — NLI Probability Correction

DeBERTa raw logits were converted to probabilities using
softmax before applying thresholds.

### OS result

Accuracy:

    32.86%

Although the NLI probabilities were technically interpreted
correctly, performance decreased.

This demonstrated that the primary problem was not simply
logit calibration.

---

## 005.2 — Reference-Grounded Matching

The model answer was introduced as a reference.

For each required concept, the system selected the most
semantically relevant sentence from:

- the model answer
- the student answer

DeBERTa then compared the selected student evidence against
the reference evidence.

### OS result

Accuracy:

    44.29%

This improved over 005.1, demonstrating that reference
information provides useful signal.

However, multiple concepts frequently appeared in the same
reference sentence, preventing reliable concept-specific
matching.

---

## 005.3 — Concept-Specific Context Window

The system was modified to use a context window around the
sentence most relevant to each concept.

A higher contradiction threshold was also introduced to
reduce false INCORRECT predictions.

### OS result

Accuracy:

    47.14%

The improvement over 005.2 was small.

The same reference sentence was still frequently selected
for multiple concepts.

---

# Final Results

| Variant | OS Accuracy |
|---|---:|
| 005.0 Initial | 47.14% |
| 005.1 NLI Probability | 32.86% |
| 005.2 Reference Grounded | 44.29% |
| 005.3 Context Window | 47.14% |

---

# Findings

The experiments demonstrate that generic semantic similarity
and general-domain NLI are insufficient for reliable automatic
concept-level grading when concepts are represented only as
short labels.

A major failure mode occurs when several required concepts are
described together in the same model-answer sentence.

For example, a reference sentence may simultaneously describe:

- Mutual Exclusion
- Hold and Wait
- No Preemption
- Circular Wait

Sentence-level retrieval therefore cannot reliably determine
which portion of the reference corresponds to an individual
concept.

---

# Dataset Requirements Identified

The experiments indicate that the revised dataset should
preferably contain concept-level reference information.

For example:

    Concept:
    Mutual Exclusion

    Reference:
    A resource can be held by only one process at a time.

This allows the grading model to evaluate the student's evidence
against a concept-specific reference rather than against an entire
multi-concept sentence.

The dataset should also contain substantially more incorrect
concept examples.

The current OS sample contains only 5 INCORRECT concept instances,
which is insufficient for robust evaluation of contradiction
detection.

---

# Conclusion

Automatic concept matching remains a promising component of
TechGrade AI, but the tested generic approaches are not sufficiently
reliable to be used directly as the final grading mechanism.

Experiment 005 therefore serves as an exploratory baseline and
failure-analysis experiment.

The results motivate a concept-level dataset revision before
further development of automatic concept grading.

No automatic concept matcher from this experiment is considered
production-ready.