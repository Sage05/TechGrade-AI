# Experiment 004 — Concept Coverage Diagnostic

## Objective

Determine whether concept-level annotations in the revised
dataset provide useful evidence for technical answer grading.

## Inputs

The experiment uses:

- required_concepts
- missing_concepts
- incorrect_concepts
- label
- human_score

## Derived features

- required_count
- covered_count
- missing_count
- incorrect_count
- coverage_ratio
- missing_ratio
- incorrect_ratio

## Important limitation

The current revised dataset does not contain an automatically
predicted student_concepts field.

Therefore, this experiment is an annotation-based diagnostic.

It must NOT be interpreted as an automatic concept detection
system.

The derived concept features are intended to determine whether
concept coverage is associated with human grading outcomes.

## Research questions

1. Does higher concept coverage correspond to higher human scores?

2. Do Partially Correct answers exhibit intermediate concept
   coverage?

3. Do Incorrect or Contradictory answers exhibit higher
   incorrect-concept ratios?

4. Can concept-level information provide evidence that is
   complementary to semantic similarity and NLI?

## Future work

A later experiment will implement automatic concept matching
using only the question, required concepts, and student answer.

Those predicted concept features can then be evaluated as
inputs to the final hybrid grading model.