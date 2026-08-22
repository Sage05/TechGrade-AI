# Experiment 001 — TF-IDF Baseline

## Objective

Establish a simple lexical baseline for technical answer evaluation.

The experiment measures the relationship between:

1. Question and student answer
2. Reference answer and student answer
3. Question + reference answer and student answer

## Why this baseline?

The purpose is not to create the final evaluator.

It provides a simple benchmark against which more advanced semantic models can be compared.

## Inputs

- Question
- Reference/model answer
- Student answer

## Outputs

- Question–student similarity
- Reference–student similarity
- Question+reference–student similarity

## Future comparison

This baseline will eventually be compared against:

- Sentence embedding models
- Cross-encoders
- NLI models
- Concept-aware models
- Hybrid models

## Dataset

The current OS and DBMS datasets are being used only for development and baseline testing.

They are not considered the final dataset.