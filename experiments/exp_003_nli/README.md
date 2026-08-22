# Experiment 003 — NLI Baseline

## Objective

Determine whether explicit Natural Language Inference signals
improve technical answer classification over lexical and semantic
similarity baselines.

## Model

cross-encoder/nli-deberta-v3-base

## NLI signals

For both:

- Question → Student Answer
- Reference Answer → Student Answer

extract:

- Entailment probability
- Contradiction probability
- Neutral probability

## Classifier

Logistic Regression.

## Experimental control

The experiment uses the same:

- dataset
- question-level split
- random seed
- evaluation metrics

as Experiments 001 and 002.

## Research question

Does explicit entailment/contradiction information improve
classification of technical student answers?

## Important limitation

The NLI model is pretrained on general-domain NLI data rather
than OS/DBMS educational data. Its performance therefore needs
to be evaluated empirically on the target domain.