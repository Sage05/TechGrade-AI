# TechGrade AI

## Automated Technical Answer Evaluation using NLP

TechGrade AI is a research-oriented system for the automated evaluation of descriptive technical answers in Computer Science and Information Technology subjects.

The project investigates how NLP, semantic similarity, Natural Language Inference (NLI), answer-quality analysis, and concept-level reasoning can be combined to evaluate technical student answers beyond simple keyword matching.

The project has progressed through a sequence of controlled experiments to identify an effective and generalizable grading architecture.

---

## Problem

Manual evaluation of descriptive technical answers is:

- Time-consuming, particularly for large student cohorts.
- Subjective, resulting in variation between evaluators.
- Difficult to scale across multiple technical subjects.
- Challenging to automate because technically correct answers may use wording that differs substantially from reference answers.
- Difficult to evaluate using keyword matching alone because an answer can contain relevant terminology while still being technically incorrect.

A technically meaningful answer may:

- Express the same concept using different terminology.
- Cover only some of the concepts required by a question.
- Contain technically incorrect or contradictory statements.
- Be relevant but incomplete.
- Be structurally weak or uncertain despite containing some correct information.

TechGrade AI therefore investigates a multi-factor evaluation framework that combines semantic, linguistic, and concept-level evidence.

---

## Current Research Status

The project has completed 10 controlled experiments covering baseline models, concept matching, hybrid grading, answer-quality features, feature ablation, and robustness evaluation.

The current research architecture is:

> Experiment 008 — Concept-Aware Hybrid Grader

It combines:

- TF-IDF similarity
- Sentence embeddings
- Natural Language Inference
- Answer-quality features
- Concept-level evidence matching

Experiment 007A remains an important simpler hybrid baseline.

The architecture is not considered production-ready yet. Further evaluation on larger and more diverse datasets is required before making claims about production-level grading accuracy.

---

# Research Progression

```text
Exp 001–004
    │
    ├── Baselines / diagnostics
    │
    ▼
Exp 005
    │
    └── Automatic concept matching
    │
    ▼
Exp 006
    │
    └── Hybrid grader
    │
    ▼
Exp 007A
    │
    └── Answer-quality features
    │
    ▼
Exp 008
    │
    └── Concept-aware hybrid grader
    │
    ▼
Exp 009
    │
    └── Feature ablation study
    │
    ▼
Exp 010
    │
    └── Robustness / stability study
    │
    ▼
Current Architecture
    │
    └── Exp 008 Concept-Aware Hybrid
```

---

# Current Architecture

The current research architecture combines several complementary sources of evidence.

```text
                    Question
                       │
              Reference Answer
                       │
                       ▼
              ┌─────────────────┐
              │ Feature Extractor│
              └────────┬────────┘
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
    TF-IDF        Embeddings           NLI
    Similarity    Similarity       Entailment /
                                   Contradiction /
                                   Neutral
       │               │                │
       └───────────────┼────────────────┘
                       │
              Answer-Quality Features
                       │
                       ▼
             Concept-Level Evidence
                       │
                       ▼
             ┌───────────────────┐
             │ Hybrid Classifier │
             └─────────┬─────────┘
                       │
                       ▼
                 Grade / Label
```

The current classifier predicts one of five answer-quality classes:

- `Correct`
- `Incorrect`
- `Partially Correct`
- `Irrelevant`
- `Contradictory`

---

# Feature Groups

## 1. TF-IDF Similarity

Traditional lexical similarity features provide a lightweight baseline for comparing:

- Question ↔ Student answer
- Reference answer ↔ Student answer
- Question ↔ Reference answer

These features provide lexical evidence but do not reliably capture semantic equivalence.

---

## 2. Sentence Embeddings

Sentence-transformer embeddings are used to measure semantic similarity between:

- Question and student answer
- Reference answer and student answer
- Question and reference answer

The current experiments use:

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

## 3. Natural Language Inference

NLI provides semantic relationship signals between the student answer and the question/reference answer.

The current experiments use:

```text
cross-encoder/nli-deberta-v3-base
```

The extracted signals include:

- Entailment
- Contradiction
- Neutrality

Contradiction is particularly important because semantic similarity alone can incorrectly treat technically incorrect answers as good matches.

---

## 4. Answer-Quality Features

The system also extracts structural and linguistic characteristics of the student's answer, including:

- Answer length
- Sentence count
- Average words per sentence
- Lexical overlap
- Reference keyword coverage
- Negation frequency
- Uncertainty indicators
- Unique-token ratio
- Answer/reference length ratio
- Answer/question length ratio

These features provide information that pure semantic similarity cannot capture.

---

## 5. Concept-Level Evidence

The concept-aware component identifies evidence relevant to required technical concepts.

For each required concept, the system attempts to determine whether the student answer:

```text
Concept
   │
   ├── Relevant evidence found?
   │
   ├── Semantically supported?
   │
   └── Contradicted?
        │
        ▼
   COVERED / MISSING / INCORRECT
```

Concept-level statistics are then supplied to the hybrid classifier.

---

# Experimental Results

The experiments were evaluated on two revised datasets:

- OS
- DBMS

Question-level splitting was used to prevent answers belonging to the same question from appearing across different partitions.

This is important because random row-level splitting could allow the model to indirectly see the same question during training and testing.

---

## Experiment 009 — Feature Ablation

The ablation study showed that no individual feature family was consistently strongest across both domains.

On OS, answer-quality-only features achieved:

```text
Accuracy: 72.22%
Macro F1: 69.86%
```

On DBMS, the full Experiment 007A hybrid achieved:

```text
Accuracy: 73.68%
Macro F1: 73.33%
```

The results demonstrated significant domain sensitivity between individual feature groups.

---

# Experiment 010 — Robustness Study

Experiment 010 evaluated the leading configurations across five repeated question-level holdout splits.

### Revised OS

| Configuration | Mean Macro F1 | Macro F1 Std |
|---|---:|---:|
| **Exp 008 Concept-Aware** | **75.58%** | 18.35% |
| Exp 007A Full Hybrid | 73.11% | 18.09% |
| TF-IDF + Embeddings + NLI | 65.97% | **6.77%** |
| Answer-quality only | 64.19% | 11.96% |

### Revised DBMS

| Configuration | Mean Macro F1 | Macro F1 Std |
|---|---:|---:|
| **Exp 008 Concept-Aware** | **70.07%** | 16.32% |
| Exp 007A Full Hybrid | 64.38% | 12.15% |
| TF-IDF + Embeddings + NLI | 53.14% | **3.04%** |
| Answer-quality only | 52.61% | 16.01% |

### Cross-Dataset Mean

| Configuration | OS | DBMS | Mean |
|---|---:|---:|---:|
| **Exp 008 Concept-Aware** | 75.58% | 70.07% | **72.83%** |
| Exp 007A Full Hybrid | 73.11% | 64.38% | 68.75% |
| TF-IDF + Embeddings + NLI | 65.97% | 53.14% | 59.56% |
| Answer-quality only | 64.19% | 52.61% | 58.40% |

These results currently make Experiment 008 the strongest research candidate.

However, the concept-aware architecture also exhibits higher split-to-split variance.

The simpler TF-IDF + Embeddings + NLI configuration is considerably more stable, despite achieving lower average performance.

This represents an important performance-versus-stability tradeoff.

---

# Evaluation Methodology

The experiments use question-level data separation.

For the current robustness experiments:

- Five repeated question-level holdout splits are used.
- Approximately 20% of unique questions are held out per split.
- The same splits are used across configurations.
- Question leakage is explicitly checked.
- Macro F1 is the primary metric.
- Accuracy and Weighted F1 are secondary metrics.

Macro F1 is emphasized because the grading task contains multiple classes and minority-class performance is important.

---

# Current Limitations

The current results should be considered research findings rather than production-level validation.

### Dataset Size

The current datasets contain only a small number of unique questions:

```text
OS:   14 questions
DBMS: 15 questions
```

This makes performance sensitive to the particular questions included in each evaluation split.

### Limited Domains

Current evaluation is focused on Computer Science / IT technical subjects represented by the available OS and DBMS datasets.

Additional subjects are required to evaluate broader generalization.

### Human Agreement

A substantially larger human-graded dataset is required to determine how closely automated predictions agree with human evaluators.

### Statistical Confidence

The repeated robustness experiment provides stronger evidence than a single train/test split, but five repetitions over small datasets are not sufficient to establish definitive statistical superiority.

### OCR / Handwriting

The current research pipeline operates on extracted text.

Handwritten student answers are not yet directly processed by the grading model.

---

# Future System Architecture

The eventual system is intended to separate answer ingestion from answer evaluation.

A future architecture can therefore be structured as:

```text
        Student Answer Sheet
                 │
                 ▼
        OCR / Computer Vision
                 │
                 ▼
          Extracted Text
                 │
                 ▼
          SQL Database
                 │
       ┌─────────┴─────────┐
       │                   │
       ▼                   ▼
    Question          Student Answer
    Metadata              Data
       │                   │
       └─────────┬─────────┘
                 ▼
        TechGrade Grading
             Pipeline
                 │
                 ▼
       Concept-Aware Hybrid
              Model
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
      Grade    Label   Confidence
                 │
                 ▼
        Explainable Feedback
```

This database and OCR/CV architecture is a future implementation direction. It is not required for the current NLP research phase.

---

# Repository Structure

```text
TechGrade-AI/
│
├── experiments/
│   ├── exp_001_*/
│   ├── exp_002_*/
│   ├── exp_003_nli/
│   ├── exp_004_*/
│   ├── exp_005_concept_matching/
│   ├── exp_006_hybrid_grader/
│   ├── exp_007_answer_quality/
│   ├── exp_008_concept_aware_grader/
│   ├── exp_009_feature_ablation/
│   └── exp_010_robustness/
│
├── src/
│   └── models/
│       ├── concept_coverage.py
│       ├── concept_matcher.py
│       ├── embedding_baseline.py
│       ├── embedding_classifier.py
│       ├── nli_baseline.py
│       ├── nli_classifier.py
│       ├── tfidf_baseline.py
│       └── tfidf_classifier.py
│
├── README.md
└── ...
```

Each experiment is kept separately so that model development and experimental conclusions remain reproducible and traceable.

---

# Running an Experiment

Activate the project virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run an experiment using the module path and dataset:

```powershell
python -m experiments.exp_010_robustness.run_experiment `
    "C:\path\to\revised_sample_os.csv"
```

For example:

```powershell
python -m experiments.exp_010_robustness.run_experiment `
    "C:\Users\Ayuj Shingte\Downloads\revised_sample_os.csv"
```

---

# Research Direction

The immediate research direction is no longer to continuously add features.

The next stage is to:

1. Freeze the current Exp 008 research architecture.
2. Refactor it into a reusable grading pipeline.
3. Expand the number of questions and subjects.
4. Introduce larger human-graded datasets.
5. Evaluate agreement between human and automated grading.
6. Develop interpretable scoring and feedback.
7. Integrate the grading pipeline with a database-backed system.
8. Add OCR / computer-vision answer ingestion as a separate upstream component.

The long-term objective is a system capable of converting a student's technical answer into a structured evaluation consisting of:

```text
Student Answer
      │
      ▼
Concept / Semantic Analysis
      │
      ▼
Technical Evaluation
      │
      ├── Correct
      ├── Partially Correct
      ├── Incorrect
      ├── Irrelevant
      └── Contradictory
      │
      ▼
Score + Confidence
      │
      ▼
Explainable Feedback
```

---

## Status

**Research prototype — active development**

Current primary architecture:

> **Experiment 008 — Concept-Aware Hybrid Grader**

Current strongest evidence:

> **72.83% mean Macro F1 across the current OS and DBMS robustness evaluations**

The system is not yet considered production-ready. Larger datasets, human-agreement evaluation, and further validation are required before deployment claims can be made.
