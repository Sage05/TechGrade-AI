# TechGrade AI

## Automated Technical Answer Evaluation using NLP

TechGrade AI is a research-oriented system for the automated evaluation of descriptive technical answers in Computer Science and Information Technology subjects.

The project investigates how NLP, semantic similarity, Natural Language Inference (NLI), answer-quality analysis, and concept-level evidence can be combined to evaluate technical student answers beyond simple keyword matching.

The project is being developed as a sequence of controlled experiments followed by a reusable application layer. The research model and the application architecture are deliberately kept separate so that experimental findings remain reproducible while the application can evolve independently.

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

TechGrade AI therefore uses a multi-factor evaluation framework combining lexical, semantic, linguistic, and concept-level evidence.

---

# Current Research Status

The project has progressed through a sequence of controlled experiments covering lexical baselines, semantic embeddings, NLI, concept matching, hybrid grading, answer-quality features, feature ablation, robustness, grading-policy analysis, and convergence diagnostics.

The primary research candidate remains:

> **Experiment 008 — Concept-Aware Hybrid Grader**

The Exp 008 representation contains **36 features** from:

- TF-IDF similarity
- Sentence embeddings
- Natural Language Inference
- Answer-quality features
- Concept-level evidence

The overall classifier is:

```python
LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42,
)
```

The Exp 008 classifier predicts five classes:

- `Correct`
- `Incorrect`
- `Partially Correct`
- `Irrelevant`
- `Contradictory`

The Exp 008 research configuration is intentionally preserved unchanged for reproducibility. Application-level improvements are built around it rather than silently modifying the research experiment.

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
Exp 011
    │
    └── Grading-policy analysis
    │
    ▼
Exp 012
    │
    └── Logistic Regression convergence diagnostic
    │
    ▼
Application Layer
    │
    ├── HybridGrader
    ├── ConceptEvidenceEvaluator
    └── GradingService
    │
    ▼
Next
    │
    ├── 0–10 scoring
    ├── Explainable feedback
    └── Final OS + DBMS validation
```

---

# Current Architecture

The current system separates research-time feature extraction from the application-facing grading interface.

```text
Question + Reference Answer + Student Answer
                    │
                    ▼
             Feature Extractor
                    │
      ┌─────────────┼──────────────┐
      │             │              │
      ▼             ▼              ▼
   TF-IDF       Embeddings        NLI
   Similarity   Similarity     Entailment /
                                Contradiction /
                                Neutral
      │             │              │
      └─────────────┼──────────────┘
                    │
             Answer-Quality
                Features
                    │
                    ▼
          Concept-Level Evidence
                    │
                    ▼
              36 Features
                    │
                    ▼
          Logistic Regression
                    │
                    ▼
       Overall Label + Confidence
                    │
                    ▼
        Concept Evidence Layer
                    │
                    ▼
        Explainability / Feedback
```

The application-facing interface is implemented through `GradingService`. It delegates overall classification to `HybridGrader` and reuses the concept predictions already produced during feature extraction.

The application-level concept evidence layer does **not** override the overall classifier label.

This distinction is intentional: concept coverage alone is not sufficient to establish that an answer is correct.

---

# Feature Groups

## 1. TF-IDF Similarity

Traditional lexical similarity features compare:

- Question ↔ Student answer
- Reference answer ↔ Student answer
- Question + Reference answer ↔ Student answer

These features provide lexical evidence but do not reliably capture semantic equivalence.

---

## 2. Sentence Embeddings

Sentence-transformer embeddings provide semantic similarity signals between the question, reference answer, and student answer.

Current model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

## 3. Natural Language Inference

NLI provides semantic relationship signals between the student answer and the question/reference answer.

Current model:

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

The 36-feature representation includes structural and linguistic characteristics such as:

- Answer word count
- Answer character count
- Sentence count
- Average words per sentence
- Lexical overlap with the reference
- Lexical overlap with the question
- Reference keyword coverage
- Negation frequency
- Uncertainty indicators
- Unique-token ratio
- Answer/reference length ratio
- Answer/question length ratio

---

## 5. Concept-Level Evidence

For each required technical concept, the system evaluates evidence relevant to that concept.

The research `ConceptMatcher` produces concept-level decisions:

```text
Concept
   │
   ├── Relevant evidence found?
   ├── Semantic similarity
   ├── Entailment
   └── Contradiction
        │
        ▼
   COVERED / MISSING / INCORRECT
```

Concept statistics are incorporated into the 36-feature Exp 008 representation.

The application also contains `ConceptEvidenceEvaluator`, which adds explicit terminology evidence and produces structured concept-level results for explanation. It does not modify the original Exp 008 `ConceptMatcher`.

---

# Application Layer

The reusable application layer currently contains:

```text
src/
├── grading/
│   ├── concept_evidence.py
│   ├── feature_extractor.py
│   ├── grading_service.py
│   └── hybrid_grader.py
│
└── models/
    ├── concept_matcher.py
    ├── embedding_baseline.py
    ├── nli_baseline.py
    └── tfidf_baseline.py
```

### `HybridGrader`

Wraps the Exp 008 36-feature representation and Logistic Regression classifier. It returns:

- Overall label
- Confidence
- Class probabilities
- Concept predictions generated during feature extraction

### `ConceptEvidenceEvaluator`

Provides application-level concept evidence by combining explicit terminology matching with the existing concept predictions.

### `GradingService`

Provides the public application-facing interface for evaluating a single student answer.

Its current flow is:

```text
Student Answer
      │
      ▼
HybridGrader
      │
      ├── Label
      ├── Confidence
      ├── Probabilities
      └── Concept predictions
              │
              ▼
     ConceptEvidenceEvaluator
              │
              ▼
       AnswerEvaluation
```

---

# Experiment 011 — Grading Policy Analysis

Experiment 011 tested whether concept coverage should be allowed to override the overall Exp 008 classification.

Three policies were compared:

- **Policy A:** raw Exp 008 prediction.
- **Policy B:** override an `Incorrect` prediction when concept coverage is sufficiently high and no contradiction is detected.
- **Policy C:** the same concept-coverage override with an additional contradiction threshold.

On the current OS evaluation:

| Policy | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---|---:|---:|---:|
| **Policy A — Exp 008 Baseline** | **61.11%** | **57.81%** | **62.06%** |
| Policy B — Concept Coverage | 61.11% | 51.43% | 56.19% |
| Policy C — Coverage + Contradiction | 61.11% | 51.43% | 56.19% |

The override policies changed three predictions. One change corrected a deadlock answer, but two changes incorrectly converted answers about the Banker's Algorithm / symbolic links from `Incorrect` to `Correct`.

This demonstrated an important research finding:

> **Presence of all required concepts does not guarantee overall answer correctness.**

Therefore, **Policy A remains authoritative** and concept evidence is retained for explanation rather than label override.

---

# Experiment 012 — Logistic Regression Convergence Diagnostic

Exp 012 investigated the convergence warning produced by the Exp 008 Logistic Regression configuration.

The same OS split and feature representation were evaluated using different iteration limits:

| `max_iter` | Converged | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---:|:---:|---:|---:|---:|
| **3000** | No | **61.11%** | **57.81%** | **62.06%** |
| 5000 | Yes | 55.56% | 51.10% | 56.47% |
| 10000 | Yes | 55.56% | 51.10% | 56.47% |

The 3000-iteration model produced a convergence warning. Increasing the limit to 5000 removed the warning, and 10000 produced the same converged solution as 5000.

Only **1 of 18 test predictions** changed between `max_iter=3000` and the converged models. That single prediction changed from `Partially Correct` to `Incorrect` for a Banker's Algorithm answer whose ground-truth label was `Partially Correct`.

Conclusion:

- The convergence warning is not merely cosmetic.
- Removing it changes model behavior on the fixed test split.
- The converged solution performs worse on this particular split.
- `max_iter=3000` remains the frozen Exp 008 research configuration for reproducibility.
- Higher iteration limits are not adopted as an improvement.

Any future change to solver, scaling, or convergence configuration should be treated as a separate engineering/research experiment rather than silently changing Exp 008.

---

# Environment and Reproducibility

A development-environment mismatch was identified during the convergence investigation.

Earlier work was accidentally executed using the global Python installation, while the repository's `.venv` uses different dependency versions.

### Global environment observed

```text
Python:               3.11.0
scikit-learn:         1.7.2
sentence-transformers: 5.4.1
transformers:         5.5.4
torch:                 2.7.1+cu118
numpy:                2.1.2
```

### Project `.venv`

```text
Python:               3.11.0
scikit-learn:         1.9.0
sentence-transformers: 6.0.0
transformers:         5.15.1
torch:                 2.13.0+cpu
numpy:                2.4.6
```

The authoritative project environment going forward is the repository `.venv`.

A clean `.venv` evaluation of the current Exp 008 pipeline produced:

```text
Validation Accuracy:  50.00%
Validation Macro F1:  40.00%
Validation Weighted F1: 50.00%

Test Accuracy:        55.56%
Test Macro F1:        51.10%
Test Weighted F1:     56.47%
```

These results match the converged `max_iter=5000/10000` behavior observed in Exp 012.

The previously recorded higher robustness numbers from earlier experiments are retained as historical experiment records but **should not yet be treated as the authoritative final benchmark**, because the environment mismatch must be resolved by rerunning the relevant experiments under the project `.venv`.

From this point onward, project experiments should be run only after activating:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# Historical Experiment 009 — Feature Ablation

The earlier ablation study showed that no individual feature family was consistently strongest across both domains.

Previously recorded results included:

### OS

```text
Answer-quality-only:
Accuracy: 72.22%
Macro F1: 69.86%
```

### DBMS

```text
Exp 007A Full Hybrid:
Accuracy: 73.68%
Macro F1: 73.33%
```

These values are retained as historical experiment outputs. They require authoritative reruns under the project `.venv` before being used as final benchmark claims.

---

# Historical Experiment 010 — Robustness Study

Experiment 010 evaluated leading configurations across five repeated question-level holdout splits.

Previously recorded results were:

### Revised OS

| Configuration | Mean Macro F1 | Macro F1 Std |
|---|---:|---:|
| Exp 008 Concept-Aware | 75.58% | 18.35% |
| Exp 007A Full Hybrid | 73.11% | 18.09% |
| TF-IDF + Embeddings + NLI | 65.97% | 6.77% |
| Answer-quality only | 64.19% | 11.96% |

### Revised DBMS

| Configuration | Mean Macro F1 | Macro F1 Std |
|---|---:|---:|
| Exp 008 Concept-Aware | 70.07% | 16.32% |
| Exp 007A Full Hybrid | 64.38% | 12.15% |
| TF-IDF + Embeddings + NLI | 53.14% | 3.04% |
| Answer-quality only | 52.61% | 16.01% |

### Historical Cross-Dataset Mean

| Configuration | OS | DBMS | Mean |
|---|---:|---:|---:|
| Exp 008 Concept-Aware | 75.58% | 70.07% | 72.83% |
| Exp 007A Full Hybrid | 73.11% | 64.38% | 68.75% |
| TF-IDF + Embeddings + NLI | 65.97% | 53.14% | 59.56% |
| Answer-quality only | 64.19% | 52.61% | 58.40% |

The earlier study suggested a performance-versus-stability tradeoff: the concept-aware model achieved higher mean Macro F1 but showed substantially higher split-to-split variance than the simpler TF-IDF + Embeddings + NLI configuration.

Because of the environment mismatch subsequently identified, these results are **historical and pending rerun under `.venv`**.

---

# Evaluation Methodology

The project uses question-level data separation to reduce question leakage.

For each split:

- Entire questions, rather than individual answer rows, are assigned to train/validation/test partitions.
- Question leakage is explicitly checked.
- TF-IDF is fitted using training data only.
- Feature extraction uses the frozen research components.
- Macro F1 is emphasized because the task contains multiple classes and minority-class performance matters.
- Accuracy and Weighted F1 are secondary metrics.

Current OS dataset size:

```text
84 rows
14 unique questions
```

Current DBMS dataset size:

```text
15 unique questions
```

The small number of unique questions makes individual train/test splits sensitive to which questions are held out.

---

# Dataset Labels and Concepts

The grading task currently uses five labels:

```text
Correct
Incorrect
Partially Correct
Irrelevant
Contradictory
```

Required concepts are question-specific technical concepts that should appear as evidence in a correct answer.

Concept-level annotations are used as evidence rather than as a deterministic replacement for the overall human-derived label.

This is important because an answer may mention the expected terminology while still giving an incorrect or contradictory explanation.

---

# Current Limitations

The current results should be considered research findings rather than production-level validation.

### Dataset Size

The current OS dataset contains only 14 unique questions, and the current DBMS dataset contains 15 unique questions. Performance is therefore sensitive to the particular questions included in each split.

### Limited Domains

Current evaluation is focused on Computer Science / IT technical subjects represented by the available OS and DBMS datasets.

### Human Agreement

A substantially larger human-graded dataset is required to determine how closely automated predictions agree with human evaluators.

### Statistical Confidence

Repeated holdout evaluation provides stronger evidence than a single split, but the current small question counts are insufficient to establish definitive statistical superiority.

### Convergence

The current Exp 008 Logistic Regression configuration reaches the configured iteration limit on the project data. This has been documented rather than silently changed.

### OCR / Handwriting

The current grading pipeline operates on extracted text. Handwritten student answers are not yet directly processed by the grading model.

---

# Future System Architecture

The eventual system is intended to separate answer ingestion from answer evaluation.

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

The SQL database and OCR/CV components are future system-level components. They are not part of the current NLP research model.

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
│   ├── exp_010_robustness/
│   ├── exp_011_grading_policy/
│   └── exp_012_convergence_diagnostic/
│
├── src/
│   ├── grading/
│   │   ├── concept_evidence.py
│   │   ├── feature_extractor.py
│   │   ├── grading_service.py
│   │   └── hybrid_grader.py
│   │
│   ├── models/
│   │   ├── concept_matcher.py
│   │   ├── embedding_baseline.py
│   │   ├── nli_baseline.py
│   │   └── tfidf_baseline.py
│   │
│   ├── evaluate.py
│   └── train.py
│
├── README.md
└── ...
```

Temporary validation scripts currently include:

```text
src/test_concept_evidence.py
src/test_grading_service.py
```

Each experiment is kept separately so that model development and experimental conclusions remain traceable.

---

# Running the Current Evaluation

Activate the project virtual environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run the current OS evaluation:

```powershell
python -m src.evaluate `
    "C:\Users\Ayuj Shingte\Downloads\revised_sample_os.csv"
```

The project datasets are currently maintained outside the repository.

---

# Next Development Stage

The immediate development direction is to finish the reusable application layer before building the database-backed system.

Planned sequence:

1. Establish `.venv` as the authoritative project environment.
2. Rerun the relevant experiments under `.venv` to establish authoritative benchmarks.
3. Implement application-level 0–10 score generation without changing Exp 008.
4. Add explainable feedback using the structured concept evidence and classifier outputs.
5. Run final OS + DBMS validation.
6. Freeze the model/application layer.
7. Build the SQL database and integration layer.
8. Add OCR / computer-vision answer ingestion as a separate upstream component.

The project is intentionally moving toward a stable application interface rather than continuously adding features to the research model without validation.

---

# Long-Term Objective

The long-term system is intended to convert a student's technical answer into a structured evaluation:

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

The final system is intended to support structured storage, answer-level evaluation, concept-level evidence, numerical scoring, and explainable feedback while keeping the research model independently reproducible.

---

## Status

**Research prototype — active development**

Current research candidate:

> **Experiment 008 — Concept-Aware Hybrid Grader**

Current authoritative OS result under the project `.venv`:

> **55.56% test accuracy, 51.10% test Macro F1, 56.47% test Weighted F1**

Current application layer:

> **HybridGrader → ConceptEvidenceEvaluator → GradingService**

The system is not yet considered production-ready. Larger datasets, human-agreement evaluation, authoritative reruns across OS and DBMS, and further validation are required before deployment claims can be made.
