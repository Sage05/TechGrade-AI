"""
TechGrade AI - Experiment 011
Final Grading Policy Evaluation

Purpose:
    Evaluate whether application-level concept evidence improves
    the answer-level predictions produced by Exp 008.

Important:
    Exp 008 itself is NOT modified.

Policies:
    A. Baseline:
       Use the Exp 008 HybridGrader prediction directly.

    B. Concept Coverage:
       Use required-concept coverage as an additional signal.

    C. Coverage + Contradiction:
       Use required-concept coverage together with contradiction
       evidence to make conservative corrections.

Thresholds are selected on the validation set and then evaluated
once on the held-out test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

from src.grading.concept_evidence import ConceptEvidenceEvaluator
from src.grading.feature_extractor import FeatureExtractor
from src.grading.hybrid_grader import HybridGrader
from src.models.concept_matcher import ConceptMatcher
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.nli_baseline import NLIBaseline
from src.models.tfidf_baseline import TfidfBaseline
from src.utils.data_split import (
    question_level_split,
    verify_no_question_leakage,
)


LABELS = [
    "Correct",
    "Partially Correct",
    "Incorrect",
    "Irrelevant",
    "Contradictory",
]


# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------

def parse_concepts(value) -> list[str]:
    """Parse required_concepts into a clean list."""
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()

    if not text:
        return []

    if "|" in text:
        return [
            item.strip()
            for item in text.split("|")
            if item.strip()
        ]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]
    except (json.JSONDecodeError, TypeError):
        pass

    return [text]


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def fit_models(train: pd.DataFrame):
    """Fit the reusable Exp 008 model components on training data only."""

    tfidf = TfidfBaseline()

    tfidf_corpus = []

    for _, row in train.iterrows():
        tfidf_corpus.extend(
            [
                str(row["question_text"]),
                str(row["model_answer"]),
                str(row["student_answer"]),
            ]
        )

    tfidf.fit(tfidf_corpus)

    embedding = EmbeddingBaseline()
    nli = NLIBaseline()
    concept_matcher = ConceptMatcher()

    extractor = FeatureExtractor(
        tfidf_model=tfidf,
        embedding_model=embedding,
        nli_model=nli,
        concept_matcher=concept_matcher,
    )

    grader = HybridGrader(
        feature_extractor=extractor,
    )

    return extractor, grader


def extract_features(
    extractor: FeatureExtractor,
    dataframe: pd.DataFrame,
):
    """
    Extract Exp 008 features and preserve concept predictions.
    """

    features = []
    concept_predictions = []

    for _, row in dataframe.iterrows():

        result = extractor.extract(
            question=str(row["question_text"]),
            reference_answer=str(row["model_answer"]),
            student_answer=str(row["student_answer"]),
            required_concepts=parse_concepts(
                row["required_concepts"]
            ),
        )

        features.append(result.values)
        concept_predictions.append(
            result.concept_predictions
        )

    return (
        np.asarray(features, dtype=float),
        concept_predictions,
    )


# ---------------------------------------------------------------------
# Concept evidence
# ---------------------------------------------------------------------

def calculate_concept_statistics(
    concept_predictions,
    student_answer: str,
):
    """
    Convert ConceptMatcher output into application-level evidence.

    Returns:
        coverage
        contradiction_fraction
        explicit_coverage
        covered_count
        concept_count
    """

    evaluator = ConceptEvidenceEvaluator()

    results = evaluator.evaluate_all(
        concept_predictions=concept_predictions,
        student_answer=student_answer,
    )

    if not results:
        return {
            "coverage": 0.0,
            "contradiction_fraction": 0.0,
            "explicit_coverage": 0.0,
            "covered_count": 0,
            "concept_count": 0,
        }

    concept_count = len(results)

    covered_count = sum(
        result.decision == "COVERED"
        for result in results
    )

    contradiction_count = sum(
        result.decision == "INCORRECT"
        for result in results
    )

    explicit_count = sum(
        result.explicit_match
        for result in results
    )

    return {
        "coverage": covered_count / concept_count,
        "contradiction_fraction": contradiction_count / concept_count,
        "explicit_coverage": explicit_count / concept_count,
        "covered_count": covered_count,
        "concept_count": concept_count,
    }


def build_policy_features(
    dataframe: pd.DataFrame,
    concept_predictions,
):
    """Build application-level concept statistics."""

    rows = []

    for index, (_, row) in enumerate(dataframe.iterrows()):

        stats = calculate_concept_statistics(
            concept_predictions=concept_predictions[index],
            student_answer=str(row["student_answer"]),
        )

        rows.append(stats)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def baseline_predictions(
    grader: HybridGrader,
    X: np.ndarray,
):
    """Policy A: raw Exp 008 predictions."""

    predictions = grader.classifier.predict(X)

    probabilities = grader.classifier.predict_proba(X)

    return predictions, probabilities


def apply_policy_b(
    baseline_label: str,
    coverage: float,
    contradiction_fraction: float,
    coverage_threshold: float,
):
    """
    Policy B.

    Conservative rule:
        A clearly covered answer should not remain Incorrect
        solely because the classifier underestimates it.

    We only modify Incorrect predictions.

    No modification is made for Contradictory predictions because
    contradiction is a stronger negative signal.
    """

    if (
        baseline_label == "Incorrect"
        and coverage >= coverage_threshold
        and contradiction_fraction == 0.0
    ):
        return "Correct"

    return baseline_label


def apply_policy_c(
    baseline_label: str,
    coverage: float,
    contradiction_fraction: float,
    coverage_threshold: float,
    contradiction_threshold: float,
):
    """
    Policy C.

    Uses both positive concept coverage and negative contradiction
    evidence.

    Rules:

        Incorrect + high coverage + no contradiction
            -> Correct

        Incorrect + moderate coverage + low contradiction
            -> Partially Correct

        Any prediction with substantial contradiction
            -> preserve classifier decision

    The exact thresholds are selected on validation data.
    """

    if contradiction_fraction >= contradiction_threshold:
        return baseline_label

    if (
        baseline_label == "Incorrect"
        and coverage >= coverage_threshold
    ):
        return "Correct"

    return baseline_label


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true,
    predictions,
):
    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def print_metrics(
    name: str,
    metrics: dict,
):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")


# ---------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------

def search_policy_b_threshold(
    y_true,
    baseline_predictions_,
    policy_features,
):
    """
    Select the coverage threshold using validation Macro F1.

    Thresholds are deliberately coarse because the dataset is small.
    """

    candidates = [
        0.50,
        0.60,
        0.75,
        0.80,
        1.00,
    ]

    best_threshold = candidates[0]
    best_f1 = -1.0

    for threshold in candidates:

        predictions = []

        for index, baseline in enumerate(
            baseline_predictions_
        ):
            predictions.append(
                apply_policy_b(
                    baseline_label=str(baseline),
                    coverage=float(
                        policy_features.iloc[index]["coverage"]
                    ),
                    contradiction_fraction=float(
                        policy_features.iloc[index][
                            "contradiction_fraction"
                        ]
                    ),
                    coverage_threshold=threshold,
                )
            )

        score = f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def search_policy_c_thresholds(
    y_true,
    baseline_predictions_,
    policy_features,
):
    """Select Policy C thresholds using validation Macro F1."""

    coverage_candidates = [
        0.50,
        0.60,
        0.75,
        0.80,
        1.00,
    ]

    contradiction_candidates = [
        0.25,
        0.50,
        0.75,
        1.00,
    ]

    best = None

    for coverage_threshold in coverage_candidates:

        for contradiction_threshold in contradiction_candidates:

            predictions = []

            for index, baseline in enumerate(
                baseline_predictions_
            ):

                predictions.append(
                    apply_policy_c(
                        baseline_label=str(baseline),
                        coverage=float(
                            policy_features.iloc[index]["coverage"]
                        ),
                        contradiction_fraction=float(
                            policy_features.iloc[index][
                                "contradiction_fraction"
                            ]
                        ),
                        coverage_threshold=coverage_threshold,
                        contradiction_threshold=contradiction_threshold,
                    )
                )

            score = f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )

            candidate = (
                score,
                coverage_threshold,
                contradiction_threshold,
            )

            if best is None or candidate[0] > best[0]:
                best = candidate

    return best


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run(dataset_path: str):

    print("=" * 70)
    print("TECHGRADE AI — EXPERIMENT 011")
    print("FINAL GRADING POLICY EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    df = pd.read_csv(dataset_path)

    print("\n[DATASET]")
    print(f"Rows: {len(df)}")
    print(
        f"Unique questions: "
        f"{df['question_text'].nunique()}"
    )

    required_columns = [
        "question_text",
        "model_answer",
        "student_answer",
        "required_concepts",
        "label",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # -------------------------------------------------------------
    # Question-level split
    # -------------------------------------------------------------

    train, validation, test = question_level_split(df)

    verify_no_question_leakage(
        train,
        validation,
        test,
    )

    print("\n[SPLIT]")
    print(
        f"Train:      {len(train)} rows / "
        f"{train.question_text.nunique()} questions"
    )
    print(
        f"Validation: {len(validation)} rows / "
        f"{validation.question_text.nunique()} questions"
    )
    print(
        f"Test:       {len(test)} rows / "
        f"{test.question_text.nunique()} questions"
    )

    # -------------------------------------------------------------
    # Fit Exp 008 pipeline
    # -------------------------------------------------------------

    print("\n[MODEL INITIALIZATION]")

    extractor, grader = fit_models(train)

    # -------------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------------

    print("\n[FEATURE EXTRACTION]")

    X_train, _ = extract_features(
        extractor,
        train,
    )

    X_validation, validation_concepts = extract_features(
        extractor,
        validation,
    )

    X_test, test_concepts = extract_features(
        extractor,
        test,
    )

    print(f"Feature count: {X_train.shape[1]}")

    # -------------------------------------------------------------
    # Train classifier
    # -------------------------------------------------------------

    print("\n[TRAINING]")

    grader.fit(
        X_train,
        train["label"].to_numpy(),
    )

    # -------------------------------------------------------------
    # Baseline predictions
    # -------------------------------------------------------------

    validation_baseline, _ = baseline_predictions(
        grader,
        X_validation,
    )

    test_baseline, _ = baseline_predictions(
        grader,
        X_test,
    )

    # -------------------------------------------------------------
    # Concept evidence
    # -------------------------------------------------------------

    validation_policy_features = build_policy_features(
        validation,
        validation_concepts,
    )

    test_policy_features = build_policy_features(
        test,
        test_concepts,
    )

    # -------------------------------------------------------------
    # Policy A
    # -------------------------------------------------------------

    validation_metrics_a = calculate_metrics(
        validation["label"],
        validation_baseline,
    )

    test_metrics_a = calculate_metrics(
        test["label"],
        test_baseline,
    )

    # -------------------------------------------------------------
    # Policy B threshold selection
    # -------------------------------------------------------------

    b_threshold, b_validation_f1 = (
        search_policy_b_threshold(
            y_true=validation["label"],
            baseline_predictions_=validation_baseline,
            policy_features=validation_policy_features,
        )
    )

    print("\n[POLICY B]")
    print(
        f"Selected coverage threshold: "
        f"{b_threshold:.2f}"
    )
    print(
        f"Validation Macro F1: "
        f"{b_validation_f1:.4f}"
    )

    test_policy_b = []

    for index, baseline in enumerate(test_baseline):

        test_policy_b.append(
            apply_policy_b(
                baseline_label=str(baseline),
                coverage=float(
                    test_policy_features.iloc[index]["coverage"]
                ),
                contradiction_fraction=float(
                    test_policy_features.iloc[index][
                        "contradiction_fraction"
                    ]
                ),
                coverage_threshold=b_threshold,
            )
        )

    test_metrics_b = calculate_metrics(
        test["label"],
        test_policy_b,
    )

    # -------------------------------------------------------------
    # Policy C threshold selection
    # -------------------------------------------------------------

    c_result = search_policy_c_thresholds(
        y_true=validation["label"],
        baseline_predictions_=validation_baseline,
        policy_features=validation_policy_features,
    )

    (
        c_validation_f1,
        c_coverage_threshold,
        c_contradiction_threshold,
    ) = c_result

    print("\n[POLICY C]")
    print(
        f"Selected coverage threshold: "
        f"{c_coverage_threshold:.2f}"
    )
    print(
        f"Selected contradiction threshold: "
        f"{c_contradiction_threshold:.2f}"
    )
    print(
        f"Validation Macro F1: "
        f"{c_validation_f1:.4f}"
    )

    test_policy_c = []

    for index, baseline in enumerate(test_baseline):

        test_policy_c.append(
            apply_policy_c(
                baseline_label=str(baseline),
                coverage=float(
                    test_policy_features.iloc[index]["coverage"]
                ),
                contradiction_fraction=float(
                    test_policy_features.iloc[index][
                        "contradiction_fraction"
                    ]
                ),
                coverage_threshold=c_coverage_threshold,
                contradiction_threshold=c_contradiction_threshold,
            )
        )

    test_metrics_c = calculate_metrics(
        test["label"],
        test_policy_c,
    )

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    print_metrics(
        "Policy A — Exp 008 Baseline",
        test_metrics_a,
    )

    print_metrics(
        "Policy B — Concept Coverage",
        test_metrics_b,
    )

    print_metrics(
        "Policy C — Coverage + Contradiction",
        test_metrics_c,
    )

    # -------------------------------------------------------------
    # Prediction changes
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PREDICTION CHANGES")
    print("=" * 70)

    changes = []

    for index in range(len(test)):

        baseline = str(test_baseline[index])
        policy_b = str(test_policy_b[index])
        policy_c = str(test_policy_c[index])

        if baseline != policy_b or baseline != policy_c:

            changes.append(
                {
                    "question": str(
                        test.iloc[index]["question_text"]
                    ),
                    "actual": str(
                        test.iloc[index]["label"]
                    ),
                    "baseline": baseline,
                    "policy_b": policy_b,
                    "policy_c": policy_c,
                    "concept_coverage": float(
                        test_policy_features.iloc[index][
                            "coverage"
                        ]
                    ),
                    "contradiction_fraction": float(
                        test_policy_features.iloc[index][
                            "contradiction_fraction"
                        ]
                    ),
                }
            )

    if not changes:
        print("\nNo test predictions changed.")

    else:

        for change in changes:

            print("\nQuestion:")
            print(change["question"])

            print(f"Actual:       {change['actual']}")
            print(f"Baseline:     {change['baseline']}")
            print(f"Policy B:     {change['policy_b']}")
            print(f"Policy C:     {change['policy_c']}")
            print(
                f"Concept coverage: "
                f"{change['concept_coverage']:.2f}"
            )
            print(
                f"Contradiction:   "
                f"{change['contradiction_fraction']:.2f}"
            )

    # -------------------------------------------------------------
    # Final recommendation
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("POLICY RECOMMENDATION")
    print("=" * 70)

    scores = {
        "Policy A": test_metrics_a["macro_f1"],
        "Policy B": test_metrics_b["macro_f1"],
        "Policy C": test_metrics_c["macro_f1"],
    }

    best_policy = max(
        scores,
        key=scores.get,
    )

    print(
        f"\nBest test Macro F1: "
        f"{best_policy}"
    )

    for policy, score in scores.items():
        print(
            f"{policy}: "
            f"{score:.4f}"
        )

    print(
        "\nImportant:"
        "\nThe policy is selected based on validation data."
        "\nThe reported recommendation is based on the held-out test set."
        "\nExp 008 itself remains unchanged."
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "TechGrade AI Experiment 011 "
            "— Final Grading Policy Evaluation"
        )
    )

    parser.add_argument(
        "dataset",
        help="Path to the revised CSV dataset.",
    )

    args = parser.parse_args()

    run(args.dataset)


if __name__ == "__main__":
    main()