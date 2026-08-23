"""
TechGrade AI — Experiment 005.2
Reference-Grounded Concept Matching
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.models.concept_coverage import (
    parse_concepts,
)

from src.models.concept_matcher import (
    ConceptMatcher,
)


LABELS = [
    "COVERED",
    "MISSING",
    "INCORRECT",
]


def build_ground_truth(
    required: list[str],
    missing: list[str],
    incorrect: list[str],
) -> dict[str, str]:

    missing_set = set(missing)
    incorrect_set = set(incorrect)

    result = {}

    for concept in required:

        if concept in incorrect_set:

            result[concept] = "INCORRECT"

        elif concept in missing_set:

            result[concept] = "MISSING"

        else:

            result[concept] = "COVERED"

    return result


def evaluate(
    predictions: list[str],
    ground_truth: list[str],
) -> None:

    print("\n" + "=" * 70)
    print("CONCEPT-LEVEL EVALUATION")
    print("=" * 70)

    print(
        f"\nConcept instances: "
        f"{len(predictions)}"
    )

    print(
        f"Accuracy: "
        f"{accuracy_score(ground_truth, predictions):.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            ground_truth,
            predictions,
            labels=LABELS,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    matrix = confusion_matrix(
        ground_truth,
        predictions,
        labels=LABELS,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual COVERED",
            "Actual MISSING",
            "Actual INCORRECT",
        ],
        columns=[
            "Pred COVERED",
            "Pred MISSING",
            "Pred INCORRECT",
        ],
    )

    print(
        matrix_df.to_string()
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "TechGrade AI Experiment 005.2"
        )
    )

    parser.add_argument(
        "dataset",
        help="Path to revised CSV dataset.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TECHGRADE AI — EXPERIMENT 005.2")
    print("REFERENCE-GROUNDED CONCEPT MATCHING")
    print("=" * 70)

    df = pd.read_csv(
        args.dataset
    )

    print("\n[DATASET]")

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Unique questions: "
        f"{df['question_text'].nunique()}"
    )

    required_columns = [
        "question_text",
        "model_answer",
        "student_answer",
        "required_concepts",
        "missing_concepts",
        "incorrect_concepts",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    matcher = ConceptMatcher()

    all_predictions = []
    all_ground_truth = []

    samples = []

    print("\n[PROCESSING]")

    for row_index, row in df.iterrows():

        required = parse_concepts(
            row["required_concepts"]
        )

        missing = parse_concepts(
            row["missing_concepts"]
        )

        incorrect = parse_concepts(
            row["incorrect_concepts"]
        )

        if not required:
            continue

        ground_truth = (
            build_ground_truth(
                required,
                missing,
                incorrect,
            )
        )

        predictions = matcher.predict(
            required_concepts=required,
            model_answer=str(
                row["model_answer"]
            ),
            student_answer=str(
                row["student_answer"]
            ),
        )

        for prediction in predictions:

            all_predictions.append(
                prediction.decision
            )

            all_ground_truth.append(
                ground_truth[
                    prediction.concept
                ]
            )

            if len(samples) < 12:

                samples.append(
                    {
                        "row": row_index,
                        "concept": (
                            prediction.concept
                        ),
                        "actual": (
                            ground_truth[
                                prediction.concept
                            ]
                        ),
                        "predicted": (
                            prediction.decision
                        ),
                        "similarity": (
                            prediction.semantic_similarity
                        ),
                        "entailment": (
                            prediction.entailment
                        ),
                        "contradiction": (
                            prediction.contradiction
                        ),
                        "neutral": (
                            prediction.neutral
                        ),
                        "reference_evidence": (
                            prediction.reference_evidence
                        ),
                        "student_evidence": (
                            prediction.student_evidence
                        ),
                    }
                )

    evaluate(
        all_predictions,
        all_ground_truth,
    )

    print("\n" + "=" * 70)
    print("SAMPLE CONCEPT PREDICTIONS")
    print("=" * 70)

    if samples:

        sample_df = pd.DataFrame(
            samples
        )

        print(
            sample_df.to_string(
                index=False,
                max_colwidth=60,
                float_format=lambda x: (
                    f"{x:.4f}"
                ),
            )
        )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()