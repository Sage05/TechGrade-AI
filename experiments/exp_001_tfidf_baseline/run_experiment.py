"""
Experiment 001 — TF-IDF Classification Baseline.

Runs a question-level split and evaluates a TF-IDF similarity
feature classifier on the validation and test sets.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import classification_metrics
from src.models.tfidf_classifier import TfidfClassifier
from src.utils.data_split import (
    question_level_split,
    verify_no_question_leakage,
)


DATASET_PATH = (
    r"C:\Users\Ayuj Shingte\Downloads\os_dataset (1).csv"
)


def main() -> None:

    print("=" * 70)
    print("TECHGRADE AI — EXPERIMENT 001")
    print("TF-IDF CLASSIFICATION BASELINE")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    print("\n[DATASET]")
    print(f"Rows: {len(df)}")
    print(
        f"Unique questions: "
        f"{df['question_text'].nunique()}"
    )

    # --------------------------------------------------------------
    # Question-level split
    # --------------------------------------------------------------

    train_df, val_df, test_df = question_level_split(
        df,
        test_size=0.15,
        validation_size=0.15,
        random_state=42,
    )

    print("\n[SPLIT]")

    print(
        f"Train: {len(train_df)} rows / "
        f"{train_df['question_text'].nunique()} questions"
    )

    print(
        f"Validation: {len(val_df)} rows / "
        f"{val_df['question_text'].nunique()} questions"
    )

    print(
        f"Test: {len(test_df)} rows / "
        f"{test_df['question_text'].nunique()} questions"
    )

    leakage_free = verify_no_question_leakage(
        train_df,
        val_df,
        test_df,
    )

    print(f"Question leakage: {not leakage_free}")

    if not leakage_free:
        raise RuntimeError(
            "Question leakage detected."
        )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = TfidfClassifier(
        random_state=42
    )

    print("\n[TRAINING]")

    model.fit(train_df)

    print("TF-IDF + Logistic Regression trained.")

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    print("\n[VALIDATION]")

    _, val_predictions = model.predict(
        val_df
    )

    val_metrics = classification_metrics(
        val_df["label"],
        val_predictions,
    )

    print(
        f"Accuracy: "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{val_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{val_metrics['weighted_f1']:.4f}"
    )

    print(
        "\nClassification Report:\n"
        f"{val_metrics['classification_report']}"
    )

    # --------------------------------------------------------------
    # Test
    # --------------------------------------------------------------

    print("\n[TEST]")

    test_features, test_predictions = (
        model.predict(test_df)
    )

    test_metrics = classification_metrics(
        test_df["label"],
        test_predictions,
    )

    print(
        f"Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    print(
        "\nClassification Report:\n"
        f"{test_metrics['classification_report']}"
    )

    # --------------------------------------------------------------
    # Inspect predictions
    # --------------------------------------------------------------

    print("\n[SAMPLE PREDICTIONS]")

    results = test_df[
        [
            "question_text",
            "student_answer",
            "label",
        ]
    ].copy()

    results["predicted_label"] = (
        test_predictions.values
    )

    results["question_student"] = (
        test_features["question_student"].values
    )

    results["reference_student"] = (
        test_features["reference_student"].values
    )

    results["question_reference_student"] = (
        test_features[
            "question_reference_student"
        ].values
    )

    print(
        results.head(10).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()