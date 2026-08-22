"""
Experiment 002 — Semantic Embedding Baseline.

Compares sentence-transformer semantic features against
the TF-IDF baseline using the same question-level split.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import (
    classification_metrics,
)
from src.models.embedding_classifier import (
    EmbeddingClassifier,
)
from src.utils.data_split import (
    question_level_split,
    verify_no_question_leakage,
)


DATASET_PATH = (
    r"C:\Users\Ayuj Shingte\Downloads\os_dataset (1).csv"
)


def main() -> None:

    print("=" * 70)
    print("TECHGRADE AI — EXPERIMENT 002")
    print("SEMANTIC EMBEDDING BASELINE")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load dataset
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

    train_df, val_df, test_df = (
        question_level_split(
            df,
            test_size=0.15,
            validation_size=0.15,
            random_state=42,
        )
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

    print(
        f"Question leakage: "
        f"{not leakage_free}"
    )

    if not leakage_free:
        raise RuntimeError(
            "Question leakage detected."
        )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    print("\n[MODEL]")
    print(
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )

    model = EmbeddingClassifier(
        random_state=42
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    print("\n[TRAINING]")

    model.fit(train_df)

    print(
        "Semantic embeddings + Logistic "
        "Regression trained."
    )

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
    # Sample predictions
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
        test_features[
            "question_student"
        ].values
    )

    results["reference_student"] = (
        test_features[
            "reference_student"
        ].values
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