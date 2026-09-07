"""
TechGrade AI - Reusable evaluation pipeline.

Evaluates the reusable grading pipeline using a
question-level train/validation/test split.

This is an integration/verification script.
It does not modify the research experiment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

from src.models.tfidf_baseline import TfidfBaseline
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.nli_baseline import NLIBaseline
from src.models.concept_matcher import ConceptMatcher

from src.grading.feature_extractor import FeatureExtractor
from src.grading.hybrid_grader import HybridGrader

from src.utils.data_split import (
    question_level_split,
    verify_no_question_leakage,
)


DATASET = r"C:\Users\Ayuj Shingte\Downloads\revised_sample_os.csv"


def build_feature_matrix(
    df: pd.DataFrame,
    extractor: FeatureExtractor,
) -> np.ndarray:
    """Extract the 36-feature vector for every row."""

    rows = []

    for _, row in df.iterrows():

        features = extractor.extract(
            question=row["question_text"],
            reference_answer=row["model_answer"],
            student_answer=row["student_answer"],
            required_concepts=row["required_concepts"],
        )

        rows.append(features.values)

    return np.vstack(rows)


def main() -> None:

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    print("[DATASET]")

    df = pd.read_csv(DATASET)

    print(f"Rows: {len(df)}")
    print(
        f"Questions: "
        f"{df['question_text'].nunique()}"
    )

    # -------------------------------------------------
    # Question-level split
    # -------------------------------------------------

    train, validation, test = (
        question_level_split(df)
    )

    print("\n[SPLIT]")

    print(
        f"Train: {len(train)} rows / "
        f"{train.question_text.nunique()} questions"
    )

    print(
        f"Validation: {len(validation)} rows / "
        f"{validation.question_text.nunique()} questions"
    )

    print(
        f"Test: {len(test)} rows / "
        f"{test.question_text.nunique()} questions"
    )

    if not verify_no_question_leakage(
        train,
        validation,
        test,
    ):
        raise RuntimeError(
            "Question leakage detected."
        )

    print("Question leakage: False")

    # -------------------------------------------------
    # Models
    # -------------------------------------------------

    print("\n[MODELS]")

    tfidf_model = TfidfBaseline()

    # IMPORTANT:
    # TF-IDF is fitted on TRAIN only.
    train_corpus = (
        train["question_text"].astype(str).tolist()
        + train["model_answer"].astype(str).tolist()
        + train["student_answer"].astype(str).tolist()
    )

    tfidf_model.fit(
        train_corpus
    )

    print("TF-IDF ready.")

    embedding_model = EmbeddingBaseline()
    print("Embeddings ready.")

    nli_model = NLIBaseline()
    print("NLI ready.")

    concept_matcher = ConceptMatcher()
    print("Concept matcher ready.")

    # -------------------------------------------------
    # Feature extractor
    # -------------------------------------------------

    extractor = FeatureExtractor(
        tfidf_model=tfidf_model,
        embedding_model=embedding_model,
        nli_model=nli_model,
        concept_matcher=concept_matcher,
    )

    # -------------------------------------------------
    # Feature extraction
    # -------------------------------------------------

    print("\n[FEATURE EXTRACTION]")

    X_train = build_feature_matrix(
        train,
        extractor,
    )

    X_validation = build_feature_matrix(
        validation,
        extractor,
    )

    X_test = build_feature_matrix(
        test,
        extractor,
    )

    print(
        f"Train features: {X_train.shape}"
    )

    print(
        f"Validation features: "
        f"{X_validation.shape}"
    )

    print(
        f"Test features: "
        f"{X_test.shape}"
    )

    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    print("\n[TRAINING]")

    grader = HybridGrader(
        feature_extractor=extractor,
    )

    grader.fit(
        X=X_train,
        labels=train["label"].astype(str).to_numpy(),
    )

    print(
        "Exp 008 Logistic Regression trained."
    )

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------

    print("\n[EVALUATION]")

    for split_name, X, y in [
        (
            "VALIDATION",
            X_validation,
            validation["label"].astype(str).to_numpy(),
        ),
        (
            "TEST",
            X_test,
            test["label"].astype(str).to_numpy(),
        ),
    ]:

        predictions = (
            grader.classifier.predict(X)
        )

        accuracy = accuracy_score(
            y,
            predictions,
        )

        macro_f1 = f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0,
        )

        print(
            f"\n[{split_name}]"
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Macro F1: {macro_f1:.4f}"
        )

        print(
            f"Weighted F1: {weighted_f1:.4f}"
        )


if __name__ == "__main__":
    main()