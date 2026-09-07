"""
TechGrade AI - Initial reusable training pipeline.

This is the first integration test for the reusable src/ pipeline.
It trains the current Exp 008 architecture on the OS dataset.
"""

from __future__ import annotations

import pandas as pd

from src.models.tfidf_baseline import TfidfBaseline
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.nli_baseline import NLIBaseline
from src.models.concept_matcher import ConceptMatcher

from src.grading.feature_extractor import FeatureExtractor
from src.grading.hybrid_grader import HybridGrader


DATASET = r"C:\Users\Ayuj Shingte\Downloads\revised_sample_os.csv"


def main() -> None:

    print("[DATASET]")

    df = pd.read_csv(DATASET)

    print(f"Rows: {len(df)}")
    print(
        f"Questions: "
        f"{df['question_text'].nunique()}"
    )

    # -------------------------------------------------
    # Models
    # -------------------------------------------------

    print("\n[MODELS]")

    tfidf_model = TfidfBaseline()

    train_corpus = (
        df["question_text"].astype(str).tolist()
        + df["model_answer"].astype(str).tolist()
        + df["student_answer"].astype(str).tolist()
    )

    tfidf_model.fit(train_corpus)

    print("TF-IDF ready.")

    embedding_model = EmbeddingBaseline()
    print("Embeddings ready.")

    nli_model = NLIBaseline()
    print("NLI ready.")

    concept_matcher = ConceptMatcher()
    print("Concept matcher ready.")

    # -------------------------------------------------
    # Feature extraction
    # -------------------------------------------------

    print("\n[FEATURE EXTRACTION]")

    extractor = FeatureExtractor(
        tfidf_model=tfidf_model,
        embedding_model=embedding_model,
        nli_model=nli_model,
        concept_matcher=concept_matcher,
    )

    first_row = df.iloc[0]

    features = extractor.extract(
        question=first_row["question_text"],
        reference_answer=first_row["model_answer"],
        student_answer=first_row["student_answer"],
        required_concepts=first_row["required_concepts"],
    )

    print(
        f"Feature count: {len(features.values)}"
    )

    print(
        f"Feature vector shape: "
        f"{features.values.shape}"
    )

    print("\nFeature names:")

    for index, name in enumerate(
        features.names,
        start=1,
    ):
        print(
            f"{index:02d}. {name}"
        )

    # -------------------------------------------------
    # Build training matrix
    # -------------------------------------------------

    print("\n[BUILDING TRAINING MATRIX]")

    feature_rows = []

    for _, row in df.iterrows():

        row_features = extractor.extract(
            question=row["question_text"],
            reference_answer=row["model_answer"],
            student_answer=row["student_answer"],
            required_concepts=row["required_concepts"],
        )

        feature_rows.append(
            row_features.values
        )

    import numpy as np

    X = np.vstack(feature_rows)

    y = df["label"].astype(str).to_numpy()

    print(
        f"Training matrix: {X.shape}"
    )

    print(
        f"Labels: {len(y)}"
    )

    # -------------------------------------------------
    # Train grader
    # -------------------------------------------------

    print("\n[TRAINING]")

    grader = HybridGrader(
        feature_extractor=extractor,
    )

    grader.fit(
        X=X,
        labels=y,
    )

    print("Hybrid grader trained.")

    # -------------------------------------------------
    # Test inference
    # -------------------------------------------------

    print("\n[INFERENCE]")

    result = grader.predict(
        question=first_row["question_text"],
        reference_answer=first_row["model_answer"],
        student_answer=first_row["student_answer"],
        required_concepts=first_row["required_concepts"],
    )

    print(
        f"Prediction: {result.label}"
    )

    print(
        f"Confidence: "
        f"{result.confidence:.4f}"
    )

    print(
        "Probabilities:"
    )

    for label, probability in (
        result.probabilities.items()
    ):
        print(
            f"  {label}: "
            f"{probability:.4f}"
        )


if __name__ == "__main__":
    main()