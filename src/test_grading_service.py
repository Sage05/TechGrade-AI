from pathlib import Path

import pandas as pd

import numpy as np

from src.grading.feature_extractor import FeatureExtractor
from src.grading.hybrid_grader import HybridGrader
from src.grading.grading_service import GradingService
from src.models.concept_matcher import ConceptMatcher
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.nli_baseline import NLIBaseline
from src.models.tfidf_baseline import TfidfBaseline
from src.utils.data_split import (
    question_level_split,
    verify_no_question_leakage,
)


DATASET = Path(r"C:\Users\Ayuj Shingte\Downloads\revised_sample_os.csv")


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv(DATASET)

    train_df, _, _ = question_level_split(df)

    print(
        f"Training rows: {len(train_df)} | "
        f"Training questions: {train_df['question_text'].nunique()}"
    )

    # ---------------------------------------------------------
    # Build the same components used by Exp 008
    # ---------------------------------------------------------

    tfidf = TfidfBaseline()

    train_corpus = (
        train_df["question_text"].tolist()
        + train_df["model_answer"].tolist()
        + train_df["student_answer"].tolist()
    )

    tfidf.fit(train_corpus)

    embedding = EmbeddingBaseline()
    nli = NLIBaseline()
    concept_matcher = ConceptMatcher(
        embedding_model=embedding.model_name,
        nli_model=nli.model_name,
    )

    feature_extractor = FeatureExtractor(
    tfidf_model=tfidf,
    embedding_model=embedding,
    nli_model=nli,
    concept_matcher=concept_matcher,
    )

    # ---------------------------------------------------------
    # Extract training features
    # ---------------------------------------------------------

    X_train = []
    y_train = []

    print("Extracting training features...")

    for _, row in train_df.iterrows():
        features = feature_extractor.extract(
            question=row["question_text"],
            reference_answer=row["model_answer"],
            student_answer=row["student_answer"],
            required_concepts=row["required_concepts"],
        )

        X_train.append(features.values)
        y_train.append(row["label"])

    # ---------------------------------------------------------
    # Train HybridGrader
    # ---------------------------------------------------------

    grader = HybridGrader(feature_extractor)
    grader.fit(np.asarray(X_train, dtype=float), y_train)

    # ---------------------------------------------------------
    # Create application service
    # ---------------------------------------------------------

    service = GradingService(
        grader=grader,
    )

    # ---------------------------------------------------------
    # Test one real answer
    # ---------------------------------------------------------

    row = df.iloc[0]

    print("\n" + "=" * 60)
    print("TECHGRADE AI — GRADING SERVICE TEST")
    print("=" * 60)

    print("\nQuestion:")
    print(row["question_text"])

    print("\nStudent Answer:")
    print(row["student_answer"])

    result = service.evaluate(
        question=row["question_text"],
        reference_answer=row["model_answer"],
        required_concepts=row["required_concepts"],
        student_answer=row["student_answer"],
    )

    print("\nOverall Result:")
    print(f"Label:      {result.label}")
    print(f"Confidence: {result.confidence:.4f}")

    print("\nProbabilities:")

    for label, probability in result.probabilities.items():
        print(f"  {label}: {probability:.4f}")

    print("\nConcept Evaluation:")

    if not result.concept_results:
        print("  No required concepts found.")

    else:
        for concept in result.concept_results:
             print(f"\n  Concept: {concept.concept}")
             print(f"  Decision:       {concept.decision}")
             print(f"  Explicit Match: {concept.explicit_match}")
             print(f"  Similarity:     {concept.semantic_similarity:.4f}")
             print(f"  Entailment:     {concept.entailment:.4f}")
             print(f"  Contradiction:  {concept.contradiction:.4f}")
    print("\n" + "=" * 60)
    print("GRADING SERVICE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()