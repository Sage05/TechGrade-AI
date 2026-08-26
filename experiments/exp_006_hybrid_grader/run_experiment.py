"""
TechGrade AI — Experiment 006
Hybrid Grader

Combines independent grading signals:

    1. TF-IDF similarity
    2. Semantic embedding similarity
    3. NLI features
    4. Concept coverage
    5. Basic answer statistics

The target label is:

    Correct
    Incorrect
    Partially Correct
    Irrelevant
    Contradictory

Important:
Human annotation fields such as missing_concepts and
incorrect_concepts are NOT used as model features.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from src.models.embedding_baseline import (
    EmbeddingBaseline,
)

from src.models.nli_baseline import (
    NLIBaseline,
)

from src.models.tfidf_baseline import (
    TfidfBaseline,
)


from src.utils.data_split import (
    question_level_split,
    verify_no_question_leakage,
)


LABELS = [
    "Correct",
    "Incorrect",
    "Partially Correct",
    "Irrelevant",
    "Contradictory",
]


def safe_parse_concepts(value) -> list[str]:
    """
    Parse a concept field using the existing project parser.

    Returns an empty list for missing values.
    """

    if pd.isna(value):
        return []

    try:
        from src.models.concept_coverage import (
            parse_concepts,
        )

        return parse_concepts(value)

    except Exception:
        return []


def build_features(
    df: pd.DataFrame,
    tfidf_model: TfidfBaseline,
    embedding_model: EmbeddingBaseline,
    nli_model: NLIBaseline,
) -> np.ndarray:
    """
    Build the hybrid feature matrix.

    Features:

        0  question-student TF-IDF similarity
        1  reference-student TF-IDF similarity
        2  question-reference TF-IDF similarity

        3  question-student embedding similarity
        4  reference-student embedding similarity
        5  question-reference embedding similarity

        6  question-student NLI entailment
        7  question-student NLI contradiction
        8  question-student NLI neutral

        9  reference-student NLI entailment
        10 reference-student NLI contradiction
        11 reference-student NLI neutral

        12 concept coverage ratio
        13 concept count

        14 answer word count
        15 answer character count
    """

    features = []

    for _, row in df.iterrows():

        question = str(
            row["question_text"]
        )

        reference = str(
            row["model_answer"]
        )

        student = str(
            row["student_answer"]
        )

        # -------------------------------------------------
        # TF-IDF
        # -------------------------------------------------

        tfidf = tfidf_model.compare(
            question,
            reference,
            student,
        )

        # -------------------------------------------------
        # Semantic embeddings
        # -------------------------------------------------

        embedding = (
            embedding_model.compare(
                question,
                reference,
                student,
            )
        )

        # -------------------------------------------------
        # NLI
        # -------------------------------------------------

        nli = nli_model.compare(
            question,
            reference,
            student,
        )

        # -------------------------------------------------
        # Concept coverage
        # -------------------------------------------------

        

        # -------------------------------------------------
        # Basic answer statistics
        # -------------------------------------------------

        word_count = float(
            len(
                student.split()
            )
        )

        character_count = float(
            len(student)
        )

        row_features = [
            # TF-IDF
            float(
                tfidf.question_student
            ),
            float(
                tfidf.reference_student
            ),
            float(
                tfidf.question_reference_student
            ),

            # Embeddings
            float(
                embedding.question_student
            ),
            float(
                embedding.reference_student
            ),
            float(
                embedding.question_reference_student
            ),

            # NLI question/student
            float(
                nli.question_entailment
            ),
            float(
                nli.question_contradiction
            ),
            float(
                nli.question_neutral
            ),

            # NLI reference/student
            float(
                nli.reference_entailment
            ),
            float(
                nli.reference_contradiction
            ),
            float(
                nli.reference_neutral
            ),

            

            # Answer statistics
            word_count,
            character_count,
        ]

        features.append(
            row_features
        )

    return np.asarray(
        features,
        dtype=np.float32,
    )


def train_classifier(
    X_train: np.ndarray,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    Train the hybrid classifier.
    """

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate(
    model: LogisticRegression,
    X: np.ndarray,
    y: pd.Series,
    split_name: str,
) -> None:

    predictions = model.predict(
        X
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

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y,
            predictions,
            labels=LABELS,
            zero_division=0,
        )
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run TechGrade AI "
            "Experiment 006."
        )
    )

    parser.add_argument(
        "dataset",
        help="Path to the CSV dataset.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TECHGRADE AI — EXPERIMENT 006")
    print("HYBRID GRADER")
    print("=" * 70)

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Question-level split
    # -----------------------------------------------------

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

    no_leakage = verify_no_question_leakage(
       train,
       validation,
       test,
    )

    print(
        f"Question leakage: "
        f"{not no_leakage}"
    )

    if not no_leakage:
        raise RuntimeError(
            "Question leakage detected."
        )

    # -----------------------------------------------------
    # Feature models
    # -----------------------------------------------------

    print("\n[MODELS]")

    print(
        "Loading TF-IDF baseline..."
    )

    tfidf_model = TfidfBaseline()

    # Fit TF-IDF ONLY on training text.
    tfidf_corpus = (
        train["question_text"].tolist()
        + train["model_answer"].tolist()
        + train["student_answer"].tolist()
    )

    tfidf_model.fit(
        tfidf_corpus
    )

    print(
        "Loading embedding baseline..."
    )

    embedding_model = (
        EmbeddingBaseline()
    )

    print(
        "Loading NLI baseline..."
    )

    nli_model = NLIBaseline()

    

    

    # -----------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------

    print("\n[FEATURE EXTRACTION]")

    X_train = build_features(
       train,
       tfidf_model,
       embedding_model,
       nli_model,
    )

    X_validation = build_features(
       validation,
       tfidf_model,
       embedding_model,
       nli_model,
    )

    X_test = build_features(
       test,
       tfidf_model,
       embedding_model,
       nli_model,
    )

    print(
        f"Feature count: "
        f"{X_train.shape[1]}"
    )

    print(
        f"Training matrix: "
        f"{X_train.shape}"
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print("\n[TRAINING]")

    classifier = train_classifier(
        X_train,
        train["label"],
    )

    print(
        "Hybrid Logistic Regression trained."
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    evaluate(
        classifier,
        X_validation,
        validation["label"],
        "VALIDATION",
    )

    evaluate(
        classifier,
        X_test,
        test["label"],
        "TEST",
    )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    print(
        "\n[FEATURE IMPORTANCE]"
    )

    feature_names = [
        "tfidf_question_student",
        "tfidf_reference_student",
        "tfidf_question_reference",
        "embedding_question_student",
        "embedding_reference_student",
        "embedding_question_reference",
        "nli_question_entailment",
        "nli_question_contradiction",
        "nli_question_neutral",
        "nli_reference_entailment",
        "nli_reference_contradiction",
        "nli_reference_neutral",
        "answer_word_count",
        "answer_character_count",
    ]

    coefficients = np.mean(
        np.abs(
            classifier.coef_
        ),
        axis=0,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": coefficients,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        importance.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
