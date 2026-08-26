"""
TechGrade AI — Experiment 008
Concept-Aware Hybrid Grader

Research question:

Can automatically inferred concept-level evidence improve
the hybrid grader without using human concept annotations?

Allowed grading-time information:

    question_text
    model_answer
    student_answer
    required_concepts

Forbidden predictive features:

    label
    missing_concepts
    incorrect_concepts
    human_score
"""

from __future__ import annotations

import argparse
import ast
import json
import re

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
from src.models.concept_matcher import (
    ConceptMatcher,
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


def parse_concepts(value) -> list[str]:
    """
    Parse required_concepts safely.

    Supports:

        ["Concept A", "Concept B"]

    and stringified Python lists / JSON lists.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    text = str(value).strip()

    if not text:
        return []

    # Try JSON first.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]

    except Exception:
        pass

    # Try Python literal syntax.
    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]

    except Exception:
        pass

    # Fallback for simple comma-separated values.
    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def build_features(
    df: pd.DataFrame,
    tfidf_model: TfidfBaseline,
    embedding_model: EmbeddingBaseline,
    nli_model: NLIBaseline,
    concept_matcher: ConceptMatcher,
) -> np.ndarray:
    """
    Build the Exp 007A feature set plus automatic concept
    evidence features.

    Base features:

        TF-IDF
        embeddings
        NLI
        answer statistics

    New concept features:

        mean concept similarity
        max concept similarity
        minimum concept similarity
        mean concept entailment
        max concept entailment
        mean concept contradiction
        max concept contradiction
        mean concept neutral
        fraction of concepts with strong entailment
        fraction of concepts with strong contradiction
        concept evidence spread
        number of required concepts

    Human concept annotations are never used.
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
        # Exp 006 / Exp 007A features
        # -------------------------------------------------

        tfidf = tfidf_model.compare(
            question,
            reference,
            student,
        )

        embedding = (
            embedding_model.compare(
                question,
                reference,
                student,
            )
        )

        nli = nli_model.compare(
            question,
            reference,
            student,
        )

        student_tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            student.lower(),
        )

        reference_tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            reference.lower(),
        )

        question_tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            question.lower(),
        )

        word_count = len(
            student_tokens
        )

        character_count = len(
            student
        )

        sentence_count = max(
            1,
            len(
                re.findall(
                    r"[.!?]+",
                    student,
                )
            ),
        )

        avg_words_per_sentence = (
            word_count
            / sentence_count
        )

        student_set = set(
            student_tokens
        )

        reference_set = set(
            reference_tokens
        )

        question_set = set(
            question_tokens
        )

        if student_set and reference_set:
            lexical_reference = (
                len(
                    student_set
                    & reference_set
                )
                / len(
                    student_set
                    | reference_set
                )
            )
        else:
            lexical_reference = 0.0

        if student_set and question_set:
            lexical_question = (
                len(
                    student_set
                    & question_set
                )
                / len(
                    student_set
                    | question_set
                )
            )
        else:
            lexical_question = 0.0

        reference_keywords = {
            token
            for token in reference_set
            if len(token) >= 4
        }

        if reference_keywords:
            reference_keyword_coverage = (
                len(
                    reference_keywords
                    & student_set
                )
                / len(reference_keywords)
            )
        else:
            reference_keyword_coverage = 0.0

        negation_words = {
            "not",
            "never",
            "no",
            "none",
            "cannot",
            "can't",
            "doesn't",
            "isn't",
            "aren't",
            "without",
        }

        negation_count = sum(
            token in negation_words
            for token in student_tokens
        )

        normalized_student = (
            student.lower()
        )

        uncertainty_phrases = [
            "maybe",
            "perhaps",
            "probably",
            "possibly",
            "might",
            "i think",
            "i believe",
            "i don't know",
            "not sure",
            "unsure",
        ]

        uncertainty_count = sum(
            phrase in normalized_student
            for phrase in uncertainty_phrases
        )

        unique_token_ratio = (
            len(set(student_tokens))
            / max(
                1,
                word_count,
            )
        )

        answer_reference_length_ratio = (
            word_count
            / max(
                1,
                len(reference_tokens),
            )
        )

        answer_question_length_ratio = (
            word_count
            / max(
                1,
                len(question_tokens),
            )
        )

        # -------------------------------------------------
        # Automatic concept evidence
        # -------------------------------------------------

        required_concepts = parse_concepts(
            row["required_concepts"]
        )

        concept_predictions = (
            concept_matcher.predict(
                required_concepts=required_concepts,
                model_answer=reference,
                student_answer=student,
            )
        )

        similarities = [
            float(
                prediction.semantic_similarity
            )
            for prediction
            in concept_predictions
        ]

        entailments = [
            float(
                prediction.entailment
            )
            for prediction
            in concept_predictions
        ]

        contradictions = [
            float(
                prediction.contradiction
            )
            for prediction
            in concept_predictions
        ]

        neutrals = [
            float(
                prediction.neutral
            )
            for prediction
            in concept_predictions
        ]

        if similarities:

            mean_similarity = float(
                np.mean(similarities)
            )

            max_similarity = float(
                np.max(similarities)
            )

            min_similarity = float(
                np.min(similarities)
            )

            mean_entailment = float(
                np.mean(entailments)
            )

            max_entailment = float(
                np.max(entailments)
            )

            mean_contradiction = float(
                np.mean(contradictions)
            )

            max_contradiction = float(
                np.max(contradictions)
            )

            mean_neutral = float(
                np.mean(neutrals)
            )

            strong_entailment_fraction = (
                np.mean(
                    np.asarray(entailments)
                    >= 0.60
                )
            )

            strong_contradiction_fraction = (
                np.mean(
                    np.asarray(contradictions)
                    >= 0.75
                )
            )

            concept_spread = float(
                np.std(similarities)
            )

        else:

            mean_similarity = 0.0
            max_similarity = 0.0
            min_similarity = 0.0
            mean_entailment = 0.0
            max_entailment = 0.0
            mean_contradiction = 0.0
            max_contradiction = 0.0
            mean_neutral = 1.0
            strong_entailment_fraction = 0.0
            strong_contradiction_fraction = 0.0
            concept_spread = 0.0

        concept_count = len(
            required_concepts
        )

        row_features = [

            # ---------------------------------------------
            # Exp 006 — TF-IDF
            # ---------------------------------------------

            float(
                tfidf.question_student
            ),
            float(
                tfidf.reference_student
            ),
            float(
                tfidf.question_reference_student
            ),

            # ---------------------------------------------
            # Exp 006 — Embeddings
            # ---------------------------------------------

            float(
                embedding.question_student
            ),
            float(
                embedding.reference_student
            ),
            float(
                embedding.question_reference_student
            ),

            # ---------------------------------------------
            # Exp 006 — NLI
            # ---------------------------------------------

            float(
                nli.question_entailment
            ),
            float(
                nli.question_contradiction
            ),
            float(
                nli.question_neutral
            ),
            float(
                nli.reference_entailment
            ),
            float(
                nli.reference_contradiction
            ),
            float(
                nli.reference_neutral
            ),

            # ---------------------------------------------
            # Exp 007A — Answer quality
            # ---------------------------------------------

            float(word_count),
            float(character_count),
            float(sentence_count),
            float(avg_words_per_sentence),
            float(lexical_reference),
            float(lexical_question),
            float(reference_keyword_coverage),
            float(negation_count),
            float(uncertainty_count),
            float(unique_token_ratio),
            float(answer_reference_length_ratio),
            float(answer_question_length_ratio),

            # ---------------------------------------------
            # Exp 008 — Concept evidence
            # ---------------------------------------------

            float(mean_similarity),
            float(max_similarity),
            float(min_similarity),
            float(mean_entailment),
            float(max_entailment),
            float(mean_contradiction),
            float(max_contradiction),
            float(mean_neutral),
            float(strong_entailment_fraction),
            float(strong_contradiction_fraction),
            float(concept_spread),
            float(concept_count),
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

    model = LogisticRegression(
        max_iter=3000,
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
            "TechGrade AI "
            "Experiment 008."
        )
    )

    parser.add_argument(
        "dataset",
        help="Path to the revised CSV dataset.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(
        "TECHGRADE AI — EXPERIMENT 008"
    )
    print(
        "CONCEPT-AWARE HYBRID GRADER"
    )
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

    # Verify required grading-time fields.
    required_columns = [
        "question_text",
        "model_answer",
        "student_answer",
        "required_concepts",
        "label",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # -----------------------------------------------------
    # Split
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

    no_leakage = (
        verify_no_question_leakage(
            train,
            validation,
            test,
        )
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
    # Models
    # -----------------------------------------------------

    print("\n[MODELS]")

    print(
        "Loading TF-IDF baseline..."
    )

    tfidf_model = TfidfBaseline()

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

    print(
        "Loading concept matcher..."
    )

    concept_matcher = (
        ConceptMatcher()
    )

    # -----------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------

    print(
        "\n[FEATURE EXTRACTION]"
    )

    X_train = build_features(
        train,
        tfidf_model,
        embedding_model,
        nli_model,
        concept_matcher,
    )

    X_validation = build_features(
        validation,
        tfidf_model,
        embedding_model,
        nli_model,
        concept_matcher,
    )

    X_test = build_features(
        test,
        tfidf_model,
        embedding_model,
        nli_model,
        concept_matcher,
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
        "Concept-aware hybrid "
        "Logistic Regression trained."
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

    feature_names = [

        # Exp 006
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

        # Exp 007A
        "answer_sentence_count",
        "avg_words_per_sentence",
        "lexical_overlap_reference",
        "lexical_overlap_question",
        "reference_keyword_coverage",
        "negation_count",
        "uncertainty_count",
        "unique_token_ratio",
        "answer_reference_length_ratio",
        "answer_question_length_ratio",

        # Exp 008
        "concept_mean_similarity",
        "concept_max_similarity",
        "concept_min_similarity",
        "concept_mean_entailment",
        "concept_max_entailment",
        "concept_mean_contradiction",
        "concept_max_contradiction",
        "concept_mean_neutral",
        "concept_strong_entailment_fraction",
        "concept_strong_contradiction_fraction",
        "concept_similarity_spread",
        "required_concept_count",
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
        "\n[FEATURE IMPORTANCE]"
    )

    print(
        importance.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print(
        "EXPERIMENT COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()