"""
TechGrade AI — Experiment 007A
Answer Quality Features

Research question:

Can answer-level linguistic and structural features improve
the hybrid grader's handling of Partially Correct and
Contradictory answers?

Important:
No human annotation fields are used as predictive features.

Allowed:
    question
    model_answer
    student_answer
    required_concepts

Forbidden:
    label
    missing_concepts
    incorrect_concepts
    annotation_confidence
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
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


# ---------------------------------------------------------
# Basic answer-quality helpers
# ---------------------------------------------------------

HEDGE_WORDS = {
    "maybe",
    "perhaps",
    "probably",
    "might",
    "possibly",
    "usually",
    "sometimes",
    "i think",
    "i believe",
}

UNCERTAINTY_PHRASES = {
    "i don't know",
    "not sure",
    "unsure",
    "i am not sure",
    "i'm not sure",
    "cannot remember",
    "can't remember",
}

NEGATION_WORDS = {
    "not",
    "never",
    "no",
    "none",
    "cannot",
    "can't",
    "doesn't",
    "doesn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "without",
}


def word_tokens(text: str) -> list[str]:
    """
    Return normalized word tokens.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        str(text).lower(),
    )


def sentence_count(text: str) -> int:
    """
    Approximate sentence count.
    """

    text = str(text).strip()

    if not text:
        return 0

    sentences = re.split(
        r"[.!?]+",
        text,
    )

    return sum(
        bool(sentence.strip())
        for sentence in sentences
    )


def count_negations(text: str) -> int:
    """
    Count simple negation markers.
    """

    tokens = word_tokens(text)

    return sum(
        token in NEGATION_WORDS
        for token in tokens
    )


def count_hesitation_signals(
    text: str,
) -> int:
    """
    Count uncertainty / hedging signals.
    """

    normalized = (
        str(text)
        .lower()
        .strip()
    )

    count = 0

    for phrase in (
        HEDGE_WORDS
        | UNCERTAINTY_PHRASES
    ):

        if phrase in normalized:
            count += 1

    return count


def lexical_overlap(
    text_a: str,
    text_b: str,
) -> float:
    """
    Calculate token-level Jaccard overlap.
    """

    a = set(
        word_tokens(text_a)
    )

    b = set(
        word_tokens(text_b)
    )

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def keyword_coverage(
    reference: str,
    student: str,
) -> float:
    """
    Calculate what fraction of reference tokens occur
    in the student answer.

    Stopwords and very short tokens are ignored.
    """

    reference_tokens = {
        token
        for token in word_tokens(
            reference
        )
        if len(token) >= 4
    }

    student_tokens = set(
        word_tokens(student)
    )

    if not reference_tokens:
        return 0.0

    return (
        len(
            reference_tokens
            & student_tokens
        )
        / len(reference_tokens)
    )


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    tfidf_model: TfidfBaseline,
    embedding_model: EmbeddingBaseline,
    nli_model: NLIBaseline,
) -> np.ndarray:
    """
    Build the Exp 007 feature matrix.

    Exp 006 features:

        0  TF-IDF question/student
        1  TF-IDF reference/student
        2  TF-IDF question/reference

        3  embedding question/student
        4  embedding reference/student
        5  embedding question/reference

        6  NLI question/student entailment
        7  NLI question/student contradiction
        8  NLI question/student neutral

        9  NLI reference/student entailment
        10 NLI reference/student contradiction
        11 NLI reference/student neutral

        12 answer word count
        13 answer character count

    New Exp 007 features:

        14 answer sentence count
        15 average words per sentence
        16 lexical overlap with reference
        17 lexical overlap with question
        18 reference keyword coverage
        19 negation count
        20 uncertainty / hedge count
        21 unique-token ratio
        22 answer/reference length ratio
        23 answer/question length ratio
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
        # Existing Exp 006 signals
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

        # -------------------------------------------------
        # Basic answer statistics
        # -------------------------------------------------

        tokens = word_tokens(
            student
        )

        word_count = len(tokens)

        character_count = len(
            student
        )

        sentences = sentence_count(
            student
        )

        if sentences > 0:
            avg_words_per_sentence = (
                word_count / sentences
            )
        else:
            avg_words_per_sentence = 0.0

        # -------------------------------------------------
        # Lexical relationships
        # -------------------------------------------------

        reference_overlap = (
            lexical_overlap(
                student,
                reference,
            )
        )

        question_overlap = (
            lexical_overlap(
                student,
                question,
            )
        )

        reference_keyword_coverage = (
            keyword_coverage(
                reference,
                student,
            )
        )

        # -------------------------------------------------
        # Linguistic signals
        # -------------------------------------------------

        negation_count = (
            count_negations(
                student
            )
        )

        uncertainty_count = (
            count_hesitation_signals(
                student
            )
        )

        unique_token_ratio = (
            len(set(tokens))
            / max(
                1,
                word_count,
            )
        )

        reference_word_count = len(
            word_tokens(reference)
        )

        question_word_count = len(
            word_tokens(question)
        )

        answer_reference_length_ratio = (
            word_count
            / max(
                1,
                reference_word_count,
            )
        )

        answer_question_length_ratio = (
            word_count
            / max(
                1,
                question_word_count,
            )
        )

        row_features = [
            # Exp 006 — TF-IDF
            float(
                tfidf.question_student
            ),
            float(
                tfidf.reference_student
            ),
            float(
                tfidf.question_reference_student
            ),

            # Exp 006 — embeddings
            float(
                embedding.question_student
            ),
            float(
                embedding.reference_student
            ),
            float(
                embedding.question_reference_student
            ),

            # Exp 006 — NLI
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

            # Exp 006 — length
            float(word_count),
            float(character_count),

            # Exp 007 — answer quality
            float(sentences),
            float(avg_words_per_sentence),
            float(reference_overlap),
            float(question_overlap),
            float(reference_keyword_coverage),
            float(negation_count),
            float(uncertainty_count),
            float(unique_token_ratio),
            float(answer_reference_length_ratio),
            float(answer_question_length_ratio),
        ]

        features.append(
            row_features
        )

    return np.asarray(
        features,
        dtype=np.float32,
    )


# ---------------------------------------------------------
# Classifier
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

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
            "Experiment 007A."
        )
    )

    parser.add_argument(
        "dataset",
        help="Path to the revised CSV dataset.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TECHGRADE AI — EXPERIMENT 007A")
    print("ANSWER QUALITY FEATURES")
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

    # Fit TF-IDF only on training text.
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
    # Features
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
        "Answer-quality hybrid "
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
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()