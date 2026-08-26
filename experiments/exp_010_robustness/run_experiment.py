"""
TechGrade AI — Experiment 010
Robustness / Stability Study

Purpose:
    Evaluate whether the strongest feature configurations remain
    effective across multiple question-level train/test splits.

Configurations:
    E  TF-IDF + Embeddings + NLI
    F  Experiment 007A Full Hybrid
    D  Answer-quality only
    H  Experiment 008 Full Concept-Aware

Protocol:
    - Question-level splitting
    - Repeated holdout
    - Same split used by every configuration
    - No human concept annotations used as predictive features
    - Report mean and standard deviation across repetitions

Primary metric:
    Macro F1
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from src.models.embedding_baseline import EmbeddingBaseline
from src.models.nli_baseline import NLIBaseline
from src.models.tfidf_baseline import TfidfBaseline
from src.models.concept_matcher import ConceptMatcher


# ============================================================
# Configuration
# ============================================================

N_REPEATS = 5

TEST_QUESTION_FRACTION = 0.20

RANDOM_SEEDS = [
    11,
    22,
    33,
    44,
    55,
]

LABELS = [
    "Correct",
    "Incorrect",
    "Partially Correct",
    "Irrelevant",
    "Contradictory",
]


FEATURE_GROUPS = {
    "tfidf": [
        "tfidf_question_student",
        "tfidf_reference_student",
        "tfidf_question_reference",
    ],

    "embedding": [
        "embedding_question_student",
        "embedding_reference_student",
        "embedding_question_reference",
    ],

    "nli": [
        "nli_question_entailment",
        "nli_question_contradiction",
        "nli_question_neutral",
        "nli_reference_entailment",
        "nli_reference_contradiction",
        "nli_reference_neutral",
    ],

    "answer_quality": [
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
    ],

    "concept": [
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
    ],
}


CONFIGURATIONS = {
    "E TF-IDF + Embeddings + NLI": [
        "tfidf",
        "embedding",
        "nli",
    ],

    "F Exp 007A Full Hybrid": [
        "tfidf",
        "embedding",
        "nli",
        "answer_quality",
    ],

    "D Answer-quality only": [
        "answer_quality",
    ],

    "H Exp 008 Full Concept-Aware": [
        "tfidf",
        "embedding",
        "nli",
        "answer_quality",
        "concept",
    ],
}


@dataclass
class SplitResult:
    repetition: int
    seed: int
    configuration: str
    train_questions: int
    test_questions: int
    accuracy: float
    macro_f1: float
    weighted_f1: float


# ============================================================
# Concept parsing
# ============================================================

def parse_concepts(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except Exception:
        pass

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


# ============================================================
# Question-level split
# ============================================================

def question_level_holdout(
    df: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(seed)

    questions = (
        df["question_text"]
        .drop_duplicates()
        .tolist()
    )

    questions = np.asarray(
        questions,
        dtype=object,
    )

    rng.shuffle(questions)

    test_count = max(
        1,
        int(
            round(
                len(questions)
                * TEST_QUESTION_FRACTION
            )
        ),
    )

    test_questions = set(
        questions[:test_count]
    )

    train_mask = ~df[
        "question_text"
    ].isin(test_questions)

    test_mask = df[
        "question_text"
    ].isin(test_questions)

    train = df.loc[
        train_mask
    ].copy()

    test = df.loc[
        test_mask
    ].copy()

    train_questions = set(
        train["question_text"]
    )

    test_questions_actual = set(
        test["question_text"]
    )

    overlap = (
        train_questions
        & test_questions_actual
    )

    if overlap:
        raise RuntimeError(
            "Question leakage detected: "
            f"{len(overlap)} overlapping questions."
        )

    return train, test


# ============================================================
# Answer-quality features
# ============================================================

def build_answer_quality_features(
    question: str,
    reference: str,
    student: str,
) -> dict[str, float]:

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

    word_count = len(student_tokens)

    character_count = len(student)

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
        lexical_overlap_reference = (
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
        lexical_overlap_reference = 0.0

    if student_set and question_set:
        lexical_overlap_question = (
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
        lexical_overlap_question = 0.0

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
        len(student_set)
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

    return {
        "answer_word_count":
            float(word_count),

        "answer_character_count":
            float(character_count),

        "answer_sentence_count":
            float(sentence_count),

        "avg_words_per_sentence":
            float(
                avg_words_per_sentence
            ),

        "lexical_overlap_reference":
            float(
                lexical_overlap_reference
            ),

        "lexical_overlap_question":
            float(
                lexical_overlap_question
            ),

        "reference_keyword_coverage":
            float(
                reference_keyword_coverage
            ),

        "negation_count":
            float(
                negation_count
            ),

        "uncertainty_count":
            float(
                uncertainty_count
            ),

        "unique_token_ratio":
            float(
                unique_token_ratio
            ),

        "answer_reference_length_ratio":
            float(
                answer_reference_length_ratio
            ),

        "answer_question_length_ratio":
            float(
                answer_question_length_ratio
            ),
    }


# ============================================================
# Concept features
# ============================================================

def build_concept_features(
    required_concepts: list[str],
    reference: str,
    student: str,
    concept_matcher: ConceptMatcher,
) -> dict[str, float]:

    predictions = (
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
        for prediction in predictions
    ]

    entailments = [
        float(
            prediction.entailment
        )
        for prediction in predictions
    ]

    contradictions = [
        float(
            prediction.contradiction
        )
        for prediction in predictions
    ]

    neutrals = [
        float(
            prediction.neutral
        )
        for prediction in predictions
    ]

    if not similarities:
        return {
            "concept_mean_similarity": 0.0,
            "concept_max_similarity": 0.0,
            "concept_min_similarity": 0.0,
            "concept_mean_entailment": 0.0,
            "concept_max_entailment": 0.0,
            "concept_mean_contradiction": 0.0,
            "concept_max_contradiction": 0.0,
            "concept_mean_neutral": 1.0,
            "concept_strong_entailment_fraction": 0.0,
            "concept_strong_contradiction_fraction": 0.0,
            "concept_similarity_spread": 0.0,
            "required_concept_count": 0.0,
        }

    return {
        "concept_mean_similarity":
            float(
                np.mean(
                    similarities
                )
            ),

        "concept_max_similarity":
            float(
                np.max(
                    similarities
                )
            ),

        "concept_min_similarity":
            float(
                np.min(
                    similarities
                )
            ),

        "concept_mean_entailment":
            float(
                np.mean(
                    entailments
                )
            ),

        "concept_max_entailment":
            float(
                np.max(
                    entailments
                )
            ),

        "concept_mean_contradiction":
            float(
                np.mean(
                    contradictions
                )
            ),

        "concept_max_contradiction":
            float(
                np.max(
                    contradictions
                )
            ),

        "concept_mean_neutral":
            float(
                np.mean(
                    neutrals
                )
            ),

        "concept_strong_entailment_fraction":
            float(
                np.mean(
                    np.asarray(
                        entailments
                    )
                    >= 0.60
                )
            ),

        "concept_strong_contradiction_fraction":
            float(
                np.mean(
                    np.asarray(
                        contradictions
                    )
                    >= 0.75
                )
            ),

        "concept_similarity_spread":
            float(
                np.std(
                    similarities
                )
            ),

        "required_concept_count":
            float(
                len(
                    required_concepts
                )
            ),
    }


# ============================================================
# Feature extraction
# ============================================================

def extract_features(
    df: pd.DataFrame,
    tfidf_model: TfidfBaseline,
    embedding_model: EmbeddingBaseline,
    nli_model: NLIBaseline,
    concept_matcher: ConceptMatcher,
) -> pd.DataFrame:

    rows = []

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

        features = {}

        # -------------------------------
        # TF-IDF
        # -------------------------------

        tfidf = tfidf_model.compare(
            question,
            reference,
            student,
        )

        features.update(
            {
                "tfidf_question_student":
                    float(
                        tfidf.question_student
                    ),

                "tfidf_reference_student":
                    float(
                        tfidf.reference_student
                    ),

                "tfidf_question_reference":
                    float(
                        tfidf.question_reference_student
                    ),
            }
        )

        # -------------------------------
        # Embeddings
        # -------------------------------

        embedding = (
            embedding_model.compare(
                question,
                reference,
                student,
            )
        )

        features.update(
            {
                "embedding_question_student":
                    float(
                        embedding.question_student
                    ),

                "embedding_reference_student":
                    float(
                        embedding.reference_student
                    ),

                "embedding_question_reference":
                    float(
                        embedding.question_reference_student
                    ),
            }
        )

        # -------------------------------
        # NLI
        # -------------------------------

        nli = nli_model.compare(
            question,
            reference,
            student,
        )

        features.update(
            {
                "nli_question_entailment":
                    float(
                        nli.question_entailment
                    ),

                "nli_question_contradiction":
                    float(
                        nli.question_contradiction
                    ),

                "nli_question_neutral":
                    float(
                        nli.question_neutral
                    ),

                "nli_reference_entailment":
                    float(
                        nli.reference_entailment
                    ),

                "nli_reference_contradiction":
                    float(
                        nli.reference_contradiction
                    ),

                "nli_reference_neutral":
                    float(
                        nli.reference_neutral
                    ),
            }
        )

        # -------------------------------
        # Answer quality
        # -------------------------------

        features.update(
            build_answer_quality_features(
                question,
                reference,
                student,
            )
        )

        # -------------------------------
        # Concept features
        # -------------------------------

        required_concepts = (
            parse_concepts(
                row["required_concepts"]
            )
        )

        features.update(
            build_concept_features(
                required_concepts,
                reference,
                student,
                concept_matcher,
            )
        )

        rows.append(
            features
        )

    return pd.DataFrame(
        rows,
        index=df.index,
    )


# ============================================================
# Train / evaluate
# ============================================================

def train_and_evaluate(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_labels: pd.Series,
    test_labels: pd.Series,
    feature_names: list[str],
) -> tuple[float, float, float]:

    X_train = (
        train_features[
            feature_names
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    X_test = (
        test_features[
            feature_names
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train,
        train_labels,
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        test_labels,
        predictions,
    )

    macro_f1 = f1_score(
        test_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        test_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return (
        float(accuracy),
        float(macro_f1),
        float(weighted_f1),
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "TechGrade AI — "
            "Experiment 010 Robustness Study"
        )
    )

    parser.add_argument(
        "dataset",
        help="Path to revised CSV dataset.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(
        "TECHGRADE AI — EXPERIMENT 010"
    )
    print(
        "ROBUSTNESS / STABILITY STUDY"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        args.dataset
    )

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
            + ", ".join(
                missing_columns
            )
        )

    question_count = (
        df["question_text"]
        .nunique()
    )

    print("\n[DATASET]")

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Unique questions: "
        f"{question_count}"
    )

    print(
        f"Repeated splits: "
        f"{N_REPEATS}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print("\n[MODELS]")

    print(
        "Loading TF-IDF..."
    )

    tfidf_model = TfidfBaseline()

    # TF-IDF vocabulary is deliberately fitted once on the
    # complete text corpus. This keeps the representation
    # identical across repeated evaluation splits.
    tfidf_corpus = (
        df["question_text"].tolist()
        + df["model_answer"].tolist()
        + df["student_answer"].tolist()
    )

    tfidf_model.fit(
        tfidf_corpus
    )

    print(
        "Loading embeddings..."
    )

    embedding_model = (
        EmbeddingBaseline()
    )

    print(
        "Loading NLI..."
    )

    nli_model = NLIBaseline()

    print(
        "Loading concept matcher..."
    )

    concept_matcher = (
        ConceptMatcher()
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print(
        "\n[FEATURE EXTRACTION]"
    )

    all_features = extract_features(
        df,
        tfidf_model,
        embedding_model,
        nli_model,
        concept_matcher,
    )

    print(
        f"Feature count: "
        f"{len(all_features.columns)}"
    )

    # --------------------------------------------------------
    # Repeated evaluation
    # --------------------------------------------------------

    results: list[SplitResult] = []

    print(
        "\n" + "=" * 70
    )

    print(
        "REPEATED QUESTION-LEVEL EVALUATION"
    )

    print(
        "=" * 70
    )

    for repetition, seed in enumerate(
        RANDOM_SEEDS,
        start=1,
    ):

        print(
            f"\n[REPETITION {repetition}/{N_REPEATS}]"
        )

        print(
            f"Seed: {seed}"
        )

        train, test = (
            question_level_holdout(
                df,
                seed,
            )
        )

        print(
            f"Train: {len(train)} rows / "
            f"{train.question_text.nunique()} questions"
        )

        print(
            f"Test: {len(test)} rows / "
            f"{test.question_text.nunique()} questions"
        )

        train_features = (
            all_features.loc[
                train.index
            ]
        )

        test_features = (
            all_features.loc[
                test.index
            ]
        )

        for configuration_name, groups in (
            CONFIGURATIONS.items()
        ):

            feature_names = []

            for group in groups:

                feature_names.extend(
                    FEATURE_GROUPS[group]
                )

            feature_names = list(
                dict.fromkeys(
                    feature_names
                )
            )

            accuracy, macro_f1, weighted_f1 = (
                train_and_evaluate(
                    train_features,
                    test_features,
                    train["label"],
                    test["label"],
                    feature_names,
                )
            )

            print(
                f"  {configuration_name}: "
                f"Macro F1={macro_f1:.4f} "
                f"Accuracy={accuracy:.4f}"
            )

            results.append(
                SplitResult(
                    repetition=repetition,
                    seed=seed,
                    configuration=(
                        configuration_name
                    ),
                    train_questions=(
                        train[
                            "question_text"
                        ].nunique()
                    ),
                    test_questions=(
                        test[
                            "question_text"
                        ].nunique()
                    ),
                    accuracy=accuracy,
                    macro_f1=macro_f1,
                    weighted_f1=weighted_f1,
                )
            )

    # --------------------------------------------------------
    # Convert results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        [
            {
                "repetition":
                    result.repetition,

                "seed":
                    result.seed,

                "configuration":
                    result.configuration,

                "train_questions":
                    result.train_questions,

                "test_questions":
                    result.test_questions,

                "accuracy":
                    result.accuracy,

                "macro_f1":
                    result.macro_f1,

                "weighted_f1":
                    result.weighted_f1,
            }
            for result in results
        ]
    )

    # --------------------------------------------------------
    # Per-split table
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PER-SPLIT RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        results_df[
            [
                "repetition",
                "configuration",
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        ].to_string(
            index=False,
            formatters={
                "accuracy":
                    "{:.4f}".format,

                "macro_f1":
                    "{:.4f}".format,

                "weighted_f1":
                    "{:.4f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    summary = (
        results_df
        .groupby(
            "configuration"
        )
        .agg(
            accuracy_mean=(
                "accuracy",
                "mean",
            ),
            accuracy_std=(
                "accuracy",
                "std",
            ),
            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),
            macro_f1_std=(
                "macro_f1",
                "std",
            ),
            weighted_f1_mean=(
                "weighted_f1",
                "mean",
            ),
            weighted_f1_std=(
                "weighted_f1",
                "std",
            ),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "macro_f1_mean",
        ascending=False,
    )

    # --------------------------------------------------------
    # Summary output
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ROBUSTNESS SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "accuracy_mean":
                    "{:.4f}".format,

                "accuracy_std":
                    "{:.4f}".format,

                "macro_f1_mean":
                    "{:.4f}".format,

                "macro_f1_std":
                    "{:.4f}".format,

                "weighted_f1_mean":
                    "{:.4f}".format,

                "weighted_f1_std":
                    "{:.4f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    best = summary.iloc[0]

    print(
        "\n[BEST CONFIGURATION]"
    )

    print(
        f"Configuration: "
        f"{best['configuration']}"
    )

    print(
        f"Mean Accuracy: "
        f"{best['accuracy_mean']:.4f}"
    )

    print(
        f"Accuracy Std: "
        f"{best['accuracy_std']:.4f}"
    )

    print(
        f"Mean Macro F1: "
        f"{best['macro_f1_mean']:.4f}"
    )

    print(
        f"Macro F1 Std: "
        f"{best['macro_f1_std']:.4f}"
    )

    print(
        f"Mean Weighted F1: "
        f"{best['weighted_f1_mean']:.4f}"
    )

    print(
        f"Weighted F1 Std: "
        f"{best['weighted_f1_std']:.4f}"
    )

    # --------------------------------------------------------
    # Stability interpretation
    # --------------------------------------------------------

    most_stable = summary.sort_values(
        "macro_f1_std",
        ascending=True,
    ).iloc[0]

    print(
        "\n[MOST STABLE CONFIGURATION]"
    )

    print(
        f"Configuration: "
        f"{most_stable['configuration']}"
    )

    print(
        f"Macro F1 Std: "
        f"{most_stable['macro_f1_std']:.4f}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()