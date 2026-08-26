"""
TechGrade AI — Experiment 009
Feature Ablation Study

Research question:

Which feature groups contribute useful, generalizable
information to technical answer grading?

Ablation groups:

A  TF-IDF only
B  Embeddings only
C  NLI only
D  Answer-quality only
E  TF-IDF + Embeddings + NLI
F  Exp 007A full hybrid
G  Concept features only
H  Exp 008 full concept-aware hybrid

Important:
No human concept annotations are used as predictive features.
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


# =========================================================
# Feature-group names
# =========================================================

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


FEATURE_NAMES = (
    FEATURE_GROUPS["tfidf"]
    + FEATURE_GROUPS["embedding"]
    + FEATURE_GROUPS["nli"]
    + FEATURE_GROUPS["answer_quality"]
    + FEATURE_GROUPS["concept"]
)


@dataclass
class ExperimentResult:
    name: str
    feature_count: int
    accuracy: float
    macro_f1: float
    weighted_f1: float


# =========================================================
# Required-concept parsing
# =========================================================

def parse_concepts(value) -> list[str]:
    """
    Parse required_concepts safely.

    Supports JSON lists, Python literal lists,
    and simple comma-separated strings.
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

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


# =========================================================
# Answer-quality features
# =========================================================

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

    detected_sentences = re.findall(
        r"[.!?]+",
        student,
    )

    sentence_count = max(
        1,
        len(detected_sentences),
    )

    avg_words_per_sentence = (
        word_count / sentence_count
    )

    student_set = set(student_tokens)
    reference_set = set(reference_tokens)
    question_set = set(question_tokens)

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

    normalized_student = student.lower()

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
        / max(1, word_count)
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
        "answer_word_count": float(
            word_count
        ),
        "answer_character_count": float(
            character_count
        ),
        "answer_sentence_count": float(
            sentence_count
        ),
        "avg_words_per_sentence": float(
            avg_words_per_sentence
        ),
        "lexical_overlap_reference": float(
            lexical_overlap_reference
        ),
        "lexical_overlap_question": float(
            lexical_overlap_question
        ),
        "reference_keyword_coverage": float(
            reference_keyword_coverage
        ),
        "negation_count": float(
            negation_count
        ),
        "uncertainty_count": float(
            uncertainty_count
        ),
        "unique_token_ratio": float(
            unique_token_ratio
        ),
        "answer_reference_length_ratio": float(
            answer_reference_length_ratio
        ),
        "answer_question_length_ratio": float(
            answer_question_length_ratio
        ),
    }


# =========================================================
# Concept features
# =========================================================

def build_concept_features(
    required_concepts: list[str],
    reference: str,
    student: str,
    concept_matcher: ConceptMatcher,
) -> dict[str, float]:

    predictions = concept_matcher.predict(
        required_concepts=required_concepts,
        model_answer=reference,
        student_answer=student,
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
        "concept_mean_similarity": float(
            np.mean(similarities)
        ),
        "concept_max_similarity": float(
            np.max(similarities)
        ),
        "concept_min_similarity": float(
            np.min(similarities)
        ),
        "concept_mean_entailment": float(
            np.mean(entailments)
        ),
        "concept_max_entailment": float(
            np.max(entailments)
        ),
        "concept_mean_contradiction": float(
            np.mean(contradictions)
        ),
        "concept_max_contradiction": float(
            np.max(contradictions)
        ),
        "concept_mean_neutral": float(
            np.mean(neutrals)
        ),
        "concept_strong_entailment_fraction": float(
            np.mean(
                np.asarray(entailments)
                >= 0.60
            )
        ),
        "concept_strong_contradiction_fraction": float(
            np.mean(
                np.asarray(contradictions)
                >= 0.75
            )
        ),
        "concept_similarity_spread": float(
            np.std(similarities)
        ),
        "required_concept_count": float(
            len(required_concepts)
        ),
    }


# =========================================================
# Complete feature extraction
# =========================================================

def extract_feature_dictionary(
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

        # -------------------------------------------------
        # TF-IDF
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Embeddings
        # -------------------------------------------------

        embedding = embedding_model.compare(
            question,
            reference,
            student,
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

        # -------------------------------------------------
        # NLI
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Answer quality
        # -------------------------------------------------

        features.update(
            build_answer_quality_features(
                question,
                reference,
                student,
            )
        )

        # -------------------------------------------------
        # Concept evidence
        # -------------------------------------------------

        required_concepts = parse_concepts(
            row["required_concepts"]
        )

        features.update(
            build_concept_features(
                required_concepts,
                reference,
                student,
                concept_matcher,
            )
        )

        rows.append(features)

    return pd.DataFrame(
        rows,
        index=df.index,
    )


# =========================================================
# Model training
# =========================================================

def train_model(
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


# =========================================================
# Evaluation
# =========================================================

def evaluate_model(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> tuple[float, float, float]:

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return (
        float(accuracy),
        float(macro_f1),
        float(weighted_f1),
    )


# =========================================================
# Main experiment
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "TechGrade AI "
            "Experiment 009 — Feature Ablation"
        )
    )

    parser.add_argument(
        "dataset",
        help="Path to revised CSV dataset.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(
        "TECHGRADE AI — EXPERIMENT 009"
    )
    print(
        "FEATURE ABLATION STUDY"
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

    required_columns = [
        "question_text",
        "model_answer",
        "student_answer",
        "required_concepts",
        "label",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
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
        "Loading TF-IDF..."
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

    # -----------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------

    print(
        "\n[FEATURE EXTRACTION]"
    )

    train_features = (
        extract_feature_dictionary(
            train,
            tfidf_model,
            embedding_model,
            nli_model,
            concept_matcher,
        )
    )

    validation_features = (
        extract_feature_dictionary(
            validation,
            tfidf_model,
            embedding_model,
            nli_model,
            concept_matcher,
        )
    )

    test_features = (
        extract_feature_dictionary(
            test,
            tfidf_model,
            embedding_model,
            nli_model,
            concept_matcher,
        )
    )

    print(
        f"Total feature count: "
        f"{len(FEATURE_NAMES)}"
    )

    # -----------------------------------------------------
    # Ablation configurations
    # -----------------------------------------------------

    configurations = {

        "A TF-IDF only": [
            "tfidf",
        ],

        "B Embeddings only": [
            "embedding",
        ],

        "C NLI only": [
            "nli",
        ],

        "D Answer-quality only": [
            "answer_quality",
        ],

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

        "G Concept features only": [
            "concept",
        ],

        "H Exp 008 Full Concept-Aware": [
            "tfidf",
            "embedding",
            "nli",
            "answer_quality",
            "concept",
        ],
    }

    # -----------------------------------------------------
    # Run configurations
    # -----------------------------------------------------

    results: list[ExperimentResult] = []

    print(
        "\n" + "=" * 70
    )

    print(
        "ABLATION RESULTS"
    )

    print(
        "=" * 70
    )

    for configuration_name, groups in (
        configurations.items()
    ):

        selected_features = []

        for group in groups:

            selected_features.extend(
                FEATURE_GROUPS[group]
            )

        # Remove accidental duplicates
        selected_features = list(
            dict.fromkeys(
                selected_features
            )
        )

        X_train = (
            train_features[
                selected_features
            ].to_numpy(
                dtype=np.float32
            )
        )

        X_test = (
            test_features[
                selected_features
            ].to_numpy(
                dtype=np.float32
            )
        )

        print(
            f"\n[{configuration_name}]"
        )

        print(
            f"Features: "
            f"{len(selected_features)}"
        )

        model = train_model(
            X_train,
            train["label"],
        )

        accuracy, macro_f1, weighted_f1 = (
            evaluate_model(
                model,
                X_test,
                test["label"],
            )
        )

        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Macro F1: "
            f"{macro_f1:.4f}"
        )

        print(
            f"Weighted F1: "
            f"{weighted_f1:.4f}"
        )

        results.append(
            ExperimentResult(
                name=configuration_name,
                feature_count=len(
                    selected_features
                ),
                accuracy=accuracy,
                macro_f1=macro_f1,
                weighted_f1=weighted_f1,
            )
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        [
            {
                "configuration": result.name,
                "features": result.feature_count,
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "weighted_f1": result.weighted_f1,
            }
            for result in results
        ]
    )

    results_df = results_df.sort_values(
        "macro_f1",
        ascending=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RANKED RESULTS — MACRO F1"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "macro_f1": "{:.4f}".format,
                "weighted_f1": "{:.4f}".format,
            },
        )
    )

    # -----------------------------------------------------
    # Best configuration
    # -----------------------------------------------------

    best = results_df.iloc[0]

    print(
        "\n[BEST CONFIGURATION]"
    )

    print(
        f"Configuration: "
        f"{best['configuration']}"
    )

    print(
        f"Feature count: "
        f"{int(best['features'])}"
    )

    print(
        f"Accuracy: "
        f"{best['accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{best['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{best['weighted_f1']:.4f}"
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