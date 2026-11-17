
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import GroupShuffleSplit
from src.models.tfidf_baseline import TfidfBaseline
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.nli_baseline import NLIBaseline
from src.models.concept_matcher import ConceptMatcher
from src.grading.feature_extractor import FeatureExtractor
from src.grading.hybrid_grader import HybridGrader


ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "OS": Path(
        Path.home() / "Downloads" / "OS_1000_synthetic_v2.csv"
    ),
    "DBMS": Path(
        Path.home() / "Downloads" / "DBMS_1000_synthetic_v2.csv"
    ),
}

OUTPUT = Path(__file__).resolve().parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)


def prepare_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # Normalize question column for existing project modules.
    if "question_text" not in df.columns:
        if "question" not in df.columns:
            raise ValueError("No question or question_text column.")
        df["question_text"] = df["question"]

    required = [
        "question_id",
        "question_text",
        "model_answer",
        "student_answer",
        "required_concepts",
        "label",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in required:
        df[col] = df[col].fillna("").astype(str)

    df = df[
        (df["question_id"] != "")
        & (df["student_answer"] != "")
        & (df["label"] != "")
    ].copy()

    return df.reset_index(drop=True)


def extract_features(df, extractor, description):
    X = []

    print(f"Extracting {description} features: {len(df)} rows")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        features = extractor.extract(
            question=row["question_text"],
            reference_answer=row["model_answer"],
            student_answer=row["student_answer"],
            required_concepts=row["required_concepts"],
        )

        X.append(features.values)

        if i % 100 == 0:
            print(f"  Processed {i}/{len(df)}")

    return np.asarray(X, dtype=float)

def split_by_question(df):
    groups = df["question_id"]

    # First split: 80% training, 20% temporary
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, temp_idx = next(
        first_split.split(df, groups=groups)
    )

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # Second split: divide temporary data equally
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42
    )

    val_idx, test_idx = next(
        second_split.split(temp_df, groups=temp_df["question_id"])
    )

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    return train_df, val_df, test_df



def run_experiment(subject, path):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 022 — {subject}")
    print("=" * 70)

    df = prepare_dataset(path)

    print(f"Total rows: {len(df)}")
    print(f"Unique questions: {df['question_id'].nunique()}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    # Preserve the project's existing split protocol.
    train_df, val_df, test_df = split_by_question(df)

    train_questions = set(train_df["question_id"])
    val_questions = set(val_df["question_id"])
    test_questions = set(test_df["question_id"])

    assert train_questions.isdisjoint(val_questions)
    assert train_questions.isdisjoint(test_questions)
    assert val_questions.isdisjoint(test_questions)

    print("\nSplit sizes:")
    print(f"Train: {len(train_df)} rows / {len(train_questions)} questions")
    print(f"Validation: {len(val_df)} rows / {len(val_questions)} questions")
    print(f"Test: {len(test_df)} rows / {len(test_questions)} questions")
    print("Question leakage: NONE")

    # Fit TF-IDF only on training questions and reference answers.
    tfidf = TfidfBaseline()
    tfidf.fit(
        (
            train_df["question_text"]
            + " "
            + train_df["model_answer"]
        ).tolist()
    )

    embedding = EmbeddingBaseline()
    nli = NLIBaseline()
    concept_matcher = ConceptMatcher()

    extractor = FeatureExtractor(
        tfidf_model=tfidf,
        embedding_model=embedding,
        nli_model=nli,
        concept_matcher=concept_matcher,
    )

    X_train = extract_features(train_df, extractor, "training")
    X_val = extract_features(val_df, extractor, "validation")
    X_test = extract_features(test_df, extractor, "test")

    y_train = train_df["label"].tolist()
    y_test = test_df["label"].tolist()

    print(f"\nFeature count: {X_train.shape[1]}")

    grader = HybridGrader(extractor)
    grader.fit(X_train, y_train)

    predictions = []
    probability_rows = []

    for features in X_test:
        result = grader.predict_features(features)
        predictions.append(result.label)
        probability_rows.append(result.probabilities)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    print("\nClassification results:")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nPer-class report:")
    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True,
    )
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    labels = sorted(set(y_train) | set(y_test) | set(predictions))

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).to_csv(OUTPUT / f"exp022_{subject.lower()}_confusion_matrix.csv")

    results = test_df[
        ["entry_id", "question_id", "question_text",
         "student_answer", "label", "score"]
    ].copy()

    results["predicted_label"] = predictions
    results["correct"] = (
        results["label"] == results["predicted_label"]
    )

    results.to_csv(
        OUTPUT / f"exp022_{subject.lower()}_predictions.csv",
        index=False,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT / f"exp022_{subject.lower()}_classification_report.csv"
    )

    summary = {
        "subject": subject,
        "rows": len(df),
        "unique_questions": int(df["question_id"].nunique()),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "feature_count": int(X_train.shape[1]),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "question_leakage": False,
        "labels": labels,
    }

    with open(
        OUTPUT / f"exp022_{subject.lower()}_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=4)

    return summary


def main():
    summaries = []

    for subject, path in DATASETS.items():
        summaries.append(run_experiment(subject, path))

    pd.DataFrame(summaries).to_csv(
        OUTPUT / "exp022_combined_summary.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT 022 COMPLETED")
    print("=" * 70)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
