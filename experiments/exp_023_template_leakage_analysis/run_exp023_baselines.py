from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "OS": Path.home() / "Downloads" / "OS_1000_synthetic_v2.csv",
    "DBMS": Path.home() / "Downloads" / "DBMS_1000_synthetic_v2.csv",
}

def clean(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def run(subject, path):
    df = pd.read_csv(path).fillna("")
    qcol = "question_text" if "question_text" in df else "question"

    required = [qcol, "question_id", "student_answer", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{subject}: missing {missing}")

    df = df[
        (df["question_id"].astype(str).str.strip() != "")
        & (df["student_answer"].astype(str).str.strip() != "")
        & (df["label"].astype(str).str.strip() != "")
    ].copy().reset_index(drop=True)

    y = df["label"].astype(str).to_numpy()
    groups = df["question_id"].astype(str).to_numpy()
    questions = df[qcol].map(clean).to_numpy()
    answers = df["student_answer"].map(clean).to_numpy()
    all_labels = sorted(set(y))

    splitter = GroupKFold(n_splits=5)
    records = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(df, y, groups), start=1
    ):
        y_train, y_test = y[train_idx], y[test_idx]

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)

        train_answers = set(answers[train_idx])
        test_answers = set(answers[test_idx])

        shared_answers = train_answers & test_answers
        seen_answer_rows = sum(
            answer in train_answers for answer in answers[test_idx]
        )

        inputs = {
            "Majority class": (
                np.zeros((len(train_idx), 1)),
                np.zeros((len(test_idx), 1)),
            ),
            "Question only": (
                questions[train_idx],
                questions[test_idx],
            ),
            "Answer only": (
                answers[train_idx],
                answers[test_idx],
            ),
        }

        for method, (x_train, x_test) in inputs.items():
            if method == "Majority class":
                model = DummyClassifier(strategy="most_frequent")
            else:
                model = make_pipeline(
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        min_df=2,
                    ),
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                )

            model.fit(x_train, y_train)
            predictions = model.predict(x_test)

            records.append({
                "subject": subject,
                "fold": fold,
                "method": method,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_question_ids": len(train_groups),
                "test_question_ids": len(test_groups),
                "train_labels": ", ".join(sorted(set(y_train))),
                "test_labels": ", ".join(sorted(set(y_test))),
                "missing_train_labels": ", ".join(
                    sorted(set(all_labels) - set(y_train))
                ),
                "missing_test_labels": ", ".join(
                    sorted(set(all_labels) - set(y_test))
                ),
                "shared_exact_answer_types": len(shared_answers),
                "test_rows_with_seen_answers": int(seen_answer_rows),
                "accuracy": accuracy_score(y_test, predictions),
                "macro_f1_all_labels": f1_score(
                    y_test,
                    predictions,
                    labels=all_labels,
                    average="macro",
                    zero_division=0,
                ),
            })

    result = pd.DataFrame(records)
    result.to_csv(
        OUT / f"exp023_{subject.lower()}_baselines.csv",
        index=False,
    )

    print(f"\n{'=' * 65}")
    print(f"{subject} — GROUPED CROSS-VALIDATION")
    print(f"{'=' * 65}")
    print(
        result.groupby("method")[
            ["accuracy", "macro_f1_all_labels"]
        ].agg(["mean", "std"]).round(4).to_string()
    )

    print("\nFold label coverage and exact-answer overlap:")
    print(
        result[result["method"] == "Majority class"][
            [
                "fold",
                "test_rows",
                "test_labels",
                "missing_test_labels",
                "shared_exact_answer_types",
                "test_rows_with_seen_answers",
            ]
        ].to_string(index=False)
    )

    return result

def main():
    results = [run(subject, path) for subject, path in DATASETS.items()]
    pd.concat(results, ignore_index=True).to_csv(
        OUT / "exp023_baselines_combined.csv",
        index=False,
    )
    print("\nPhase 2 completed.")

if __name__ == "__main__":
    main()
