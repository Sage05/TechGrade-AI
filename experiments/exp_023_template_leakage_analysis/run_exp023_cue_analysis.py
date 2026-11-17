from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "OS": Path.home() / "Downloads" / "OS_1000_synthetic_v2.csv",
    "DBMS": Path.home() / "Downloads" / "DBMS_1000_synthetic_v2.csv",
}

CUES = {
    "Contradictory": [
        "contradicts", "opposite behavior",
        "opposite", "defining principle",
    ],
    "Correct": [
        "in short", "core idea",
        "this principle", "it matters",
    ],
    "Incorrect": [
        "regardless of correctness",
        "ignores the relevant",
        "fastest option",
    ],
    "Irrelevant": [
        "topic is interesting",
        "many different tools",
        "clear user interface",
        "also important",
    ],
    "Partially Correct": [
        "leaves out", "important conditions",
        "this explanation", "helps with",
    ],
}

ALL_CUES = [
    (label, cue)
    for label, cues in CUES.items()
    for cue in cues
]

LABELS = sorted(CUES)

def features(answers):
    matrix = np.zeros(
        (len(answers), len(ALL_CUES)), dtype=int
    )

    for j, (_, cue) in enumerate(ALL_CUES):
        pattern = re.compile(
            r"(?<!\w)" + re.escape(cue) + r"(?!\w)",
            re.IGNORECASE,
        )
        for i, answer in enumerate(answers):
            matrix[i, j] = int(bool(pattern.search(answer)))

    return matrix

def run(subject, path):
    df = pd.read_csv(path).fillna("")
    df = df[
        df["question_id"].astype(str).str.strip().ne("")
        & df["student_answer"].astype(str).str.strip().ne("")
        & df["label"].astype(str).str.strip().ne("")
    ].copy().reset_index(drop=True)

    answers = df["student_answer"].astype(str).tolist()
    X = features(answers)
    y = df["label"].astype(str).to_numpy()
    groups = df["question_id"].astype(str).to_numpy()

    cue_rows = []
    for j, (expected_label, cue) in enumerate(ALL_CUES):
        matched = X[:, j] == 1
        count = int(matched.sum())
        correct_label_count = int(
            ((y == expected_label) & matched).sum()
        )

        cue_rows.append({
            "subject": subject,
            "cue": cue,
            "expected_label": expected_label,
            "matching_rows": count,
            "matching_expected_label": correct_label_count,
            "label_purity": (
                correct_label_count / count if count else np.nan
            ),
        })

    pd.DataFrame(cue_rows).to_csv(
        OUT / f"exp023_{subject.lower()}_cue_purity.csv",
        index=False,
    )

    print(f"\n{'=' * 65}")
    print(f"{subject}: PHRASE AUDIT")
    print(f"{'=' * 65}")
    print(pd.DataFrame(cue_rows).to_string(index=False))

    cv_rows = []
    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=5).split(X, y, groups), 1
    ):
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X[train], y[train])
        pred = model.predict(X[test])

        cv_rows.append({
            "subject": subject,
            "fold": fold,
            "accuracy": accuracy_score(y[test], pred),
            "macro_f1_5_classes": f1_score(
                y[test], pred, labels=LABELS,
                average="macro", zero_division=0,
            ),
            "test_rows_with_cues": int(
                (X[test].sum(axis=1) > 0).sum()
            ),
            "test_rows": len(test),
        })

    result = pd.DataFrame(cv_rows)
    result.to_csv(
        OUT / f"exp023_{subject.lower()}_cue_cv.csv",
        index=False,
    )

    print("\nCUE-ONLY GROUPED CROSS-VALIDATION")
    print(result.to_string(index=False))
    print("\nMean accuracy:", round(result["accuracy"].mean(), 4))

def main():
    for subject, path in DATASETS.items():
        run(subject, path)

if __name__ == "__main__":
    main()
