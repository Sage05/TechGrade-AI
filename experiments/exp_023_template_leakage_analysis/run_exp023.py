from pathlib import Path
from collections import Counter
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "OS": Path.home() / "Downloads" / "OS_1000_synthetic_v2.csv",
    "DBMS": Path.home() / "Downloads" / "DBMS_1000_synthetic_v2.csv",
}


def normalize(text):
    text = str(text).lower().strip()
    return " ".join(text.split())


def template_normalize(text):
    text = normalize(text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def audit(subject, path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path).fillna("")

    required = [
        "question_id",
        "student_answer",
        "label",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{subject}: missing columns {missing}")

    question_col = (
        "question_text" if "question_text" in df.columns
        else "question"
    )

    if question_col not in df.columns:
        raise ValueError(f"{subject}: missing question text")

    df["normalized_question"] = df[question_col].map(normalize)
    df["normalized_answer"] = df["student_answer"].map(normalize)
    df["answer_template"] = df["student_answer"].map(template_normalize)

    question_labels = df.groupby("question_id")["label"].nunique()
    question_texts = df.groupby("question_id")["normalized_question"].nunique()

    answer_groups = df.groupby("normalized_answer")
    template_groups = df.groupby("answer_template")

    repeated_answers = answer_groups.size()
    repeated_templates = template_groups.size()

    cross_question_answers = (
        answer_groups["question_id"].nunique() > 1
    ).sum()

    cross_question_templates = (
        template_groups["question_id"].nunique() > 1
    ).sum()

    report = {
        "subject": subject,
        "total_rows": int(len(df)),
        "unique_question_ids": int(df["question_id"].nunique()),
        "unique_question_texts": int(df["normalized_question"].nunique()),
        "unique_student_answers": int(df["normalized_answer"].nunique()),
        "unique_answer_templates": int(df["answer_template"].nunique()),
        "exact_duplicate_answer_groups": int((repeated_answers > 1).sum()),
        "repeated_template_groups": int((repeated_templates > 1).sum()),
        "answers_reused_across_question_ids": int(cross_question_answers),
        "templates_reused_across_question_ids": int(cross_question_templates),
        "question_ids_with_only_one_label": int((question_labels == 1).sum()),
        "question_ids_with_multiple_labels": int((question_labels > 1).sum()),
        "question_ids_with_multiple_question_texts": int(
            (question_texts > 1).sum()
        ),
        "label_distribution": {
            str(k): int(v)
            for k, v in Counter(df["label"]).items()
        },
    }

    output_file = OUTPUT / f"exp023_{subject.lower()}_audit.json"
    output_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT 023 — {subject}")
    print(f"{'=' * 60}")

    for key, value in report.items():
        print(f"{key}: {value}")

    return report


def main():
    reports = []

    for subject, path in DATASETS.items():
        reports.append(audit(subject, path))

    summary = OUTPUT / "exp023_initial_audit.json"
    summary.write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )

    print("\nInitial leakage audit completed.")


if __name__ == "__main__":
    main()
