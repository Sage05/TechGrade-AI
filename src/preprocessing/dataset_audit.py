"""
TechGrade AI - Dataset Audit

Reusable dataset validation and exploratory audit utilities.

This module is model-independent and is intended to identify:
- schema problems
- missing values
- duplicate data
- question diversity
- label imbalance
- subject/difficulty distribution
- duplicate-answer conflicts
- potential dataset construction issues
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Expected schema
# ---------------------------------------------------------------------

CORE_COLUMNS = {
    "entry_id",
    "subject",
    "topic",
    "question_text",
    "model_answer",
    "student_answer",
    "label",
    "key_concepts",
    "difficulty_level",
    "annotator_id",
    "annotation_confidence",
}

OPTIONAL_COLUMNS = {
    "human_score",
    "required_concepts",
    "supporting_concepts",
    "missing_concepts",
    "incorrect_concepts",
    "answer_source",
}


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a TechGrade AI CSV dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected CSV file, received: {path.suffix}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


# ---------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> dict:
    """Validate the dataset against the expected schema."""

    actual_columns = set(df.columns)

    missing_core = sorted(CORE_COLUMNS - actual_columns)
    present_optional = sorted(OPTIONAL_COLUMNS & actual_columns)

    return {
        "valid_core_schema": len(missing_core) == 0,
        "missing_core_columns": missing_core,
        "present_optional_columns": present_optional,
        "all_columns": list(df.columns),
    }


# ---------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------

def inspect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages."""

    missing_count = df.isna().sum()
    missing_percentage = (missing_count / len(df)) * 100

    report = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage.round(2),
        }
    )

    return report.sort_values(
        "missing_count",
        ascending=False,
    )


# ---------------------------------------------------------------------
# Duplicate analysis
# ---------------------------------------------------------------------

def inspect_duplicates(df: pd.DataFrame) -> dict:
    """Inspect exact duplicate rows and duplicate identifiers."""

    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_entry_ids": (
            int(df["entry_id"].duplicated().sum())
            if "entry_id" in df.columns
            else None
        ),
        "duplicate_student_answers": (
            int(df["student_answer"].duplicated().sum())
            if "student_answer" in df.columns
            else None
        ),
        "duplicate_questions": (
            int(df["question_text"].duplicated().sum())
            if "question_text" in df.columns
            else None
        ),
    }


# ---------------------------------------------------------------------
# Per-question analysis
# ---------------------------------------------------------------------

def build_question_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a question-level report.

    Each row represents one unique question and contains:
    - number of answer variants
    - label distribution
    - difficulty
    """

    required = {
        "question_text",
        "label",
        "difficulty_level",
    }

    if not required.issubset(df.columns):
        return pd.DataFrame()

    grouped = (
        df.groupby("question_text")
        .agg(
            answer_count=("student_answer", "count"),
            unique_answers=("student_answer", "nunique"),
            difficulty=("difficulty_level", "first"),
        )
        .reset_index()
    )

    label_counts = (
        pd.crosstab(
            df["question_text"],
            df["label"],
        )
        .reset_index()
    )

    report = grouped.merge(
        label_counts,
        on="question_text",
        how="left",
    )

    return report.sort_values(
        "answer_count",
        ascending=False,
    )


def inspect_questions(df: pd.DataFrame) -> dict:
    """Summarize question diversity."""

    if "question_text" not in df.columns:
        return {}

    question_counts = df["question_text"].value_counts()

    return {
        "unique_questions": int(
            df["question_text"].nunique()
        ),
        "answers_per_question_mean": round(
            float(question_counts.mean()),
            2,
        ),
        "answers_per_question_min": int(
            question_counts.min()
        ),
        "answers_per_question_max": int(
            question_counts.max()
        ),
    }


# ---------------------------------------------------------------------
# Duplicate student-answer analysis
# ---------------------------------------------------------------------

def build_duplicate_answer_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find student answers that occur more than once.

    This helps distinguish legitimate repeated answers from
    potentially problematic duplicated/synthetic examples.
    """

    required = {
        "student_answer",
        "question_text",
        "label",
        "entry_id",
    }

    if not required.issubset(df.columns):
        return pd.DataFrame()

    counts = df["student_answer"].value_counts()

    duplicated_answers = counts[counts > 1].index

    report = df[
        df["student_answer"].isin(duplicated_answers)
    ].copy()

    report = report.sort_values(
        "student_answer"
    )

    return report[
        [
            "student_answer",
            "entry_id",
            "question_text",
            "label",
        ]
    ]


def inspect_duplicate_answer_conflicts(
    df: pd.DataFrame,
) -> dict:
    """
    Analyze repeated student answers for potential conflicts.
    """

    if "student_answer" not in df.columns:
        return {}

    grouped = df.groupby("student_answer")

    repeated = grouped.filter(
        lambda x: len(x) > 1
    )

    if repeated.empty:
        return {
            "repeated_answers": 0,
            "answers_with_multiple_labels": 0,
            "answers_used_for_multiple_questions": 0,
        }

    label_counts = (
        repeated.groupby("student_answer")["label"]
        .nunique()
    )

    question_counts = (
        repeated.groupby("student_answer")["question_text"]
        .nunique()
    )

    return {
        "repeated_answers": int(
            repeated["student_answer"].nunique()
        ),
        "answers_with_multiple_labels": int(
            (label_counts > 1).sum()
        ),
        "answers_used_for_multiple_questions": int(
            (question_counts > 1).sum()
        ),
    }


# ---------------------------------------------------------------------
# Label analysis
# ---------------------------------------------------------------------

def inspect_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return label counts and percentages."""

    if "label" not in df.columns:
        return pd.DataFrame()

    counts = df["label"].value_counts(
        dropna=False
    )

    percentages = (counts / len(df)) * 100

    return pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages.round(2),
        }
    )


# ---------------------------------------------------------------------
# Subject analysis
# ---------------------------------------------------------------------

def inspect_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """Return subject counts and percentages."""

    if "subject" not in df.columns:
        return pd.DataFrame()

    counts = df["subject"].value_counts(
        dropna=False
    )

    percentages = (counts / len(df)) * 100

    return pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages.round(2),
        }
    )


# ---------------------------------------------------------------------
# Difficulty analysis
# ---------------------------------------------------------------------

def inspect_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Return difficulty distribution."""

    if "difficulty_level" not in df.columns:
        return pd.DataFrame()

    return (
        df["difficulty_level"]
        .value_counts(dropna=False)
        .to_frame("count")
    )


# ---------------------------------------------------------------------
# Answer length analysis
# ---------------------------------------------------------------------

def inspect_answer_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze student-answer length."""

    if "student_answer" not in df.columns:
        return pd.DataFrame()

    answers = (
        df["student_answer"]
        .fillna("")
        .astype(str)
    )

    report = pd.DataFrame(
        {
            "character_count": answers.str.len(),
            "word_count": answers.str.split().str.len(),
        }
    )

    return report.describe().round(2)


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def run_audit(path: str | Path) -> None:
    """Run the complete dataset audit."""

    df = load_dataset(path)

    print("=" * 70)
    print("TECHGRADE AI — DATASET AUDIT")
    print("=" * 70)

    print("\n[DATASET]")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # Schema
    print("\n[SCHEMA]")

    schema = validate_schema(df)

    print(
        f"Core schema valid: "
        f"{schema['valid_core_schema']}"
    )

    if schema["missing_core_columns"]:
        print("Missing core columns:")

        for column in schema["missing_core_columns"]:
            print(f"  - {column}")

    if schema["present_optional_columns"]:
        print("Optional columns present:")

        for column in schema["present_optional_columns"]:
            print(f"  - {column}")

    # Missing values
    print("\n[MISSING VALUES]")
    print(inspect_missing_values(df))

    # Duplicates
    print("\n[DUPLICATES]")

    duplicate_summary = inspect_duplicates(df)

    for key, value in duplicate_summary.items():
        print(f"{key}: {value}")

    # Questions
    print("\n[QUESTIONS]")

    question_summary = inspect_questions(df)

    for key, value in question_summary.items():
        print(f"{key}: {value}")

    # Duplicate answers
    print("\n[DUPLICATE ANSWER ANALYSIS]")

    duplicate_conflicts = (
        inspect_duplicate_answer_conflicts(df)
    )

    for key, value in duplicate_conflicts.items():
        print(f"{key}: {value}")

    # Labels
    print("\n[LABEL DISTRIBUTION]")
    print(inspect_labels(df))

    # Subjects
    print("\n[SUBJECT DISTRIBUTION]")
    print(inspect_subjects(df))

    # Difficulty
    print("\n[DIFFICULTY DISTRIBUTION]")
    print(inspect_difficulty(df))

    # Answer lengths
    print("\n[STUDENT ANSWER LENGTH]")
    print(inspect_answer_lengths(df))

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit a TechGrade AI CSV dataset."
    )

    parser.add_argument(
        "dataset",
        help="Path to the CSV dataset.",
    )

    args = parser.parse_args()

    run_audit(args.dataset)