"""
TechGrade AI - Question-Level Data Splitting

Ensures that all answer variants belonging to the same question
remain in the same dataset split.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def question_level_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset by unique question.

    No question will appear in more than one split.

    Returns
    -------
    train_df, validation_df, test_df
    """

    if "question_text" not in df.columns:
        raise ValueError(
            "Dataset must contain 'question_text'."
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 < validation_size < 1:
        raise ValueError(
            "validation_size must be between 0 and 1."
        )

    if test_size + validation_size >= 1:
        raise ValueError(
            "test_size + validation_size must be less than 1."
        )

    questions = df["question_text"].drop_duplicates()

    train_questions, test_questions = train_test_split(
        questions,
        test_size=test_size,
        random_state=random_state,
    )

    relative_validation_size = (
        validation_size / (1 - test_size)
    )

    train_questions, validation_questions = train_test_split(
        train_questions,
        test_size=relative_validation_size,
        random_state=random_state,
    )

    train_set = set(train_questions)
    validation_set = set(validation_questions)
    test_set = set(test_questions)

    train_df = df[
        df["question_text"].isin(train_set)
    ].copy()

    validation_df = df[
        df["question_text"].isin(validation_set)
    ].copy()

    test_df = df[
        df["question_text"].isin(test_set)
    ].copy()

    # Preserve original row indices for traceability.
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, validation_df, test_df


def verify_no_question_leakage(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    """
    Verify that no question occurs across multiple splits.
    """

    train_questions = set(
        train_df["question_text"]
    )

    validation_questions = set(
        validation_df["question_text"]
    )

    test_questions = set(
        test_df["question_text"]
    )

    train_validation_overlap = (
        train_questions & validation_questions
    )

    train_test_overlap = (
        train_questions & test_questions
    )

    validation_test_overlap = (
        validation_questions & test_questions
    )

    if (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
    ):
        return False

    return True