"""
TechGrade AI — Experiment 004
Concept Coverage Diagnostic

Purpose:
Determine whether human-annotated concept coverage is
associated with answer labels and human scores.

This is a diagnostic experiment, not the final model.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.models.concept_coverage import (
    add_concept_features,
)


def summarize(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("CONCEPT COVERAGE SUMMARY")
    print("=" * 70)

    print(
        f"\nRows analyzed: {len(df)}"
    )

    if "label" in df.columns:

        print("\n[BY LABEL]")

        label_summary = (
            df.groupby("label")
            .agg(
                count=("label", "size"),
                mean_coverage=(
                    "coverage_ratio",
                    "mean",
                ),
                mean_missing=(
                    "missing_ratio",
                    "mean",
                ),
                mean_incorrect=(
                    "incorrect_ratio",
                    "mean",
                ),
            )
            .sort_index()
        )

        print(
            label_summary.to_string(
                float_format=lambda x: (
                    f"{x:.3f}"
                )
            )
        )

    if "human_score" in df.columns:

        print("\n[BY HUMAN SCORE]")

        score_columns = [
            "human_score",
            "coverage_ratio",
            "missing_ratio",
            "incorrect_ratio",
        ]

        available = [
            column
            for column in score_columns
            if column in df.columns
        ]

        print(
            df[available]
            .corr(numeric_only=True)
            .to_string(
                float_format=lambda x: (
                    f"{x:.3f}"
                )
            )
        )

    print("\n[CONCEPT COUNTS]")

    print(
        df[
            [
                "required_count",
                "covered_count",
                "missing_count",
                "incorrect_count",
            ]
        ].describe().to_string()
    )


def inspect_examples(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("SAMPLE CONCEPT ANALYSIS")
    print("=" * 70)

    columns = [
        "question_text",
        "student_answer",
        "label",
        "human_score",
        "required_concepts",
        "missing_concepts",
        "incorrect_concepts",
        "required_count",
        "covered_count",
        "missing_count",
        "incorrect_count",
        "coverage_ratio",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    print(
        df[available]
        .head(10)
        .to_string(index=False)
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run the TechGrade AI concept "
            "coverage diagnostic."
        )
    )

    parser.add_argument(
        "dataset",
        help="Path to the revised CSV dataset.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TECHGRADE AI — EXPERIMENT 004")
    print("CONCEPT COVERAGE DIAGNOSTIC")
    print("=" * 70)

    print("\n[DATASET]")

    print(
        f"Path: {args.dataset}"
    )

    df = pd.read_csv(
        args.dataset
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Unique questions: "
        f"{df['question_text'].nunique()}"
    )

    required_columns = [
        "question_text",
        "student_answer",
        "label",
        "human_score",
        "required_concepts",
        "missing_concepts",
        "incorrect_concepts",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    print("\n[FEATURE EXTRACTION]")

    analyzed = add_concept_features(
        df
    )

    print(
        "Concept diagnostic features generated."
    )

    summarize(
        analyzed
    )

    inspect_examples(
        analyzed
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()