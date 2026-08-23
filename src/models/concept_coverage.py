"""
TechGrade AI — Experiment 004
Concept Coverage Diagnostic

This module analyzes human-annotated concept information
from the revised dataset.

IMPORTANT:
This is a diagnostic experiment.

It does NOT train the final grading model and does NOT use
missing_concepts / incorrect_concepts as predictive features.

The purpose is to determine whether concept coverage,
missing concepts, and incorrect concepts correlate with
human grading outcomes.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import pandas as pd


CONCEPT_SEPARATOR = "|"


@dataclass
class ConceptAnalysis:
    required_count: int
    covered_count: int
    missing_count: int
    incorrect_count: int

    coverage_ratio: float
    missing_ratio: float
    incorrect_ratio: float


def parse_concepts(value: object) -> list[str]:
    """
    Convert a dataset concept field into a normalized list.

    Expected dataset format:
        Concept A|Concept B|Concept C

    Also supports Python-style lists in case the revised
    dataset later stores concepts as:
        ["Concept A", "Concept B"]
    """

    if value is None:
        return []

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # Support Python-style list representation.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list):
                return normalize_concepts(parsed)

        except (ValueError, SyntaxError):
            pass

    concepts = text.split(CONCEPT_SEPARATOR)

    return normalize_concepts(concepts)


def normalize_concept(value: object) -> str:
    """Normalize one concept without changing its meaning."""

    text = str(value).strip()

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_concepts(
    concepts: list[object],
) -> list[str]:
    """Normalize and deduplicate concept names."""

    normalized = []

    for concept in concepts:
        concept_name = normalize_concept(concept)

        if concept_name:
            normalized.append(concept_name)

    # Preserve original order while removing duplicates.
    return list(dict.fromkeys(normalized))


def analyze_concepts(
    required_concepts: object,
    covered_concepts: object,
    missing_concepts: object,
    incorrect_concepts: object,
) -> ConceptAnalysis:
    """
    Calculate concept-level diagnostic statistics.

    Parameters
    ----------
    required_concepts:
        Concepts required for a satisfactory answer.

    covered_concepts:
        Concepts correctly represented by the student.

    missing_concepts:
        Required concepts absent from the student answer.

    incorrect_concepts:
        Concepts represented incorrectly by the student.
    """

    required = parse_concepts(
        required_concepts
    )

    covered = parse_concepts(
        covered_concepts
    )

    missing = parse_concepts(
        missing_concepts
    )

    incorrect = parse_concepts(
        incorrect_concepts
    )

    required_set = set(required)
    covered_set = set(covered)
    missing_set = set(missing)
    incorrect_set = set(incorrect)

    # Only concepts belonging to the required concept
    # vocabulary count toward coverage.
    covered_required = (
        covered_set & required_set
    )

    missing_required = (
        missing_set & required_set
    )

    incorrect_required = (
        incorrect_set & required_set
    )

    required_count = len(required_set)

    covered_count = len(
        covered_required
    )

    missing_count = len(
        missing_required
    )

    incorrect_count = len(
        incorrect_required
    )

    if required_count > 0:
        coverage_ratio = (
            covered_count / required_count
        )

        missing_ratio = (
            missing_count / required_count
        )

        incorrect_ratio = (
            incorrect_count / required_count
        )

    else:
        coverage_ratio = 0.0
        missing_ratio = 0.0
        incorrect_ratio = 0.0

    return ConceptAnalysis(
        required_count=required_count,
        covered_count=covered_count,
        missing_count=missing_count,
        incorrect_count=incorrect_count,
        coverage_ratio=coverage_ratio,
        missing_ratio=missing_ratio,
        incorrect_ratio=incorrect_ratio,
    )


def add_concept_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add concept diagnostic features to a dataframe.

    NOTE:
    The revised dataset does not currently provide a dedicated
    student_concepts column in the schema.

    Therefore, this function uses the annotated
    missing_concepts and incorrect_concepts fields to infer
    diagnostic coverage.

    This is appropriate for dataset analysis only.

    It MUST NOT be used as-is as predictive input to the
    final grading model because these annotations are generated
    after evaluating the student answer.
    """

    required_values = (
        df["required_concepts"]
        if "required_concepts" in df.columns
        else pd.Series(
            [[] for _ in range(len(df))],
            index=df.index,
        )
    )

    missing_values = (
        df["missing_concepts"]
        if "missing_concepts" in df.columns
        else pd.Series(
            [[] for _ in range(len(df))],
            index=df.index,
        )
    )

    incorrect_values = (
        df["incorrect_concepts"]
        if "incorrect_concepts" in df.columns
        else pd.Series(
            [[] for _ in range(len(df))],
            index=df.index,
        )
    )

    results = []

    for required, missing, incorrect in zip(
        required_values,
        missing_values,
        incorrect_values,
    ):
        required_list = parse_concepts(
            required
        )

        missing_list = parse_concepts(
            missing
        )

        incorrect_list = parse_concepts(
            incorrect
        )

        required_set = set(
            required_list
        )

        missing_set = set(
            missing_list
        )

        incorrect_set = set(
            incorrect_list
        )

        required_count = len(
            required_set
        )

        missing_count = len(
            missing_set & required_set
        )

        incorrect_count = len(
            incorrect_set & required_set
        )

        # Diagnostic approximation:
        #
        # required concepts that are neither missing
        # nor incorrectly represented are treated as
        # correctly covered.
                # A concept is considered correctly covered only when it
        # is neither missing nor incorrectly represented.
        covered_set = (
            required_set
            - missing_set
            - incorrect_set
        )

        covered_count = len(covered_set)

        # Diagnostic consistency check:
        # missing and incorrect concepts should normally be
        # mutually exclusive.
        overlap_count = len(
            missing_set & incorrect_set
        )

        if required_count > 0:
            coverage_ratio = (
                covered_count / required_count
            )

            missing_ratio = (
                missing_count / required_count
            )

            incorrect_ratio = (
                incorrect_count / required_count
            )
        else:
            coverage_ratio = 0.0
            missing_ratio = 0.0
            incorrect_ratio = 0.0

        if required_count > 0:
            coverage_ratio = (
                covered_count / required_count
            )

            missing_ratio = (
                missing_count / required_count
            )

            incorrect_ratio = (
                incorrect_count / required_count
            )
        else:
            coverage_ratio = 0.0
            missing_ratio = 0.0
            incorrect_ratio = 0.0

        results.append(
            {
                "required_count": (
                    required_count
                ),
                "covered_count": (
                    covered_count
                ),
                "missing_count": (
                    missing_count
                ),
                "incorrect_count": (
                    incorrect_count
                ),
                "annotation_overlap_count": (
                    overlap_count
                ),
                "coverage_ratio": (
                    coverage_ratio
                ),
                "missing_ratio": (
                    missing_ratio
                ),
                "incorrect_ratio": (
                    incorrect_ratio
                ),
            }
        )

    feature_df = pd.DataFrame(
        results,
        index=df.index,
    )

    return pd.concat(
        [
            df.reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )