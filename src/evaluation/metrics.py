"""
TechGrade AI - Evaluation Metrics

Shared evaluation utilities for experiments.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)


def classification_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
) -> dict:
    """Calculate classification metrics."""

    y_true = list(y_true)
    y_pred = list(y_pred)

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def regression_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> dict:
    """Calculate basic regression metrics."""

    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)

    return {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
    }