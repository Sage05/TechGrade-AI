"""
TechGrade AI - Concept-Aware Hybrid Grader.

Uses the 36-feature Exp 008 feature representation with
Logistic Regression for overall answer classification.

Also preserves concept-level predictions produced by the
FeatureExtractor so the application layer can reuse them
without running the ConceptMatcher twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.grading.feature_extractor import (
    FeatureExtractor,
    GradingFeatures,
)
from src.models.concept_matcher import ConceptPrediction


@dataclass
class GradingResult:
    """Complete result produced by the hybrid grader."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    concept_predictions: list[ConceptPrediction]


class HybridGrader:
    """
    Concept-aware hybrid answer grader.

    The classifier itself remains the same as Exp 008:
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )

    Feature extraction is handled by FeatureExtractor.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
    ) -> None:

        self.feature_extractor = feature_extractor

        self.classifier = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )

        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        labels: list[str] | np.ndarray,
    ) -> None:
        """
        Train the Logistic Regression classifier.
        """

        X = np.asarray(X, dtype=float)

        if X.size == 0:
            raise ValueError(
                "Training feature matrix is empty."
            )

        if len(labels) != len(X):
            raise ValueError(
                "Number of labels must match "
                "number of training samples."
            )

        self.classifier.fit(
            X,
            labels,
        )

        self.is_fitted = True

    def predict_features(
        self,
        features: np.ndarray,
    ) -> GradingResult:
        """
        Predict from a numerical feature vector.

        This method is retained for compatibility with the
        existing evaluation pipeline.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "HybridGrader must be fitted before prediction."
            )

        features = np.asarray(
            features,
            dtype=float,
        ).reshape(1, -1)

        prediction = self.classifier.predict(
            features
        )[0]

        probabilities = (
            self.classifier.predict_proba(
                features
            )[0]
        )

        probability_map = {
            str(label): float(probability)
            for label, probability in zip(
                self.classifier.classes_,
                probabilities,
            )
        }

        confidence = float(
            np.max(probabilities)
        )

        return GradingResult(
            label=str(prediction),
            confidence=confidence,
            probabilities=probability_map,
            concept_predictions=[],
        )

    def predict_features_with_details(
        self,
        features: GradingFeatures,
    ) -> GradingResult:
        """
        Predict from extracted features while preserving
        concept-level predictions.
        """

        result = self.predict_features(
            features.values
        )

        return GradingResult(
            label=result.label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            concept_predictions=(
                features.concept_predictions
            ),
        )

    def predict(
        self,
        question: str,
        reference_answer: str,
        student_answer: str,
        required_concepts: str | list[str] | None,
    ) -> GradingResult:
        """
        Extract features and produce a complete grading result.
        """

        features = self.feature_extractor.extract(
            question=question,
            reference_answer=reference_answer,
            student_answer=student_answer,
            required_concepts=required_concepts,
        )

        return self.predict_features_with_details(
            features
        )