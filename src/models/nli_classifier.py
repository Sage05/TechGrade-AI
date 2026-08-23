"""
TechGrade AI - Experiment 003

NLI features + Logistic Regression classifier.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.nli_baseline import NLIBaseline


FEATURE_COLUMNS = [
    "question_entailment",
    "question_contradiction",
    "question_neutral",
    "reference_entailment",
    "reference_contradiction",
    "reference_neutral",
]


class NLIClassifier:
    """NLI features + Logistic Regression."""

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/nli-deberta-v3-base"
        ),
        random_state: int = 42,
        max_iter: int = 1000,
    ) -> None:

        self.feature_extractor = NLIBaseline(
            model_name
        )

        self.classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        )

        self.is_fitted = False

    def _extract_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate NLI features in batches.

        Two NLI comparisons are made per example:

        Question -> Student
        Reference -> Student
        """

        questions = (
            df["question_text"]
            .astype(str)
            .tolist()
        )

        references = (
            df["model_answer"]
            .astype(str)
            .tolist()
        )

        students = (
            df["student_answer"]
            .astype(str)
            .tolist()
        )

        question_pairs = [
            (q, s)
            for q, s in zip(
                questions,
                students,
            )
        ]

        reference_pairs = [
            (r, s)
            for r, s in zip(
                references,
                students,
            )
        ]

        question_scores = (
            self.feature_extractor.predict(
                question_pairs
            )
        )

        reference_scores = (
            self.feature_extractor.predict(
                reference_pairs
            )
        )

        return pd.DataFrame(
            {
                "question_contradiction": (
                    question_scores[:, 0]
                ),
                "question_entailment": (
                    question_scores[:, 1]
                ),
                "question_neutral": (
                    question_scores[:, 2]
                ),
                "reference_contradiction": (
                    reference_scores[:, 0]
                ),
                "reference_entailment": (
                    reference_scores[:, 1]
                ),
                "reference_neutral": (
                    reference_scores[:, 2]
                ),
            }
        )

    def fit(
        self,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate NLI features and fit classifier."""

        X_train = self._extract_features(
            train_df
        )

        y_train = train_df["label"]

        self.classifier.fit(
            X_train[FEATURE_COLUMNS],
            y_train,
        )

        self.is_fitted = True

        return X_train

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate NLI features for new data."""

        if not self.is_fitted:
            raise RuntimeError(
                "Classifier must be fitted before "
                "transformation."
            )

        return self._extract_features(df)

    def predict(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate features and predictions."""

        if not self.is_fitted:
            raise RuntimeError(
                "Classifier must be fitted before "
                "prediction."
            )

        features = self.transform(df)

        predictions = self.classifier.predict(
            features[FEATURE_COLUMNS]
        )

        return features, pd.Series(
            predictions,
            index=df.index,
            name="predicted_label",
        )