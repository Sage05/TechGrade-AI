"""
TechGrade AI - Experiment 002

Sentence embeddings + Logistic Regression classifier.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.embedding_baseline import (
    EmbeddingBaseline,
)


FEATURE_COLUMNS = [
    "question_student",
    "reference_student",
    "question_reference_student",
]


class EmbeddingClassifier:
    """Semantic similarity features + Logistic Regression."""

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        random_state: int = 42,
        max_iter: int = 1000,
    ) -> None:

        self.feature_extractor = (
            EmbeddingBaseline(model_name)
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
        Generate semantic similarity features.

        Embeddings are generated in batches rather than loading
        each row individually.
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

        question_references = [
            f"{q} {r}"
            for q, r in zip(
                questions,
                references,
            )
        ]

        all_texts = (
            questions
            + references
            + students
            + question_references
        )

        embeddings = (
            self.feature_extractor.encode(
                all_texts
            )
        )

        n = len(df)

        question_embeddings = embeddings[
            :n
        ]

        reference_embeddings = embeddings[
            n:2 * n
        ]

        student_embeddings = embeddings[
            2 * n:3 * n
        ]

        question_reference_embeddings = (
            embeddings[3 * n:4 * n]
        )

        question_student = (
            question_embeddings
            * student_embeddings
        ).sum(axis=1)

        reference_student = (
            reference_embeddings
            * student_embeddings
        ).sum(axis=1)

        question_reference_student = (
            question_reference_embeddings
            * student_embeddings
        ).sum(axis=1)

        return pd.DataFrame(
            {
                "question_student": (
                    question_student
                ),
                "reference_student": (
                    reference_student
                ),
                "question_reference_student": (
                    question_reference_student
                ),
            }
        )

    def fit(
        self,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate training features and fit classifier."""

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
        """Generate features for new data."""

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