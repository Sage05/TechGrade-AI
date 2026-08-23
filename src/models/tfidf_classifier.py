"""
TechGrade AI - Experiment 001
TF-IDF feature-based classification baseline.

The classifier uses three lexical similarity features:

1. Question <-> Student Answer
2. Reference Answer <-> Student Answer
3. Question + Reference Answer <-> Student Answer

These features are then passed to a simple Logistic Regression classifier
to predict the five answer-quality labels.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.tfidf_baseline import TfidfBaseline


FEATURE_COLUMNS = [
    "question_student",
    "reference_student",
    "question_reference_student",
]


class TfidfClassifier:
    """TF-IDF similarity features + Logistic Regression."""

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 1000,
    ) -> None:

        self.feature_extractor = TfidfBaseline()

        self.classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        )

        self.is_fitted = False

    def _build_training_corpus(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Build the text corpus used to fit TF-IDF.

        Only training data should ever be passed here.
        """

        return (
            df["question_text"].astype(str).tolist()
            + df["model_answer"].astype(str).tolist()
            + df["student_answer"].astype(str).tolist()
        )

    def _extract_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate the three similarity features."""

        rows = []

        for _, row in df.iterrows():

            result = self.feature_extractor.compare(
                question=str(row["question_text"]),
                reference_answer=str(row["model_answer"]),
                student_answer=str(row["student_answer"]),
            )

            rows.append(
                {
                    "question_student": result.question_student,
                    "reference_student": result.reference_student,
                    "question_reference_student": (
                        result.question_reference_student
                    ),
                }
            )

        return pd.DataFrame(rows)

    def fit(
        self,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit TF-IDF and Logistic Regression on training data.

        Returns
        -------
        pd.DataFrame
            Training feature matrix.
        """

        corpus = self._build_training_corpus(
            train_df
        )

        self.feature_extractor.fit(corpus)

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
        """Generate features for validation/test data."""

        if not self.is_fitted:
            raise RuntimeError(
                "Classifier must be fitted before transformation."
            )

        return self._extract_features(df)

    def predict(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Generate features and predictions.

        Returns
        -------
        features:
            Similarity feature dataframe.

        predictions:
            Predicted labels.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Classifier must be fitted before prediction."
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