"""
TechGrade AI - Experiment 001
TF-IDF baseline for technical answer evaluation.

The vectorizer is fitted once on the training corpus and then reused
for validation/test data to avoid evaluation leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SimilarityResult:
    question_student: float
    reference_student: float
    question_reference_student: float


class TfidfBaseline:
    """TF-IDF similarity baseline."""

    def __init__(
        self,
        lowercase: bool = True,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=lowercase,
            ngram_range=ngram_range,
        )
        self.is_fitted = False

    def fit(self, texts: list[str]) -> None:
        """
        Fit TF-IDF vocabulary and IDF weights on training text only.
        """

        if not texts:
            raise ValueError("Training corpus is empty.")

        self.vectorizer.fit(texts)
        self.is_fitted = True

    def _transform(self, texts: list[str]):
        """Transform text using the fitted TF-IDF vectorizer."""

        if not self.is_fitted:
            raise RuntimeError(
                "TfidfBaseline must be fitted before transformation."
            )

        return self.vectorizer.transform(texts)

    @staticmethod
    def _similarity(left, right) -> float:
        """Calculate cosine similarity."""

        return float(
            cosine_similarity(left, right)[0][0]
        )

    def compare(
        self,
        question: str,
        reference_answer: str,
        student_answer: str,
    ) -> SimilarityResult:
        """
        Calculate three similarity signals using the
        already-fitted TF-IDF representation.

        Signals:
            1. Question <-> Student
            2. Reference <-> Student
            3. Question + Reference <-> Student
        """

        question_reference = (
            f"{question} {reference_answer}"
        )

        vectors = self._transform(
            [
                question,
                reference_answer,
                student_answer,
                question_reference,
            ]
        )

        question_vector = vectors[0]
        reference_vector = vectors[1]
        student_vector = vectors[2]
        question_reference_vector = vectors[3]

        return SimilarityResult(
            question_student=self._similarity(
                question_vector,
                student_vector,
            ),
            reference_student=self._similarity(
                reference_vector,
                student_vector,
            ),
            question_reference_student=self._similarity(
                question_reference_vector,
                student_vector,
            ),
        )