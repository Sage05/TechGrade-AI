"""
TechGrade AI - Experiment 002
Semantic embedding baseline.

Uses Sentence Transformers to generate semantic representations
for questions, reference answers, and student answers.

The model is used as a feature extractor only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingSimilarityResult:
    question_student: float
    reference_student: float
    question_reference_student: float


class EmbeddingBaseline:
    """Sentence-transformer semantic similarity baseline."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Convert text into sentence embeddings."""

        if not texts:
            raise ValueError(
                "Cannot encode an empty text list."
            )

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    @staticmethod
    def similarity(
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity.

        Embeddings are normalized during encoding, so cosine
        similarity is simply their dot product.
        """

        return float(
            np.dot(left, right)
        )

    def compare(
        self,
        question: str,
        reference_answer: str,
        student_answer: str,
    ) -> EmbeddingSimilarityResult:
        """
        Generate three semantic similarity signals:

        1. Question <-> Student
        2. Reference <-> Student
        3. Question + Reference <-> Student
        """

        question_reference = (
            f"{question} {reference_answer}"
        )

        embeddings = self.encode(
            [
                question,
                reference_answer,
                student_answer,
                question_reference,
            ]
        )

        question_embedding = embeddings[0]
        reference_embedding = embeddings[1]
        student_embedding = embeddings[2]
        question_reference_embedding = embeddings[3]

        return EmbeddingSimilarityResult(
            question_student=self.similarity(
                question_embedding,
                student_embedding,
            ),
            reference_student=self.similarity(
                reference_embedding,
                student_embedding,
            ),
            question_reference_student=self.similarity(
                question_reference_embedding,
                student_embedding,
            ),
        )