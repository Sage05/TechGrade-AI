"""
TechGrade AI - Experiment 003
NLI feature extractor.

Uses a DeBERTa-v3 NLI cross-encoder to estimate:
- entailment
- contradiction
- neutral

for:
    Question -> Student Answer
    Reference Answer -> Student Answer
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder


@dataclass
class NLIResult:
    question_entailment: float
    question_contradiction: float
    question_neutral: float

    reference_entailment: float
    reference_contradiction: float
    reference_neutral: float


class NLIBaseline:
    """DeBERTa-v3 NLI feature extractor."""

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/nli-deberta-v3-base"
        ),
    ) -> None:

        self.model_name = model_name

        self.model = CrossEncoder(
            model_name
        )

    def predict(
        self,
        pairs: list[tuple[str, str]],
    ) -> np.ndarray:
        """
        Predict NLI probabilities for text pairs.

        Returns
        -------
        np.ndarray
            Array with shape:
            (number_of_pairs, 3)

            Columns:
            [contradiction, entailment, neutral]
        """

        if not pairs:
            raise ValueError(
                "Cannot predict an empty pair list."
            )

        probabilities = self.model.predict(
            pairs,
            apply_softmax=True,
            show_progress_bar=False,
        )

        return np.asarray(
            probabilities,
            dtype=float,
        )

    def compare(
        self,
        question: str,
        reference_answer: str,
        student_answer: str,
    ) -> NLIResult:
        """
        Calculate NLI probabilities for:

        Question -> Student
        Reference -> Student
        """

        pairs = [
            (
                question,
                student_answer,
            ),
            (
                reference_answer,
                student_answer,
            ),
        ]

        probabilities = self.predict(
            pairs
        )

        question_scores = probabilities[0]
        reference_scores = probabilities[1]

        return NLIResult(
            question_entailment=float(
                question_scores[1]
            ),
            question_contradiction=float(
                question_scores[0]
            ),
            question_neutral=float(
                question_scores[2]
            ),
            reference_entailment=float(
                reference_scores[1]
            ),
            reference_contradiction=float(
                reference_scores[0]
            ),
            reference_neutral=float(
                reference_scores[2]
            ),
        )