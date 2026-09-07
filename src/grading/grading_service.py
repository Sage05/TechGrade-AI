"""
TechGrade AI - Application grading service.

Public application-facing interface for grading a single
student answer.

The service combines:
    1. Overall HybridGrader prediction
    2. Application-level concept evidence evaluation
"""

from __future__ import annotations

from dataclasses import dataclass

from src.grading.concept_evidence import (
    ConceptEvidenceEvaluator,
    ConceptEvidenceResult,
)
from src.grading.hybrid_grader import HybridGrader


@dataclass
class AnswerEvaluation:
    """Complete application-level evaluation."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    concept_results: list[ConceptEvidenceResult]


class GradingService:
    """
    Public interface for TechGrade AI answer evaluation.

    The service delegates model inference to HybridGrader and
    applies the application-level concept evidence layer to
    the concept predictions already produced during feature
    extraction.
    """

    def __init__(
        self,
        grader: HybridGrader,
    ) -> None:
        self.grader = grader
        self.concept_evaluator = (
            ConceptEvidenceEvaluator()
        )

    def evaluate(
        self,
        question: str,
        reference_answer: str,
        required_concepts: str | list[str] | None,
        student_answer: str,
    ) -> AnswerEvaluation:
        """Evaluate a single student answer."""

        result = self.grader.predict(
            question=question,
            reference_answer=reference_answer,
            student_answer=student_answer,
            required_concepts=required_concepts,
        )

        concept_results = (
            self.concept_evaluator.evaluate_all(
                concept_predictions=result.concept_predictions,
                student_answer=student_answer,
            )
        )

        return AnswerEvaluation(
            label=result.label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            concept_results=concept_results,
        )