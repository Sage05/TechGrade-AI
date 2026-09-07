"""
TechGrade AI - Application-level concept evidence evaluator.

This component supplements the research ConceptMatcher with
explicit terminology matching.

The original Exp 008 ConceptMatcher is intentionally left
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.models.concept_matcher import ConceptPrediction


@dataclass
class ConceptEvidenceResult:
    """Application-level evaluation of one required concept."""

    concept: str
    decision: str
    explicit_match: bool
    semantic_similarity: float
    entailment: float
    contradiction: float
    neutral: float
    reference_evidence: str
    student_evidence: str


class ConceptEvidenceEvaluator:
    """
    Combines explicit terminology evidence with the existing
    ConceptMatcher prediction.

    Decision policy:

    1. Explicit normalized terminology match → COVERED.
    2. Otherwise preserve a strong existing COVERED result.
    3. Otherwise preserve a strong existing INCORRECT result.
    4. Otherwise → MISSING.

    This is an application-level layer and does not modify
    the Exp 008 feature extractor or ConceptMatcher.
    """

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Normalize text for terminology matching.

        Handles:
        - capitalization
        - hyphens
        - repeated whitespace
        - punctuation
        """

        text = str(text).lower()

        text = text.replace("-", " ")

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @classmethod
    def _explicit_match(
        cls,
        concept: str,
        student_answer: str,
    ) -> bool:
        """Check whether the normalized concept appears explicitly."""

        normalized_concept = cls._normalize(
            concept
        )

        normalized_student = cls._normalize(
            student_answer
        )

        if not normalized_concept:
            return False

        return normalized_concept in normalized_student

    def evaluate(
        self,
        concept_prediction: ConceptPrediction,
        student_answer: str,
    ) -> ConceptEvidenceResult:
        """
        Produce an application-level concept decision.
        """

        explicit_match = self._explicit_match(
            concept_prediction.concept,
            student_answer,
        )

        if explicit_match:
            decision = "COVERED"

        elif concept_prediction.decision == "COVERED":
            decision = "COVERED"

        elif concept_prediction.decision == "INCORRECT":
            decision = "INCORRECT"

        else:
            decision = "MISSING"

        return ConceptEvidenceResult(
            concept=concept_prediction.concept,
            decision=decision,
            explicit_match=explicit_match,
            semantic_similarity=(
                concept_prediction.semantic_similarity
            ),
            entailment=concept_prediction.entailment,
            contradiction=concept_prediction.contradiction,
            neutral=concept_prediction.neutral,
            reference_evidence=(
                concept_prediction.reference_evidence
            ),
            student_evidence=(
                concept_prediction.student_evidence
            ),
        )

    def evaluate_all(
        self,
        concept_predictions: list[ConceptPrediction],
        student_answer: str,
    ) -> list[ConceptEvidenceResult]:
        """Evaluate all required concepts."""

        return [
            self.evaluate(
                concept_prediction=prediction,
                student_answer=student_answer,
            )
            for prediction in concept_predictions
        ]