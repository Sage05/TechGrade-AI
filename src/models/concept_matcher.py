"""
TechGrade AI — Experiment 005.3
Concept-Specific Evidence Matching

For each required concept:

1. Locate concept-specific evidence in the model answer.
2. Locate concept-specific evidence in the student answer.
3. Compare the two evidence segments using DeBERTa NLI.
4. Use semantic relevance as a safeguard against false
   contradiction predictions.

Predictions:
    COVERED
    MISSING
    INCORRECT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)


@dataclass
class ConceptPrediction:
    """Prediction for one required concept."""

    concept: str

    reference_evidence: str
    student_evidence: str

    semantic_similarity: float

    entailment: float
    contradiction: float
    neutral: float

    decision: str


class ConceptMatcher:

    def __init__(
        self,
        embedding_model: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        nli_model: str = (
            "cross-encoder/"
            "nli-deberta-v3-base"
        ),
        relevance_threshold: float = 0.30,
        entailment_threshold: float = 0.60,
        contradiction_threshold: float = 0.75,
        context_window: int = 1,
    ) -> None:

        print(
            f"[MODEL] Embeddings: "
            f"{embedding_model}"
        )

        self.embedding_model = (
            SentenceTransformer(
                embedding_model
            )
        )

        print(
            f"[MODEL] NLI: "
            f"{nli_model}"
        )

        self.nli_model = CrossEncoder(
            nli_model
        )

        self.relevance_threshold = (
            relevance_threshold
        )

        self.entailment_threshold = (
            entailment_threshold
        )

        self.contradiction_threshold = (
            contradiction_threshold
        )

        self.context_window = (
            context_window
        )

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:

        text = str(text).strip()

        if not text:
            return []

        sentences = []

        current = ""

        for character in text:

            current += character

            if character in ".!?":

                sentence = current.strip()

                if sentence:
                    sentences.append(
                        sentence
                    )

                current = ""

        if current.strip():
            sentences.append(
                current.strip()
            )

        return sentences

    def _sentence_scores(
        self,
        sentences: list[str],
        concept: str,
    ) -> list[float]:

        if not sentences:
            return []

        concept_embedding = (
            self.embedding_model.encode(
                concept,
                normalize_embeddings=True,
            )
        )

        sentence_embeddings = (
            self.embedding_model.encode(
                sentences,
                normalize_embeddings=True,
            )
        )

        similarities = np.dot(
            sentence_embeddings,
            concept_embedding,
        )

        return [
            float(score)
            for score in similarities
        ]

    def _select_concept_evidence(
        self,
        text: str,
        concept: str,
    ) -> tuple[str, float]:

        sentences = (
            self._split_sentences(text)
        )

        if not sentences:
            return "", 0.0

        scores = self._sentence_scores(
            sentences,
            concept,
        )

        best_index = int(
            np.argmax(scores)
        )

        start = max(
            0,
            best_index - self.context_window,
        )

        end = min(
            len(sentences),
            best_index + self.context_window + 1,
        )

        evidence = " ".join(
            sentences[start:end]
        )

        return (
            evidence,
            scores[best_index],
        )

    def _nli_scores(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[float, float, float]:

        logits = self.nli_model.predict(
            [
                (
                    premise,
                    hypothesis,
                )
            ]
        )

        logits = np.asarray(
            logits,
            dtype=np.float32,
        )

        if logits.ndim == 2:
            logits = logits[0]

        probabilities = torch.softmax(
            torch.tensor(logits),
            dim=0,
        ).numpy()

        contradiction = float(
            probabilities[0]
        )

        entailment = float(
            probabilities[1]
        )

        neutral = float(
            probabilities[2]
        )

        return (
            entailment,
            contradiction,
            neutral,
        )

    def predict_concept(
        self,
        concept: str,
        model_answer: str,
        student_answer: str,
    ) -> ConceptPrediction:

        reference_evidence, _ = (
            self._select_concept_evidence(
                model_answer,
                concept,
            )
        )

        student_evidence, student_similarity = (
            self._select_concept_evidence(
                student_answer,
                concept,
            )
        )

        # No meaningful evidence in the student answer.
        if (
            not student_evidence
            or student_similarity
            < self.relevance_threshold
        ):

            return ConceptPrediction(
                concept=concept,
                reference_evidence=(
                    reference_evidence
                ),
                student_evidence=(
                    student_evidence
                ),
                semantic_similarity=(
                    student_similarity
                ),
                entailment=0.0,
                contradiction=0.0,
                neutral=1.0,
                decision="MISSING",
            )

        (
            entailment,
            contradiction,
            neutral,
        ) = self._nli_scores(
            student_evidence,
            reference_evidence,
        )

        # Strong support from the student evidence.
        if (
            entailment
            >= self.entailment_threshold
            and entailment > contradiction
        ):

            decision = "COVERED"

        # Contradiction is deliberately given a higher
        # threshold because false contradictions are much
        # more damaging than conservative MISSING predictions.
        elif (
            student_similarity
            >= self.relevance_threshold
            and contradiction
            >= self.contradiction_threshold
            and contradiction > entailment
        ):

            decision = "INCORRECT"

        else:

            decision = "MISSING"

        return ConceptPrediction(
            concept=concept,
            reference_evidence=(
                reference_evidence
            ),
            student_evidence=(
                student_evidence
            ),
            semantic_similarity=(
                student_similarity
            ),
            entailment=entailment,
            contradiction=contradiction,
            neutral=neutral,
            decision=decision,
        )

    def predict(
        self,
        required_concepts: list[str],
        model_answer: str,
        student_answer: str,
    ) -> list[ConceptPrediction]:

        predictions = []

        for concept in required_concepts:

            prediction = (
                self.predict_concept(
                    concept=concept,
                    model_answer=model_answer,
                    student_answer=student_answer,
                )
            )

            predictions.append(
                prediction
            )

        return predictions