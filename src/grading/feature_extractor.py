"""
TechGrade AI - Reusable feature extraction pipeline.

This module contains the complete 36-feature representation
used by Experiment 008.

Feature groups:
    1. TF-IDF similarity             (3)
    2. Semantic embeddings           (3)
    3. NLI                           (6)
    4. Answer quality                (12)
    5. Concept evidence              (12)

Human annotation fields such as:
    - label
    - missing_concepts
    - incorrect_concepts
    - human_score

are never used as predictive features.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from src.models.tfidf_baseline import TfidfBaseline
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.nli_baseline import NLIBaseline
from src.models.concept_matcher import ConceptMatcher , ConceptPrediction


@dataclass
class GradingFeatures:
    """Feature representation for a single student answer."""

    values: np.ndarray
    names: list[str]
    concept_predictions: list[ConceptPrediction]

    def as_dict(self) -> dict[str, float]:
        """Return features as a named dictionary."""

        return dict(zip(self.names, self.values))


class FeatureExtractor:
    """
    Extract the complete Exp 008 feature vector.

    Inputs:
        question
        reference answer
        required concepts
        student answer

    Output:
        36-dimensional numerical feature vector.
    """

    FEATURE_NAMES = [

        # ---------------------------------------------
        # Exp 006 - TF-IDF
        # ---------------------------------------------

        "tfidf_question_student",
        "tfidf_reference_student",
        "tfidf_question_reference_student",

        # ---------------------------------------------
        # Exp 006 - Embeddings
        # ---------------------------------------------

        "embedding_question_student",
        "embedding_reference_student",
        "embedding_question_reference_student",

        # ---------------------------------------------
        # Exp 006 - NLI
        # ---------------------------------------------

        "nli_question_entailment",
        "nli_question_contradiction",
        "nli_question_neutral",
        "nli_reference_entailment",
        "nli_reference_contradiction",
        "nli_reference_neutral",

        # ---------------------------------------------
        # Exp 007A - Answer quality
        # ---------------------------------------------

        "answer_word_count",
        "answer_character_count",
        "answer_sentence_count",
        "avg_words_per_sentence",
        "lexical_overlap_reference",
        "lexical_overlap_question",
        "reference_keyword_coverage",
        "negation_count",
        "uncertainty_count",
        "unique_token_ratio",
        "answer_reference_length_ratio",
        "answer_question_length_ratio",

        # ---------------------------------------------
        # Exp 008 - Concept evidence
        # ---------------------------------------------

        "concept_mean_similarity",
        "concept_max_similarity",
        "concept_min_similarity",
        "concept_mean_entailment",
        "concept_max_entailment",
        "concept_mean_contradiction",
        "concept_max_contradiction",
        "concept_mean_neutral",
        "concept_strong_entailment_fraction",
        "concept_strong_contradiction_fraction",
        "concept_evidence_spread",
        "required_concept_count",
    ]

    def __init__(
        self,
        tfidf_model: TfidfBaseline,
        embedding_model: EmbeddingBaseline,
        nli_model: NLIBaseline,
        concept_matcher: ConceptMatcher,
    ) -> None:

        self.tfidf_model = tfidf_model
        self.embedding_model = embedding_model
        self.nli_model = nli_model
        self.concept_matcher = concept_matcher

    @staticmethod
    def _parse_concepts(
        required_concepts: str | list[str] | None,
    ) -> list[str]:
        """
        Parse required_concepts using the same behavior
        as Experiment 008.
        """

        if required_concepts is None:
            return []

        if isinstance(required_concepts, list):
            return [
                str(x).strip()
                for x in required_concepts
                if str(x).strip()
            ]

        text = str(
            required_concepts
        ).strip()

        if not text:
            return []
        if "|" in text:
            return [
                concept.strip()
                for concept in text.split("|")
                if concept.strip()
    ]

        # Try JSON first.
        import json

        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                ]

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            pass

        # Try Python-list syntax.
        import ast

        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list):
                return [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                ]

        except (
            ValueError,
            SyntaxError,
        ):
            pass

        # Fallback: treat the value as one concept.
        return [text]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text using the Exp 008 token pattern."""

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

    def extract(
        self,
        question: str,
        reference_answer: str,
        student_answer: str,
        required_concepts: str | list[str],
    ) -> GradingFeatures:
        """
        Extract all 36 Exp 008 features for one answer.
        """

        question = str(question)
        reference_answer = str(reference_answer)
        student_answer = str(student_answer)

        # -------------------------------------------------
        # TF-IDF
        # -------------------------------------------------

        tfidf = self.tfidf_model.compare(
            question,
            reference_answer,
            student_answer,
        )

        # -------------------------------------------------
        # Embeddings
        # -------------------------------------------------

        embedding = self.embedding_model.compare(
            question,
            reference_answer,
            student_answer,
        )

        # -------------------------------------------------
        # NLI
        # -------------------------------------------------

        nli = self.nli_model.compare(
            question,
            reference_answer,
            student_answer,
        )

        # -------------------------------------------------
        # Answer quality
        # -------------------------------------------------

        student_tokens = self._tokenize(
            student_answer
        )

        reference_tokens = self._tokenize(
            reference_answer
        )

        question_tokens = self._tokenize(
            question
        )

        word_count = len(
            student_tokens
        )

        character_count = len(
            student_answer
        )

        sentence_count = max(
            1,
            len(
                re.findall(
                    r"[.!?]+",
                    student_answer,
                )
            ),
        )

        avg_words_per_sentence = (
            word_count / sentence_count
        )

        student_set = set(
            student_tokens
        )

        reference_set = set(
            reference_tokens
        )

        question_set = set(
            question_tokens
        )

        if student_set and reference_set:
            lexical_reference = (
                len(
                    student_set & reference_set
                )
                / len(
                    student_set | reference_set
                )
            )
        else:
            lexical_reference = 0.0

        if student_set and question_set:
            lexical_question = (
                len(
                    student_set & question_set
                )
                / len(
                    student_set | question_set
                )
            )
        else:
            lexical_question = 0.0

        reference_keywords = {
            token
            for token in reference_set
            if len(token) >= 4
        }

        if reference_keywords:
            reference_keyword_coverage = (
                len(
                    reference_keywords
                    & student_set
                )
                / len(reference_keywords)
            )
        else:
            reference_keyword_coverage = 0.0

        negation_words = {
            "not",
            "never",
            "no",
            "none",
            "cannot",
            "can't",
            "doesn't",
            "isn't",
            "aren't",
            "without",
        }

        negation_count = sum(
            token in negation_words
            for token in student_tokens
        )

        normalized_student = (
            student_answer.lower()
        )

        uncertainty_phrases = [
            "maybe",
            "perhaps",
            "probably",
            "possibly",
            "might",
            "i think",
            "i believe",
            "i don't know",
            "not sure",
            "unsure",
        ]

        uncertainty_count = sum(
            phrase in normalized_student
            for phrase in uncertainty_phrases
        )

        unique_token_ratio = (
            len(student_set)
            / max(
                1,
                word_count,
            )
        )

        answer_reference_length_ratio = (
            word_count
            / max(
                1,
                len(reference_tokens),
            )
        )

        answer_question_length_ratio = (
            word_count
            / max(
                1,
                len(question_tokens),
            )
        )

        # -------------------------------------------------
        # Automatic concept evidence
        # -------------------------------------------------

        concepts = self._parse_concepts(
            required_concepts
        )

        concept_predictions = (
            self.concept_matcher.predict(
                required_concepts=concepts,
                model_answer=reference_answer,
                student_answer=student_answer,
            )
        )

        similarities = [
            float(
                prediction.semantic_similarity
            )
            for prediction
            in concept_predictions
        ]

        entailments = [
            float(
                prediction.entailment
            )
            for prediction
            in concept_predictions
        ]

        contradictions = [
            float(
                prediction.contradiction
            )
            for prediction
            in concept_predictions
        ]

        neutrals = [
            float(
                prediction.neutral
            )
            for prediction
            in concept_predictions
        ]

        if similarities:

            mean_similarity = float(
                np.mean(similarities)
            )

            max_similarity = float(
                np.max(similarities)
            )

            min_similarity = float(
                np.min(similarities)
            )

            mean_entailment = float(
                np.mean(entailments)
            )

            max_entailment = float(
                np.max(entailments)
            )

            mean_contradiction = float(
                np.mean(contradictions)
            )

            max_contradiction = float(
                np.max(contradictions)
            )

            mean_neutral = float(
                np.mean(neutrals)
            )

            strong_entailment_fraction = float(
                np.mean(
                    np.asarray(entailments)
                    >= 0.60
                )
            )

            strong_contradiction_fraction = float(
                np.mean(
                    np.asarray(contradictions)
                    >= 0.75
                )
            )

            concept_spread = float(
                np.std(similarities)
            )

        else:

            mean_similarity = 0.0
            max_similarity = 0.0
            min_similarity = 0.0
            mean_entailment = 0.0
            max_entailment = 0.0
            mean_contradiction = 0.0
            max_contradiction = 0.0
            mean_neutral = 1.0
            strong_entailment_fraction = 0.0
            strong_contradiction_fraction = 0.0
            concept_spread = 0.0

        concept_count = len(
            concepts
        )

        # -------------------------------------------------
        # Final 36-feature vector
        # -------------------------------------------------

        values = np.array(
            [

                # TF-IDF
                float(
                    tfidf.question_student
                ),
                float(
                    tfidf.reference_student
                ),
                float(
                    tfidf.question_reference_student
                ),

                # Embeddings
                float(
                    embedding.question_student
                ),
                float(
                    embedding.reference_student
                ),
                float(
                    embedding.question_reference_student
                ),

                # NLI
                float(
                    nli.question_entailment
                ),
                float(
                    nli.question_contradiction
                ),
                float(
                    nli.question_neutral
                ),
                float(
                    nli.reference_entailment
                ),
                float(
                    nli.reference_contradiction
                ),
                float(
                    nli.reference_neutral
                ),

                # Answer quality
                float(word_count),
                float(character_count),
                float(sentence_count),
                float(avg_words_per_sentence),
                float(lexical_reference),
                float(lexical_question),
                float(reference_keyword_coverage),
                float(negation_count),
                float(uncertainty_count),
                float(unique_token_ratio),
                float(answer_reference_length_ratio),
                float(answer_question_length_ratio),

                # Concept evidence
                float(mean_similarity),
                float(max_similarity),
                float(min_similarity),
                float(mean_entailment),
                float(max_entailment),
                float(mean_contradiction),
                float(max_contradiction),
                float(mean_neutral),
                float(strong_entailment_fraction),
                float(strong_contradiction_fraction),
                float(concept_spread),
                float(concept_count),
            ],
            dtype=float,
        )

        if len(values) != len(
            self.FEATURE_NAMES
        ):
            raise RuntimeError(
                "Feature count mismatch: "
                f"generated {len(values)}, "
                f"expected {len(self.FEATURE_NAMES)}."
            )

        return GradingFeatures(
            values=values,
            names=self.FEATURE_NAMES.copy(),
            concept_predictions=concept_predictions,
        )