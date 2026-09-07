import pandas as pd

from src.grading.concept_evidence import ConceptEvidenceEvaluator
from src.models.concept_matcher import ConceptMatcher


DATASET = r"C:\Users\Ayuj Shingte\Downloads\revised_sample_os.csv"


def main() -> None:
    df = pd.read_csv(DATASET)
    row = df.iloc[0]

    matcher = ConceptMatcher()

    concepts = [
        "Mutual Exclusion",
        "Hold and Wait",
        "No Preemption",
        "Circular Wait",
    ]

    predictions = matcher.predict(
        required_concepts=concepts,
        model_answer=row["model_answer"],
        student_answer=row["student_answer"],
    )

    evaluator = ConceptEvidenceEvaluator()

    results = evaluator.evaluate_all(
        concept_predictions=predictions,
        student_answer=row["student_answer"],
    )

    print("\n" + "=" * 60)
    print("CONCEPT EVIDENCE TEST")
    print("=" * 60)

    for result in results:
        print(f"\nConcept: {result.concept}")
        print(f"Explicit Match: {result.explicit_match}")
        print(f"Decision:       {result.decision}")
        print(f"Similarity:     {result.semantic_similarity:.4f}")
        print(f"Entailment:     {result.entailment:.4f}")
        print(f"Contradiction:  {result.contradiction:.4f}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()