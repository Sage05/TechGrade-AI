"""
TechGrade AI - Experiment 012
Logistic Regression Convergence Diagnostic

Purpose:
    Determine whether the Exp 008 Logistic Regression convergence
    warning materially affects predictions.

Important:
    This experiment does NOT modify Exp 008.

    Exp 008 configuration:
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )

We compare the Exp 008 configuration against higher iteration
limits using the exact same features, split, and training data.

This is a diagnostic, not a new model experiment.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.grading.feature_extractor import FeatureExtractor
from src.models.concept_matcher import ConceptMatcher
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.nli_baseline import NLIBaseline
from src.models.tfidf_baseline import TfidfBaseline
from src.utils.data_split import (
    question_level_split,
    verify_no_question_leakage,
)


MAX_ITER_VALUES = [
    3000,
    5000,
    10000,
]


def fit_models(train: pd.DataFrame):

    tfidf = TfidfBaseline()

    corpus = []

    for _, row in train.iterrows():
        corpus.extend(
            [
                str(row["question_text"]),
                str(row["model_answer"]),
                str(row["student_answer"]),
            ]
        )

    tfidf.fit(corpus)

    embedding = EmbeddingBaseline()
    nli = NLIBaseline()
    concept_matcher = ConceptMatcher()

    extractor = FeatureExtractor(
        tfidf_model=tfidf,
        embedding_model=embedding,
        nli_model=nli,
        concept_matcher=concept_matcher,
    )

    return extractor


def extract_features(
    extractor: FeatureExtractor,
    dataframe: pd.DataFrame,
):

    features = []

    for _, row in dataframe.iterrows():

        result = extractor.extract(
            question=str(row["question_text"]),
            reference_answer=str(row["model_answer"]),
            student_answer=str(row["student_answer"]),
            required_concepts=row["required_concepts"],
        )

        features.append(result.values)

    return np.asarray(
        features,
        dtype=float,
    )


def train_classifier(
    X_train,
    y_train,
    max_iter,
):

    model = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=42,
    )

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        model.fit(
            X_train,
            y_train,
        )

    convergence_warning = any(
        issubclass(
            warning.category,
            ConvergenceWarning,
        )
        for warning in caught
    )

    return model, convergence_warning


def metrics(
    y_true,
    predictions,
):

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "TechGrade AI Experiment 012 "
            "— Logistic Regression convergence diagnostic"
        )
    )

    parser.add_argument(
        "dataset",
        help="Path to the revised CSV dataset.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TECHGRADE AI — EXPERIMENT 012")
    print("LOGISTIC REGRESSION CONVERGENCE DIAGNOSTIC")
    print("=" * 70)

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    df = pd.read_csv(
        args.dataset
    )

    print("\n[DATASET]")
    print(f"Rows: {len(df)}")
    print(
        f"Unique questions: "
        f"{df['question_text'].nunique()}"
    )

    # -------------------------------------------------------------
    # Split
    # -------------------------------------------------------------

    train, validation, test = question_level_split(df)

    verify_no_question_leakage(
        train,
        validation,
        test,
    )

    print("\n[SPLIT]")
    print(
        f"Train:      {len(train)} rows / "
        f"{train.question_text.nunique()} questions"
    )
    print(
        f"Validation: {len(validation)} rows / "
        f"{validation.question_text.nunique()} questions"
    )
    print(
        f"Test:       {len(test)} rows / "
        f"{test.question_text.nunique()} questions"
    )

    # -------------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------------

    print("\n[FEATURE EXTRACTION]")

    extractor = fit_models(
        train
    )

    X_train = extract_features(
        extractor,
        train,
    )

    X_validation = extract_features(
        extractor,
        validation,
    )

    X_test = extract_features(
        extractor,
        test,
    )

    print(
        f"Feature count: {X_train.shape[1]}"
    )

    # -------------------------------------------------------------
    # Train multiple iteration limits
    # -------------------------------------------------------------

    results = []

    predictions_by_model = {}

    print("\n[CONVERGENCE TEST]")

    for max_iter in MAX_ITER_VALUES:

        print(
            f"\nTesting max_iter={max_iter}..."
        )

        model, convergence_warning = train_classifier(
            X_train=X_train,
            y_train=train["label"].to_numpy(),
            max_iter=max_iter,
        )

        validation_predictions = model.predict(
            X_validation
        )

        test_predictions = model.predict(
            X_test
        )

        validation_metrics = metrics(
            validation["label"],
            validation_predictions,
        )

        test_metrics = metrics(
            test["label"],
            test_predictions,
        )

        predictions_by_model[max_iter] = test_predictions

        results.append(
            {
                "max_iter": max_iter,
                "convergence_warning": convergence_warning,
                "validation_accuracy": validation_metrics[
                    "accuracy"
                ],
                "validation_macro_f1": validation_metrics[
                    "macro_f1"
                ],
                "validation_weighted_f1": validation_metrics[
                    "weighted_f1"
                ],
                "test_accuracy": test_metrics[
                    "accuracy"
                ],
                "test_macro_f1": test_metrics[
                    "macro_f1"
                ],
                "test_weighted_f1": test_metrics[
                    "weighted_f1"
                ],
                "iterations_used": int(
                    np.max(model.n_iter_)
                ),
            }
        )

    # -------------------------------------------------------------
    # Print metrics
    # -------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # -------------------------------------------------------------
    # Prediction stability
    # -------------------------------------------------------------

    baseline_predictions = predictions_by_model[3000]

    print("\n")
    print("=" * 70)
    print("PREDICTION STABILITY")
    print("=" * 70)

    for max_iter in MAX_ITER_VALUES[1:]:

        predictions = predictions_by_model[
            max_iter
        ]

        changed = np.sum(
            predictions != baseline_predictions
        )

        total = len(
            baseline_predictions
        )

        print(
            f"3000 → {max_iter}: "
            f"{changed}/{total} predictions changed"
        )

    # -------------------------------------------------------------
    # Detailed changes
    # -------------------------------------------------------------

    for max_iter in MAX_ITER_VALUES[1:]:

        predictions = predictions_by_model[
            max_iter
        ]

        print("\n")
        print(
            f"CHANGES: 3000 → {max_iter}"
        )
        print("-" * 70)

        found_change = False

        for index in range(
            len(test)
        ):

            old = str(
                baseline_predictions[index]
            )

            new = str(
                predictions[index]
            )

            if old != new:

                found_change = True

                print(
                    f"\nQuestion: "
                    f"{test.iloc[index]['question_text']}"
                )

                print(
                    f"Actual:   "
                    f"{test.iloc[index]['label']}"
                )

                print(
                    f"3000:     {old}"
                )

                print(
                    f"{max_iter}:     {new}"
                )

        if not found_change:
            print(
                "No prediction changes."
            )

    # -------------------------------------------------------------
    # Final diagnostic conclusion
    # -------------------------------------------------------------

    baseline_row = results_df[
        results_df["max_iter"] == 3000
    ].iloc[0]

    highest_row = results_df.iloc[-1]

    print("\n")
    print("=" * 70)
    print("DIAGNOSTIC CONCLUSION")
    print("=" * 70)

    if bool(baseline_row["convergence_warning"]):

        print(
            "\nExp 008 produced a convergence warning."
        )

    else:

        print(
            "\nExp 008 did not produce a convergence warning."
        )

    if (
        baseline_row["test_macro_f1"]
        == highest_row["test_macro_f1"]
        and
        np.array_equal(
            predictions_by_model[3000],
            predictions_by_model[MAX_ITER_VALUES[-1]],
        )
    ):

        print(
            "\nHigher max_iter values produced the same "
            "test predictions and Macro F1."
        )

        print(
            "\nConclusion:"
            "\nThe convergence warning does not appear "
            "to materially affect the evaluated predictions."
        )

        print(
            "\nRecommendation:"
            "\nKeep the Exp 008 configuration unchanged."
        )

    else:

        print(
            "\nHigher max_iter values changed model behavior."
        )

        print(
            "\nConclusion:"
            "\nThe convergence warning may materially affect "
            "the model and requires a separate engineering/"
            "research decision."
        )

        print(
            "\nRecommendation:"
            "\nDo NOT modify Exp 008 yet."
            "\nInvestigate scaling/solver choices separately."
        )


if __name__ == "__main__":
    main()