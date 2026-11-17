"""
TechGrade AI — Experiment 015
Learned Regression Scorer

Objective
---------
Predict numerical human scores (0–10) directly from the existing
36-feature representation and compare Ridge Regression with the
deterministic baselines evaluated in Experiment 014.

Methods compared
----------------
1. Global training-score mean.
2. Predicted-category training-score mean.
3. Probability-weighted categorical scorer.
4. Ridge Regression.

Evaluation protocol
-------------------
- Split data by question.
- Fit TF-IDF on training data only.
- Train models using the training split.
- Select Ridge alpha using validation MAE.
- Evaluate using the held-out test split.
- Never use test scores for model fitting or selection.
- Save metrics, tuning results, predictions, plots, and a report.

This is an exploratory experiment. The datasets contain relatively
few unique questions, so all results must be interpreted cautiously.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.data_split import (
    question_level_split,
    verify_no_question_leakage,
)

from src.models.tfidf_baseline import TfidfBaseline
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.nli_baseline import NLIBaseline
from src.models.concept_matcher import ConceptMatcher
from src.grading.feature_extractor import FeatureExtractor


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_OS_CSV = Path(
    Path.home() / "Downloads" / "revised_sample_os.csv"
)

DEFAULT_DBMS_CSV = Path(
    Path.home() / "Downloads" / "revised_sample_dbms.csv"
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

LABELS = [
    "Correct",
    "Partially Correct",
    "Incorrect",
    "Irrelevant",
    "Contradictory",
]

REQUIRED_COLUMNS = [
    "question_text",
    "model_answer",
    "student_answer",
    "required_concepts",
    "label",
    "human_score",
]

# Candidate regularization strengths.
# Select the best alpha using validation MAE only.
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]

EXPECTED_FEATURE_COUNT = 36


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Validate required fields without editing the source CSV."""

    df = df.copy()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: missing required columns: "
            f"{missing_columns}"
        )

    # Preserve the original CSV data-row number for traceability.
    df["_source_csv_line"] = np.arange(
        2,
        len(df) + 2,
    )

    # Normalize surrounding whitespace in labels.
    df["label"] = df["label"].astype(str).str.strip()

    unknown_labels = sorted(
        set(df["label"].unique()) - set(LABELS)
    )

    if unknown_labels:
        raise ValueError(
            f"{dataset_name}: unexpected labels: {unknown_labels}"
        )

    # Validate numerical targets. Never silently invent missing scores.
    scores = pd.to_numeric(
        df["human_score"],
        errors="coerce",
    )

    if scores.isna().any():
        raise ValueError(
            f"{dataset_name}: missing or non-numeric human scores found."
        )

    if ((scores < 0) | (scores > 10)).any():
        raise ValueError(
            f"{dataset_name}: scores must lie between 0 and 10."
        )

    df["human_score"] = scores.astype(float)

    # Required text columns must contain actual content.
    for column in [
        "question_text",
        "model_answer",
        "student_answer",
    ]:
        invalid = (
            df[column].isna()
            | df[column].astype(str).str.strip().eq("")
        )

        if invalid.any():
            raise ValueError(
                f"{dataset_name}: empty values found in '{column}'."
            )

    if df["question_text"].nunique() < 3:
        raise ValueError(
            f"{dataset_name}: insufficient unique questions."
        )

    return df


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def build_features(
    df: pd.DataFrame,
    extractor: FeatureExtractor,
) -> np.ndarray:
    """Extract the existing Exp 008 feature representation."""

    feature_rows = []

    for _, row in df.iterrows():

        concepts = (
            ""
            if pd.isna(row["required_concepts"])
            else str(row["required_concepts"])
        )

        result = extractor.extract(
            question=str(row["question_text"]),
            reference_answer=str(row["model_answer"]),
            student_answer=str(row["student_answer"]),
            required_concepts=concepts,
        )

        feature_rows.append(
            np.asarray(
                result.values,
                dtype=np.float32,
            )
        )

    if not feature_rows:
        raise ValueError(
            "Cannot extract features from an empty split."
        )

    matrix = np.vstack(feature_rows)

    if matrix.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"but extracted {matrix.shape[1]}. "
            "Check the current Exp 008 implementation."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    return matrix


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """Calculate numerical grading metrics."""

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.clip(
        np.asarray(predicted, dtype=float),
        0.0,
        10.0,
    )

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )
    )

    if len(actual) >= 2 and np.unique(actual).size > 1:
        r2 = float(
            r2_score(
                actual,
                predicted,
            )
        )
    else:
        r2 = np.nan

    actual_series = pd.Series(actual)
    predicted_series = pd.Series(predicted)

    if (
        actual_series.nunique() > 1
        and predicted_series.nunique() > 1
    ):
        spearman = float(
            actual_series.corr(
                predicted_series,
                method="spearman",
            )
        )
    else:
        spearman = np.nan

    return {
        "test_rows": int(len(actual)),
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": r2,
        "Spearman": spearman,
        "mean_prediction_error": float(
            np.mean(predicted - actual)
        ),
    }


# ============================================================
# BASELINE SCORERS
# ============================================================

def fit_categorical_baselines(
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    X_test: np.ndarray,
    test_row_count: int,
) -> tuple[dict, dict]:
    """
    Fit the categorical classifier and calculate the three
    deterministic scoring baselines.

    All score mappings are calculated from training data only.
    """

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )

    classifier.fit(
        X_train,
        train_df["label"],
    )

    classes = classifier.classes_

    # This mapping must never use validation or test scores.
    label_means = (
        train_df.groupby("label")["human_score"]
        .mean()
        .to_dict()
    )

    missing_training_means = [
        label
        for label in classes
        if label not in label_means
    ]

    if missing_training_means:
        raise RuntimeError(
            "Training scores are unavailable for classifier classes: "
            f"{missing_training_means}"
        )

    global_mean = float(
        train_df["human_score"].mean()
    )

    probabilities = classifier.predict_proba(
        X_test
    )

    predicted_labels = classes[
        probabilities.argmax(axis=1)
    ]

    class_score_vector = np.asarray(
        [
            label_means[label]
            for label in classes
        ],
        dtype=float,
    )

    # Baseline 1:
    # Every answer receives the training-set global score mean.
    global_mean_predictions = np.full(
        test_row_count,
        global_mean,
        dtype=float,
    )

    # Baseline 2:
    # Use the mean training score for the most likely category.
    predicted_category_predictions = np.asarray(
        [
            label_means[label]
            for label in predicted_labels
        ],
        dtype=float,
    )

    # Baseline 3:
    # Weight category means by the classifier's predicted probabilities.
    probability_weighted_predictions = (
        probabilities @ class_score_vector
    )

    predictions = {
        "Global mean baseline": global_mean_predictions,
        "Predicted-category mean": predicted_category_predictions,
        "Probability-weighted scorer": (
            probability_weighted_predictions
        ),
    }

    details = {
        "classifier": classifier,
        "classes": classes,
        "label_means": label_means,
        "predicted_labels": predicted_labels,
        "probabilities": probabilities,
        "global_mean": global_mean,
    }

    return predictions, details


# ============================================================
# REPORT FORMATTING
# ============================================================

def markdown_table(
    df: pd.DataFrame,
) -> str:
    """Format a DataFrame as Markdown without tabulate."""

    if df.empty:
        return "_No rows available._"

    headers = [
        str(column)
        for column in df.columns
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            ["---"] * len(headers)
        ) + " |",
    ]

    for _, row in df.iterrows():

        values = []

        for value in row.tolist():

            if pd.isna(value):
                formatted = "NA"

            elif isinstance(
                value,
                (float, np.floating),
            ):
                formatted = f"{value:.4f}"

            else:
                formatted = str(value)

            formatted = (
                formatted
                .replace("|", "\\|")
                .replace("\n", " ")
            )

            values.append(formatted)

        lines.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(lines)


# ============================================================
# RUN ONE DATASET
# ============================================================

def run_dataset(
    csv_path: Path,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    print("\n" + "=" * 72)
    print(f"EXPERIMENT 015 — {dataset_name}")
    print("LEARNED REGRESSION SCORER")
    print("=" * 72)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    df = validate_dataset(
        pd.read_csv(csv_path),
        dataset_name,
    )

    print("\n[DATASET]")
    print(f"Rows: {len(df)}")
    print(
        f"Unique questions: "
        f"{df['question_text'].nunique()}"
    )

    # --------------------------------------------------------
    # Question-level split
    # --------------------------------------------------------

    train_df, validation_df, test_df = question_level_split(
        df,
        random_state=42,
    )

    if (
        train_df.empty
        or validation_df.empty
        or test_df.empty
    ):
        raise RuntimeError(
            f"{dataset_name}: one or more splits are empty."
        )

    no_leakage = verify_no_question_leakage(
        train_df,
        validation_df,
        test_df,
    )

    if not no_leakage:
        raise RuntimeError(
            f"{dataset_name}: question leakage detected."
        )

    print("\n[QUESTION-LEVEL SPLIT]")

    for split_name, split_df in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:
        print(
            f"{split_name:12s}: "
            f"{len(split_df):3d} rows / "
            f"{split_df['question_text'].nunique():2d} questions"
        )

    print("Question leakage: NONE")

    # --------------------------------------------------------
    # Initialize existing Exp 008 components
    # --------------------------------------------------------

    print("\n[INITIALIZING EXP 008 COMPONENTS]")

    tfidf_model = TfidfBaseline()

    training_corpus = (
        train_df["question_text"].astype(str).tolist()
        + train_df["model_answer"].astype(str).tolist()
        + train_df["student_answer"].astype(str).tolist()
    )

    # Fit the learned TF-IDF vocabulary on training text only.
    tfidf_model.fit(
        training_corpus
    )

    embedding_model = EmbeddingBaseline()
    nli_model = NLIBaseline()
    concept_matcher = ConceptMatcher()

    extractor = FeatureExtractor(
        tfidf_model=tfidf_model,
        embedding_model=embedding_model,
        nli_model=nli_model,
        concept_matcher=concept_matcher,
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print("\n[FEATURE EXTRACTION]")

    X_train = build_features(
        train_df,
        extractor,
    )

    X_validation = build_features(
        validation_df,
        extractor,
    )

    X_test = build_features(
        test_df,
        extractor,
    )

    print(
        f"Training matrix:   {X_train.shape}"
    )

    print(
        f"Validation matrix: {X_validation.shape}"
    )

    print(
        f"Test matrix:       {X_test.shape}"
    )

    y_train = train_df[
        "human_score"
    ].to_numpy(dtype=float)

    y_validation = validation_df[
        "human_score"
    ].to_numpy(dtype=float)

    y_test = test_df[
        "human_score"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Ridge regression model selection
    # --------------------------------------------------------

    print("\n[RIDGE REGRESSION — VALIDATION TUNING]")

    tuning_rows = []
    ridge_models = {}

    for alpha in RIDGE_ALPHAS:

        model = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "regressor",
                    Ridge(alpha=alpha),
                ),
            ]
        )

        # Fit only on training examples.
        model.fit(
            X_train,
            y_train,
        )

        validation_predictions = np.clip(
            model.predict(X_validation),
            0.0,
            10.0,
        )

        validation_mae = mean_absolute_error(
            y_validation,
            validation_predictions,
        )

        validation_rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_validation,
                    validation_predictions,
                )
            )
        )

        tuning_rows.append(
            {
                "dataset": dataset_name,
                "alpha": alpha,
                "validation_rows": len(validation_df),
                "validation_MAE": float(validation_mae),
                "validation_RMSE": validation_rmse,
            }
        )

        ridge_models[alpha] = model

        print(
            f"alpha={alpha:6.1f} | "
            f"Validation MAE={validation_mae:.4f} | "
            f"Validation RMSE={validation_rmse:.4f}"
        )

    tuning_df = pd.DataFrame(
        tuning_rows
    )

    # Select alpha based on validation MAE only.
    best_alpha = float(
        tuning_df.sort_values(
            ["validation_MAE", "alpha"],
            ascending=[True, True],
        ).iloc[0]["alpha"]
    )

    regression_model = ridge_models[
        best_alpha
    ]

    print(
        f"\nSelected alpha: {best_alpha}"
    )

    print(
        "Selection criterion: lowest validation MAE"
    )

    # --------------------------------------------------------
    # Fit and evaluate categorical baselines
    # --------------------------------------------------------

    print("\n[DETERMINISTIC BASELINES]")

    baseline_predictions, baseline_details = (
        fit_categorical_baselines(
            train_df=train_df,
            X_train=X_train,
            X_test=X_test,
            test_row_count=len(test_df),
        )
    )

    print(
        "Training-set global score mean: "
        f"{baseline_details['global_mean']:.4f}"
    )

    print("\nTraining-set score means by category:")

    for label in LABELS:
        mean_value = baseline_details[
            "label_means"
        ].get(label, np.nan)

        label_count = int(
            (train_df["label"] == label).sum()
        )

        print(
            f"  {label:20s} "
            f"{mean_value:8.4f} "
            f"(n={label_count})"
        )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    regression_predictions = np.clip(
        regression_model.predict(X_test),
        0.0,
        10.0,
    )

    all_predictions = {
        **baseline_predictions,
        "Ridge regression": regression_predictions,
    }

    metric_rows = []

    for method_name, predictions in all_predictions.items():

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        metric_rows.append(
            {
                "dataset": dataset_name,
                "method": method_name,
                "selected_ridge_alpha": (
                    best_alpha
                    if method_name == "Ridge regression"
                    else np.nan
                ),
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    print("\n[HELD-OUT TEST METRICS]")

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # Save row-level predictions
    # --------------------------------------------------------

    probabilities = baseline_details[
        "probabilities"
    ]

    classes = baseline_details[
        "classes"
    ]

    predicted_labels = baseline_details[
        "predicted_labels"
    ]

    predictions_df = pd.DataFrame(
        {
            "source_csv_line": test_df[
                "_source_csv_line"
            ].to_numpy(),

            "question_text": test_df[
                "question_text"
            ].to_numpy(),

            "student_answer": test_df[
                "student_answer"
            ].to_numpy(),

            "actual_label": test_df[
                "label"
            ].to_numpy(),

            "predicted_label": predicted_labels,

            "actual_human_score": y_test,

            "global_mean_baseline": baseline_predictions[
                "Global mean baseline"
            ],

            "predicted_category_mean": baseline_predictions[
                "Predicted-category mean"
            ],

            "probability_weighted_score": baseline_predictions[
                "Probability-weighted scorer"
            ],

            "ridge_regression_score": regression_predictions,

            "ridge_absolute_error": np.abs(
                regression_predictions - y_test
            ),
        }
    )

    for index, label in enumerate(classes):

        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        predictions_df[
            f"probability_{safe_label}"
        ] = probabilities[:, index]

    prediction_path = (
        RESULTS_DIR
        / f"exp015_predictions_{dataset_name.lower()}.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False,
    )

    tuning_path = (
        RESULTS_DIR
        / f"exp015_validation_tuning_{dataset_name.lower()}.csv"
    )

    tuning_df.to_csv(
        tuning_path,
        index=False,
    )

    print(
        f"\nPredictions saved: {prediction_path}"
    )

    print(
        f"Validation tuning saved: {tuning_path}"
    )

    # --------------------------------------------------------
    # Save actual vs predicted plot
    # --------------------------------------------------------

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(9, 7)
        )

        for method_name, predictions in all_predictions.items():

            ax.scatter(
                y_test,
                np.clip(predictions, 0.0, 10.0),
                label=method_name,
                alpha=0.75,
            )

        ax.plot(
            [0, 10],
            [0, 10],
            linestyle="--",
            label="Perfect prediction",
        )

        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)

        ax.set_xlabel(
            "Human-assigned score"
        )

        ax.set_ylabel(
            "Predicted score"
        )

        ax.set_title(
            "Exp 015 — Actual vs Predicted Scores "
            f"({dataset_name})"
        )

        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()

        plot_path = (
            RESULTS_DIR
            / f"exp015_actual_vs_predicted_{dataset_name.lower()}.png"
        )

        fig.savefig(
            plot_path,
            dpi=160,
        )

        plt.close(fig)

        print(
            f"Plot saved: {plot_path}"
        )

    except ImportError:
        print(
            "WARNING: matplotlib unavailable; plot skipped."
        )

    return metrics_df, tuning_df, predictions_df


# ============================================================
# REPORT
# ============================================================

def write_report(
    metrics_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
) -> None:
    """Save methodology, validation tuning, and test results."""

    lines = [
        "# TechGrade AI — Experiment 015",
        "",
        "## Learned Regression Scorer",
        "",
        "### Objective",
        "",
        "Evaluate whether direct regression from the existing "
        "36-feature representation predicts human-assigned scores "
        "more accurately than deterministic categorical baselines.",
        "",
        "### Models compared",
        "",
        "1. Global training-score mean.",
        "2. Predicted-category training-score mean.",
        "3. Probability-weighted categorical scorer.",
        "4. StandardScaler followed by Ridge Regression.",
        "",
        "### Validation protocol",
        "",
        "Ridge alpha is selected using validation MAE. Candidate values "
        f"are {RIDGE_ALPHAS}. The held-out test set is not used to "
        "select alpha.",
        "",
        "### Validation tuning results",
        "",
        markdown_table(tuning_df),
        "",
        "### Held-out test results",
        "",
        markdown_table(metrics_df),
        "",
        "### Metric interpretation",
        "",
        "- Lower MAE means smaller average absolute score error.",
        "- Lower RMSE means smaller squared-error penalty on large mistakes.",
        "- Higher R-squared indicates better performance relative to "
        "predicting the test-set mean.",
        "- Spearman measures rank association, not numerical agreement.",
        "- Negative R-squared is possible.",
        "- Numerical predictions are clipped to the 0–10 scoring scale.",
        "",
        "### Limitations",
        "",
        "- Only a small number of unique questions are available.",
        "- Validation contains few questions, so selected alpha may be unstable.",
        "- Results use one question-level split per subject.",
        "- The predicted category probabilities may not be calibrated.",
        "- Human scores may contain annotation noise or evaluator subjectivity.",
        "- Results are exploratory and do not establish production readiness.",
        "",
        "### Next step",
        "",
        "Compare Exp 014 and Exp 015 on the same held-out questions. "
        "Repeat the evaluation across multiple question-level splits "
        "before choosing a final scoring method.",
        "",
    ]

    report_path = RESULTS_DIR / "exp015_report.md"

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"\nReport saved: {report_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "TechGrade AI Experiment 015 — Learned Regression Scorer"
        )
    )

    parser.add_argument(
        "--os-csv",
        type=Path,
        default=DEFAULT_OS_CSV,
        help="Path to the revised OS CSV.",
    )

    parser.add_argument(
        "--dbms-csv",
        type=Path,
        default=DEFAULT_DBMS_CSV,
        help="Path to the revised DBMS CSV.",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("TECHGRADE AI — EXPERIMENT 015")
    print("LEARNED REGRESSION SCORER")
    print("=" * 72)

    print(
        f"\nOS dataset:   {args.os_csv}"
    )

    print(
        f"DBMS dataset: {args.dbms_csv}"
    )

    print(
        f"Results path: {RESULTS_DIR}"
    )

    all_metrics = []
    all_tuning = []
    all_predictions = []

    for dataset_name, csv_path in [
        ("OS", args.os_csv),
        ("DBMS", args.dbms_csv),
    ]:

        metrics_df, tuning_df, predictions_df = run_dataset(
            csv_path=csv_path,
            dataset_name=dataset_name,
        )

        all_metrics.append(
            metrics_df
        )

        all_tuning.append(
            tuning_df
        )

        all_predictions.append(
            predictions_df
        )

    combined_metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    combined_tuning = pd.concat(
        all_tuning,
        ignore_index=True,
    )

    metrics_path = (
        RESULTS_DIR / "exp015_metrics.csv"
    )

    tuning_path = (
        RESULTS_DIR / "exp015_validation_tuning.csv"
    )

    combined_metrics.to_csv(
        metrics_path,
        index=False,
    )

    combined_tuning.to_csv(
        tuning_path,
        index=False,
    )

    write_report(
        metrics_df=combined_metrics,
        tuning_df=combined_tuning,
    )

    print("\n" + "=" * 72)
    print("EXPERIMENT 015 COMPLETE")
    print("=" * 72)

    print(
        f"\nMetrics: {metrics_path}"
    )

    print(
        f"Tuning results: {tuning_path}"
    )

    print(
        f"Report: {RESULTS_DIR / 'exp015_report.md'}"
    )

    print(
        "\nReminder: this is a small-data research experiment. "
        "Compare metrics and limitations before selecting a final model."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())