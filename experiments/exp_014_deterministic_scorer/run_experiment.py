"""
TechGrade AI — Experiment 014
Deterministic Numerical Scoring Baseline

Purpose
-------
Convert the Exp 008 categorical grader's class probabilities
into a numerical score between 0 and 10.

Methods compared:
1. Global training-score mean baseline.
2. Mean score of the most likely predicted category.
3. Probability-weighted mean score across categories.

All category means are calculated using the TRAINING SET ONLY.
The held-out test scores are used exclusively for evaluation.

Evaluation:
- MAE
- RMSE
- R-squared
- Spearman rank correlation

Question-level train/validation/test splitting is used to reduce
leakage between answers to the same question.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Validate required columns and targets without silently
    changing or repairing dataset annotations.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: missing columns: "
            f"{', '.join(missing_columns)}"
        )

    df = df.copy()

    # Preserve the original CSV line for traceability.
    df["_exp014_source_csv_line"] = np.arange(
        2,
        len(df) + 2,
    )

    # Normalize harmless whitespace in categorical labels.
    df["label"] = df["label"].astype(str).str.strip()

    unknown_labels = sorted(
        set(df["label"].unique()) - set(LABELS)
    )

    if unknown_labels:
        raise ValueError(
            f"{dataset_name}: unexpected labels found: "
            f"{unknown_labels}"
        )

    # Do not silently impute target scores.
    original_scores = df["human_score"]
    numeric_scores = pd.to_numeric(
        original_scores,
        errors="coerce",
    )

    invalid_scores = numeric_scores.isna()

    out_of_range = (
        numeric_scores.notna()
        & (
            (numeric_scores < 0)
            | (numeric_scores > 10)
        )
    )

    if invalid_scores.any():
        raise ValueError(
            f"{dataset_name}: "
            f"{int(invalid_scores.sum())} missing or "
            "non-numeric human scores. Resolve them before "
            "running Exp 014."
        )

    if out_of_range.any():
        raise ValueError(
            f"{dataset_name}: human scores outside 0–10 "
            f"found in {int(out_of_range.sum())} rows."
        )

    df["human_score"] = numeric_scores.astype(float)

    # Feature extraction requires these text fields.
    text_columns = [
        "question_text",
        "model_answer",
        "student_answer",
    ]

    for column in text_columns:
        missing_text = (
            df[column].isna()
            | df[column].astype(str).str.strip().eq("")
        )

        if missing_text.any():
            raise ValueError(
                f"{dataset_name}: {int(missing_text.sum())} "
                f"empty values in '{column}'."
            )

    if df["question_text"].nunique() < 3:
        raise ValueError(
            f"{dataset_name}: too few unique questions "
            "for question-level evaluation."
        )

    return df


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def build_feature_matrix(
    df: pd.DataFrame,
    extractor: FeatureExtractor,
) -> np.ndarray:
    """Use the existing application-level feature extractor."""

    rows = []

    for _, row in df.iterrows():

        features = extractor.extract(
            question=str(row["question_text"]),
            reference_answer=str(row["model_answer"]),
            student_answer=str(row["student_answer"]),
            required_concepts=(
                ""
                if pd.isna(row["required_concepts"])
                else str(row["required_concepts"])
            ),
        )

        rows.append(
            np.asarray(
                features.values,
                dtype=np.float32,
            )
        )

    if not rows:
        raise ValueError(
            "Cannot build a feature matrix from an empty split."
        )

    matrix = np.vstack(rows)

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
    """Calculate regression metrics for one scoring method."""

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
        r2 = r2_score(
            actual,
            predicted,
        )
    else:
        r2 = np.nan

    actual_series = pd.Series(actual)
    predicted_series = pd.Series(predicted)

    if (
        len(actual) >= 2
        and actual_series.nunique() > 1
        and predicted_series.nunique() > 1
    ):
        spearman = actual_series.corr(
            predicted_series,
            method="spearman",
        )
    else:
        spearman = np.nan

    return {
        "test_rows": len(actual),
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": float(r2),
        "Spearman": float(spearman),
        "mean_prediction_error": float(
            np.mean(predicted - actual)
        ),
    }


# ============================================================
# TRAIN AND EVALUATE ONE DATASET
# ============================================================

def run_dataset(
    csv_path: Path,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train the categorical grader on the training split,
    derive deterministic score mappings from training scores,
    and evaluate on the held-out test split.
    """

    print("\n" + "=" * 72)
    print(f"EXPERIMENT 014 — {dataset_name}")
    print("=" * 72)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    df = validate_dataset(
        df,
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

    print("\n[QUESTION-LEVEL SPLIT]")
    print(
        f"Train:      {len(train_df)} rows / "
        f"{train_df['question_text'].nunique()} questions"
    )
    print(
        f"Validation: {len(validation_df)} rows / "
        f"{validation_df['question_text'].nunique()} questions"
    )
    print(
        f"Test:       {len(test_df)} rows / "
        f"{test_df['question_text'].nunique()} questions"
    )

    if train_df.empty or test_df.empty:
        raise RuntimeError(
            f"{dataset_name}: empty training or test split."
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

    print("Question leakage: NONE")

    # --------------------------------------------------------
    # Fit feature extraction models using training data only
    # --------------------------------------------------------

    print("\n[INITIALIZING EXISTING EXP 008 COMPONENTS]")

    tfidf_model = TfidfBaseline()

    tfidf_corpus = (
        train_df["question_text"].astype(str).tolist()
        + train_df["model_answer"].astype(str).tolist()
        + train_df["student_answer"].astype(str).tolist()
    )

    tfidf_model.fit(tfidf_corpus)

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
    # Extract features
    # --------------------------------------------------------

    print("\n[FEATURE EXTRACTION]")
    print("Training features...")

    X_train = build_feature_matrix(
        train_df,
        extractor,
    )

    print("Held-out test features...")

    X_test = build_feature_matrix(
        test_df,
        extractor,
    )

    print(
        f"Training matrix: {X_train.shape}"
    )
    print(
        f"Test matrix: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Train the existing Exp 008 classification architecture
    # --------------------------------------------------------

    print("\n[TRAINING EXP 008 CLASSIFIER]")

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

    print(
        "Classifier categories: "
        + ", ".join(classes)
    )

    # --------------------------------------------------------
    # Derive mappings using training scores ONLY
    # --------------------------------------------------------

    print("\n[TRAINING-ONLY SCORE MAPPING]")

    label_means = (
        train_df.groupby("label")["human_score"]
        .mean()
        .to_dict()
    )

    label_counts = (
        train_df.groupby("label")["human_score"]
        .count()
        .to_dict()
    )

    unseen_training_classes = [
        label
        for label in classes
        if label not in label_means
    ]

    if unseen_training_classes:
        raise RuntimeError(
            "Cannot calculate score means for classifier "
            f"categories absent from training: "
            f"{unseen_training_classes}"
        )

    global_mean = float(
        train_df["human_score"].mean()
    )

    mapping_rows = []

    for label in LABELS:
        mapping_rows.append(
            {
                "dataset": dataset_name,
                "label": label,
                "training_count": int(
                    label_counts.get(label, 0)
                ),
                "training_mean_score": (
                    float(label_means[label])
                    if label in label_means
                    else np.nan
                ),
            }
        )

    mapping_df = pd.DataFrame(mapping_rows)

    print(
        f"Global training mean: {global_mean:.4f}"
    )

    print("\nMean human score by training label:")

    for label in classes:
        print(
            f"  {label:20s} "
            f"{label_means[label]:.4f} "
            f"(n={label_counts[label]})"
        )

    # --------------------------------------------------------
    # Predict categorical probabilities on the held-out test set
    # --------------------------------------------------------

    print("\n[HELD-OUT SCORING]")

    probabilities = classifier.predict_proba(
        X_test
    )

    predicted_label_indices = probabilities.argmax(
        axis=1
    )

    predicted_labels = classes[
        predicted_label_indices
    ]

    # Method 1: constant score using training-set global mean.
    global_predictions = np.full(
        len(test_df),
        global_mean,
        dtype=float,
    )

    # Method 2: mean score associated with predicted category.
    hard_label_predictions = np.asarray(
        [
            label_means[label]
            for label in predicted_labels
        ],
        dtype=float,
    )

    # Method 3: probability-weighted mean of category scores.
    #
    # Every category mean comes from training scores only.
    # Test human scores do not influence these predictions.
    category_score_vector = np.asarray(
        [
            label_means[label]
            for label in classes
        ],
        dtype=float,
    )

    probability_weighted_predictions = (
        probabilities @ category_score_vector
    )

    actual_scores = test_df[
        "human_score"
    ].to_numpy(dtype=float)

    actual_labels = test_df[
        "label"
    ].to_numpy()

    # Keep predictions bounded to the defined 0–10 scale.
    global_predictions = np.clip(
        global_predictions,
        0.0,
        10.0,
    )

    hard_label_predictions = np.clip(
        hard_label_predictions,
        0.0,
        10.0,
    )

    probability_weighted_predictions = np.clip(
        probability_weighted_predictions,
        0.0,
        10.0,
    )

    # --------------------------------------------------------
    # Evaluate all methods
    # --------------------------------------------------------

    methods = {
        "Global mean baseline": global_predictions,
        "Predicted-category mean": hard_label_predictions,
        "Probability-weighted scorer": (
            probability_weighted_predictions
        ),
    }

    metric_rows = []

    for method_name, predictions in methods.items():

        metrics = calculate_metrics(
            actual_scores,
            predictions,
        )

        metric_rows.append(
            {
                "dataset": dataset_name,
                "method": method_name,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    print("\n[TEST METRICS]")
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # Save row-level predictions
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "source_csv_line": test_df[
                "_exp014_source_csv_line"
            ].to_numpy(),
            "question_text": test_df[
                "question_text"
            ].to_numpy(),
            "student_answer": test_df[
                "student_answer"
            ].to_numpy(),
            "actual_label": actual_labels,
            "predicted_label": predicted_labels,
            "actual_human_score": actual_scores,
            "global_mean_baseline": global_predictions,
            "predicted_category_mean_score": (
                hard_label_predictions
            ),
            "probability_weighted_score": (
                probability_weighted_predictions
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

    output_path = (
        RESULTS_DIR
        / f"exp014_predictions_{dataset_name.lower()}.csv"
    )

    predictions_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nPredictions saved to: {output_path}"
    )

    # --------------------------------------------------------
    # Plot actual vs predicted scores
    # --------------------------------------------------------

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        ax.scatter(
            actual_scores,
            hard_label_predictions,
            label="Predicted-category mean",
            alpha=0.8,
        )

        ax.scatter(
            actual_scores,
            probability_weighted_predictions,
            label="Probability-weighted scorer",
            alpha=0.8,
        )

        ax.plot(
            [0, 10],
            [0, 10],
            linestyle="--",
            label="Perfect prediction",
        )

        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_xlabel("Human-assigned score")
        ax.set_ylabel("Predicted score")
        ax.set_title(
            f"Exp 014 — Actual vs Predicted Scores ({dataset_name})"
        )
        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()

        plot_path = (
            RESULTS_DIR
            / f"exp014_actual_vs_predicted_{dataset_name.lower()}.png"
        )

        fig.savefig(
            plot_path,
            dpi=160,
        )

        plt.close(fig)

        print(
            f"Plot saved to: {plot_path}"
        )

    except ImportError:
        print(
            "WARNING: Matplotlib is not installed; "
            "plot generation was skipped."
        )

    return metrics_df, mapping_df, predictions_df


# ============================================================
# REPORT
# ============================================================

def markdown_table(df: pd.DataFrame) -> str:
    """Format a DataFrame as Markdown without requiring tabulate."""

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
                text = "NA"

            elif isinstance(value, (float, np.floating)):
                text = f"{value:.4f}"

            else:
                text = str(value)

            text = text.replace("|", "\\|").replace(
                "\n",
                " ",
            )

            values.append(text)

        lines.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(lines)


def write_report(
    metrics_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> None:

    report_path = (
        RESULTS_DIR
        / "exp_014_report.md"
    )

    report_lines = [
        "# TechGrade AI — Experiment 014",
        "",
        "## Deterministic Numerical Scoring Baseline",
        "",
        "### Objective",
        "",
        "Test whether Exp 008 categorical predictions can be "
        "converted into 0–10 numerical scores using category "
        "means estimated from the training data.",
        "",
        "### Scoring methods",
        "",
        "1. **Global mean baseline:** predicts the training-set "
        "mean human score for every held-out answer.",
        "2. **Predicted-category mean:** uses the training-set "
        "mean for the most likely predicted category.",
        "3. **Probability-weighted scorer:** computes the weighted "
        "average of training-set category means using Exp 008 "
        "class probabilities.",
        "",
        "### Training-only rule",
        "",
        "Human scores in the held-out test set are used only to "
        "calculate evaluation metrics. They are not used to "
        "construct category means or numerical predictions.",
        "",
        "### Category score mappings",
        "",
        markdown_table(mapping_df),
        "",
        "### Held-out test results",
        "",
        markdown_table(metrics_df),
        "",
        "### Interpretation guidance",
        "",
        "- Lower MAE and RMSE indicate smaller prediction errors.",
        "- Higher R-squared indicates that more test-score variance "
        "is explained relative to predicting the test-set mean.",
        "- Spearman correlation measures rank association, not "
        "numerical agreement.",
        "- Negative R-squared is possible and means predictions "
        "perform worse than the test-set-mean reference under "
        "that metric.",
        "- Category probabilities from the current Logistic "
        "Regression model are not guaranteed to be calibrated. "
        "The probability-weighted result is a baseline, not a "
        "validated final scoring method.",
        "",
        "### Limitations",
        "",
        "- The datasets contain only a small number of unique questions.",
        "- The results come from one question-level split per subject.",
        "- The validation split is not used to tune this deterministic "
        "scoring rule.",
        "- Category means may be unstable where training counts "
        "are small.",
        "- Numerical scores may capture distinctions that are not "
        "represented by the five categorical labels.",
        "- Results do not establish production-level grading accuracy.",
        "",
        "### Next step",
        "",
        "Compare this deterministic baseline with a learned regression "
        "model in Experiment 015 using the same question-level "
        "evaluation discipline.",
        "",
    ]

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(
        f"\nReport saved to: {report_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "TechGrade AI Experiment 014 — "
            "Deterministic Numerical Scoring Baseline"
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
    print("TECHGRADE AI — EXPERIMENT 014")
    print("DETERMINISTIC NUMERICAL SCORING BASELINE")
    print("=" * 72)

    print(f"\nOS dataset:   {args.os_csv}")
    print(f"DBMS dataset: {args.dbms_csv}")
    print(f"Results path: {RESULTS_DIR}")

    all_metrics = []
    all_mappings = []

    for dataset_name, csv_path in [
        ("OS", args.os_csv),
        ("DBMS", args.dbms_csv),
    ]:

        metrics_df, mapping_df, _ = run_dataset(
            csv_path=csv_path,
            dataset_name=dataset_name,
        )

        all_metrics.append(
            metrics_df
        )

        all_mappings.append(
            mapping_df
        )

    combined_metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    combined_mappings = pd.concat(
        all_mappings,
        ignore_index=True,
    )

    metrics_path = (
        RESULTS_DIR
        / "exp014_metrics.csv"
    )

    mappings_path = (
        RESULTS_DIR
        / "exp014_training_label_means.csv"
    )

    combined_metrics.to_csv(
        metrics_path,
        index=False,
    )

    combined_mappings.to_csv(
        mappings_path,
        index=False,
    )

    write_report(
        combined_metrics,
        combined_mappings,
    )

    print("\n" + "=" * 72)
    print("EXPERIMENT 014 COMPLETE")
    print("=" * 72)

    print(
        f"\nMetrics: {metrics_path}"
    )

    print(
        f"Training label means: {mappings_path}"
    )

    print(
        "\nImportant: this experiment tests a deterministic "
        "scoring baseline. It does not yet establish that the "
        "method is accurate enough for deployment."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())