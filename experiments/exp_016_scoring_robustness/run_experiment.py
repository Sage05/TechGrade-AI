"""
TechGrade AI — Experiment 016
Scoring Robustness and Comparative Evaluation

Objective
---------
Compare numerical scoring methods across multiple question-level
train/validation/test splits.

Methods:
1. Global training-score mean.
2. Predicted-category training-score mean.
3. Probability-weighted categorical scoring.
4. Ridge Regression.

Methodology:
- Five repeatable question-level splits per subject.
- TF-IDF is fitted on training text only.
- Exp 008 classifier is fitted on training data only.
- Ridge alpha is selected using validation MAE.
- Test scores are used only for final evaluation.
- The same split is used for all scoring methods.

Outputs:
- exp016_run_metrics.csv
- exp016_summary.csv
- exp016_ridge_tuning.csv
- exp016_predictions.csv
- exp016_split_manifest.csv
- exp016_training_label_means.csv
- exp016_report.md
- Mean MAE and RMSE comparison plots
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

# Five repetitions. All methods share the same split within a run.
RANDOM_STATES = [42, 52, 62, 72, 82]

# Select Ridge alpha using validation MAE only.
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]

EXPECTED_FEATURE_COUNT = 36


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Validate required columns and score targets."""

    df = df.copy()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: missing columns: {missing_columns}"
        )

    # Retain source row information for later inspection.
    df["_source_csv_line"] = np.arange(
        2,
        len(df) + 2,
    )

    df["label"] = df["label"].astype(str).str.strip()

    unknown_labels = sorted(
        set(df["label"].unique()) - set(LABELS)
    )

    if unknown_labels:
        raise ValueError(
            f"{dataset_name}: unexpected labels: {unknown_labels}"
        )

    scores = pd.to_numeric(
        df["human_score"],
        errors="coerce",
    )

    if scores.isna().any():
        raise ValueError(
            f"{dataset_name}: missing or non-numeric human scores."
        )

    if ((scores < 0) | (scores > 10)).any():
        raise ValueError(
            f"{dataset_name}: human scores must be within 0–10."
        )

    df["human_score"] = scores.astype(float)

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
                f"{dataset_name}: empty values in {column}."
            )

    if df["question_text"].nunique() < 5:
        raise ValueError(
            f"{dataset_name}: too few unique questions."
        )

    return df


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def build_features(
    df: pd.DataFrame,
    extractor: FeatureExtractor,
) -> np.ndarray:
    """Extract the existing Exp 008 feature vectors."""

    rows = []

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

        rows.append(
            np.asarray(
                result.values,
                dtype=np.float32,
            )
        )

    if not rows:
        raise ValueError(
            "Cannot extract features from an empty split."
        )

    matrix = np.vstack(rows)

    if matrix.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"but got {matrix.shape[1]}."
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
# ONE QUESTION-LEVEL SPLIT
# ============================================================

def run_one_split(
    df: pd.DataFrame,
    dataset_name: str,
    random_state: int,
) -> tuple[list, list, list, list]:
    """Train and evaluate every scoring method on one fixed split."""

    print("\n" + "-" * 72)
    print(
        f"{dataset_name} | random_state={random_state}"
    )
    print("-" * 72)

    train_df, validation_df, test_df = question_level_split(
        df,
        random_state=random_state,
    )

    if (
        train_df.empty
        or validation_df.empty
        or test_df.empty
    ):
        raise RuntimeError(
            f"{dataset_name}, seed {random_state}: empty split."
        )

    # Explicitly ensure that no question appears in multiple splits.
    no_leakage = verify_no_question_leakage(
        train_df,
        validation_df,
        test_df,
    )

    if not no_leakage:
        raise RuntimeError(
            f"{dataset_name}, seed {random_state}: "
            "question leakage detected."
        )

    print(
        f"Train:      {len(train_df):3d} rows / "
        f"{train_df['question_text'].nunique():2d} questions"
    )

    print(
        f"Validation: {len(validation_df):3d} rows / "
        f"{validation_df['question_text'].nunique():2d} questions"
    )

    print(
        f"Test:       {len(test_df):3d} rows / "
        f"{test_df['question_text'].nunique():2d} questions"
    )

    print("Question leakage: NONE")

    split_rows = []

    for split_name, split_df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        for question in sorted(
            split_df["question_text"].astype(str).unique()
        ):
            split_rows.append(
                {
                    "dataset": dataset_name,
                    "random_state": random_state,
                    "split": split_name,
                    "question_text": question,
                }
            )

    # --------------------------------------------------------
    # Fit the existing feature components
    # --------------------------------------------------------

    tfidf = TfidfBaseline()

    training_corpus = (
        train_df["question_text"].astype(str).tolist()
        + train_df["model_answer"].astype(str).tolist()
        + train_df["student_answer"].astype(str).tolist()
    )

    # Critical: TF-IDF vocabulary comes from training text only.
    tfidf.fit(
        training_corpus
    )

    embedding = EmbeddingBaseline()
    nli = NLIBaseline()
    concept_matcher = ConceptMatcher()

    extractor = FeatureExtractor(
        tfidf_model=tfidf,
        embedding_model=embedding,
        nli_model=nli,
        concept_matcher=concept_matcher,
    )

    print("Extracting training features...")
    X_train = build_features(
        train_df,
        extractor,
    )

    print("Extracting validation features...")
    X_validation = build_features(
        validation_df,
        extractor,
    )

    print("Extracting test features...")
    X_test = build_features(
        test_df,
        extractor,
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
    # Categorical classifier
    # --------------------------------------------------------

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

    # All score mappings derive from the training subset.
    label_means = (
        train_df.groupby("label")["human_score"]
        .mean()
        .to_dict()
    )

    global_mean = float(
        train_df["human_score"].mean()
    )

    # Every classifier class must have a training score mapping.
    missing_means = [
        label
        for label in classes
        if label not in label_means
    ]

    if missing_means:
        raise RuntimeError(
            f"Missing training score mappings: {missing_means}"
        )

    probabilities = classifier.predict_proba(
        X_test
    )

    predicted_labels = classes[
        probabilities.argmax(axis=1)
    ]

    class_means = np.asarray(
        [
            label_means[label]
            for label in classes
        ],
        dtype=float,
    )

    baseline_predictions = {
        "Global mean baseline": np.full(
            len(test_df),
            global_mean,
        ),

        "Predicted-category mean": np.asarray(
            [
                label_means[label]
                for label in predicted_labels
            ],
            dtype=float,
        ),

        "Probability-weighted scorer": (
            probabilities @ class_means
        ),
    }

    # --------------------------------------------------------
    # Ridge regression tuning
    # --------------------------------------------------------

    tuning_rows = []
    ridge_models = {}

    for alpha in RIDGE_ALPHAS:

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=alpha)),
            ]
        )

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
                "random_state": random_state,
                "alpha": alpha,
                "validation_rows": len(validation_df),
                "validation_questions": (
                    validation_df["question_text"].nunique()
                ),
                "validation_MAE": float(validation_mae),
                "validation_RMSE": validation_rmse,
            }
        )

        ridge_models[alpha] = model

    tuning_df = pd.DataFrame(
        tuning_rows
    )

    # Validation MAE selects alpha; test metrics do not.
    best_alpha = float(
        tuning_df.sort_values(
            ["validation_MAE", "alpha"],
            ascending=[True, True],
        ).iloc[0]["alpha"]
    )

    ridge_model = ridge_models[
        best_alpha
    ]

    ridge_predictions = np.clip(
        ridge_model.predict(X_test),
        0.0,
        10.0,
    )

    all_predictions = {
        **baseline_predictions,
        "Ridge regression": ridge_predictions,
    }

    print(
        f"Selected Ridge alpha: {best_alpha}"
    )

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    metric_rows = []

    for method_name, predictions in all_predictions.items():

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        metric_rows.append(
            {
                "dataset": dataset_name,
                "random_state": random_state,
                "method": method_name,
                "selected_ridge_alpha": (
                    best_alpha
                    if method_name == "Ridge regression"
                    else np.nan
                ),
                **metrics,
            }
        )

    # --------------------------------------------------------
    # Training mapping audit
    # --------------------------------------------------------

    mapping_rows = []

    for label in LABELS:

        mapping_rows.append(
            {
                "dataset": dataset_name,
                "random_state": random_state,
                "label": label,
                "training_count": int(
                    (train_df["label"] == label).sum()
                ),
                "training_mean_score": label_means.get(
                    label,
                    np.nan,
                ),
            }
        )

    # --------------------------------------------------------
    # Row-level test predictions
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "dataset": dataset_name,
            "random_state": random_state,

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

            "global_mean_baseline": all_predictions[
                "Global mean baseline"
            ],

            "predicted_category_mean": all_predictions[
                "Predicted-category mean"
            ],

            "probability_weighted_score": all_predictions[
                "Probability-weighted scorer"
            ],

            "ridge_regression_score": ridge_predictions,
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

    return (
        metric_rows,
        tuning_rows,
        mapping_rows,
        predictions_df.to_dict(orient="records"),
        split_rows,
    )


# ============================================================
# REPORT AND PLOTS
# ============================================================

def markdown_table(df: pd.DataFrame) -> str:
    """Format Markdown tables without a tabulate dependency."""

    if df.empty:
        return "_No data available._"

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


def create_summary(
    runs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate performance across repeated splits."""

    summary = (
        runs_df.groupby(
            ["dataset", "method"],
            as_index=False,
        )
        .agg(
            repetitions=("random_state", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            Spearman_mean=("Spearman", "mean"),
            Spearman_std=("Spearman", "std"),
            mean_prediction_error=("mean_prediction_error", "mean"),
        )
    )

    return summary


def create_plots(
    summary_df: pd.DataFrame,
) -> None:
    """Create mean MAE and RMSE plots."""

    try:
        import matplotlib.pyplot as plt

    except ImportError:
        print(
            "Matplotlib is unavailable; skipping plots."
        )
        return

    for metric in [
        "MAE",
        "RMSE",
    ]:

        value_column = f"{metric}_mean"

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        for dataset_name in [
            "OS",
            "DBMS",
        ]:

            subset = summary_df[
                summary_df["dataset"] == dataset_name
            ]

            x_positions = np.arange(
                len(subset)
            )

            offset = (
                -0.18
                if dataset_name == "OS"
                else 0.18
            )

            ax.bar(
                x_positions + offset,
                subset[value_column].to_numpy(),
                width=0.36,
                label=dataset_name,
                yerr=subset[
                    f"{metric}_std"
                ].fillna(0).to_numpy(),
                capsize=4,
            )

        methods = (
            summary_df[
                summary_df["dataset"] == "OS"
            ]["method"].tolist()
        )

        ax.set_xticks(
            np.arange(len(methods))
        )

        ax.set_xticklabels(
            methods,
            rotation=15,
            ha="right",
        )

        ax.set_ylabel(metric)
        ax.set_title(
            f"Exp 016 — Mean {metric} Across Repeated Splits"
        )

        ax.legend()
        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        output_path = (
            RESULTS_DIR
            / f"exp016_mean_{metric.lower()}_by_method.png"
        )

        fig.savefig(
            output_path,
            dpi=160,
        )

        plt.close(fig)

        print(
            f"Plot saved: {output_path}"
        )


def write_report(
    summary_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    runs_df: pd.DataFrame,
) -> None:
    """Write the experiment report."""

    report_lines = [
        "# TechGrade AI — Experiment 016",
        "",
        "## Scoring Robustness and Comparative Evaluation",
        "",
        "### Objective",
        "",
        "Compare deterministic scoring baselines and Ridge Regression "
        "across repeated question-level splits.",
        "",
        "### Experimental protocol",
        "",
        f"- Random states: {RANDOM_STATES}",
        "- Question-level train/validation/test separation.",
        "- TF-IDF is fitted on training text only.",
        "- Score mappings use training scores only.",
        "- Ridge alpha is selected using validation MAE.",
        "- Test scores are reserved for final evaluation.",
        "- All scoring methods use the same test rows within each run.",
        "",
        "### Mean and standard deviation across repetitions",
        "",
        markdown_table(summary_df),
        "",
        "### Ridge validation tuning results",
        "",
        markdown_table(tuning_df),
        "",
        "### Per-split test metrics",
        "",
        markdown_table(
            runs_df[
                [
                    "dataset",
                    "random_state",
                    "method",
                    "test_rows",
                    "MAE",
                    "RMSE",
                    "R2",
                    "Spearman",
                ]
            ]
        ),
        "",
        "### Interpretation",
        "",
        "- Lower MAE and RMSE are better.",
        "- Lower standard deviation suggests more consistent performance "
        "across the tested splits, but does not prove future stability.",
        "- Higher R-squared and Spearman generally indicate better fit "
        "and ranking, respectively.",
        "- Mean prediction error near zero indicates less average bias.",
        "- The small number of unique questions limits the reliability "
        "of both mean metrics and standard deviations.",
        "",
        "### Limitations",
        "",
        "- Five splits provide preliminary evidence, not definitive "
        "statistical validation.",
        "- Overlapping test questions between repeated splits mean the "
        "five runs are not fully independent samples.",
        "- Validation sets contain few unique questions, so selected "
        "Ridge alpha can be unstable.",
        "- Category probabilities may not be calibrated.",
        "- Results do not establish production readiness.",
        "",
        "### Recommended decision rule",
        "",
        "Prefer a scoring method only when its error metrics and "
        "split-to-split stability are meaningfully competitive. "
        "Do not choose a final scorer from one metric or one subject alone.",
        "",
    ]

    output_path = (
        RESULTS_DIR / "exp016_report.md"
    )

    output_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(
        f"Report saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "TechGrade AI Experiment 016 — "
            "Scoring Robustness and Comparative Evaluation"
        )
    )

    parser.add_argument(
        "--os-csv",
        type=Path,
        default=DEFAULT_OS_CSV,
        help="Path to the revised OS dataset.",
    )

    parser.add_argument(
        "--dbms-csv",
        type=Path,
        default=DEFAULT_DBMS_CSV,
        help="Path to the revised DBMS dataset.",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("TECHGRADE AI — EXPERIMENT 016")
    print("SCORING ROBUSTNESS AND COMPARATIVE EVALUATION")
    print("=" * 72)

    print(
        f"\nRandom states: {RANDOM_STATES}"
    )

    print(
        f"Results path: {RESULTS_DIR}"
    )

    all_metric_rows = []
    all_tuning_rows = []
    all_mapping_rows = []
    all_prediction_rows = []
    all_split_rows = []

    datasets = [
        ("OS", args.os_csv),
        ("DBMS", args.dbms_csv),
    ]

    for dataset_name, csv_path in datasets:

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}"
            )

        df = validate_dataset(
            pd.read_csv(csv_path),
            dataset_name,
        )

        print("\n" + "=" * 72)
        print(
            f"STARTING {dataset_name} — "
            f"{len(RANDOM_STATES)} REPEATED SPLITS"
        )
        print("=" * 72)

        for random_state in RANDOM_STATES:

            (
                metric_rows,
                tuning_rows,
                mapping_rows,
                prediction_rows,
                split_rows,
            ) = run_one_split(
                df=df,
                dataset_name=dataset_name,
                random_state=random_state,
            )

            all_metric_rows.extend(
                metric_rows
            )

            all_tuning_rows.extend(
                tuning_rows
            )

            all_mapping_rows.extend(
                mapping_rows
            )

            all_prediction_rows.extend(
                prediction_rows
            )

            all_split_rows.extend(
                split_rows
            )

    # --------------------------------------------------------
    # Assemble results
    # --------------------------------------------------------

    runs_df = pd.DataFrame(
        all_metric_rows
    )

    tuning_df = pd.DataFrame(
        all_tuning_rows
    )

    mappings_df = pd.DataFrame(
        all_mapping_rows
    )

    predictions_df = pd.DataFrame(
        all_prediction_rows
    )

    split_manifest_df = pd.DataFrame(
        all_split_rows
    )

    summary_df = create_summary(
        runs_df
    )

    # --------------------------------------------------------
    # Save all output files
    # --------------------------------------------------------

    runs_df.to_csv(
        RESULTS_DIR / "exp016_run_metrics.csv",
        index=False,
    )

    summary_df.to_csv(
        RESULTS_DIR / "exp016_summary.csv",
        index=False,
    )

    tuning_df.to_csv(
        RESULTS_DIR / "exp016_ridge_tuning.csv",
        index=False,
    )

    mappings_df.to_csv(
        RESULTS_DIR / "exp016_training_label_means.csv",
        index=False,
    )

    predictions_df.to_csv(
        RESULTS_DIR / "exp016_predictions.csv",
        index=False,
    )

    split_manifest_df.to_csv(
        RESULTS_DIR / "exp016_split_manifest.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("EXP 016 SUMMARY — MEAN ACROSS REPEATED SPLITS")
    print("=" * 72)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        f"\nSaved repeated-run metrics: "
        f"{RESULTS_DIR / 'exp016_run_metrics.csv'}"
    )

    print(
        f"Saved summary: "
        f"{RESULTS_DIR / 'exp016_summary.csv'}"
    )

    print(
        f"Saved predictions: "
        f"{RESULTS_DIR / 'exp016_predictions.csv'}"
    )

    print(
        f"Saved split manifest: "
        f"{RESULTS_DIR / 'exp016_split_manifest.csv'}"
    )

    create_plots(
        summary_df
    )

    write_report(
        summary_df=summary_df,
        tuning_df=tuning_df,
        runs_df=runs_df,
    )

    print("\n" + "=" * 72)
    print("EXPERIMENT 016 COMPLETE")
    print("=" * 72)

    print(
        "\nReminder: the datasets contain few unique questions. "
        "Interpret results as exploratory, not proof of generalization."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())