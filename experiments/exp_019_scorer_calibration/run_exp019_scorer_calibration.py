"""
TechGradeAI — Experiment 019
Validation-Only Scorer Calibration

Compares:
1. Predicted-category mean
2. Probability-weighted category score
3. Ridge regression

For each scorer:
- Raw score
- Validation-fitted bias correction
- Validation-fitted affine calibration

The test set is never used to fit calibration parameters.
All dataset splits are question-level splits.

Ridge alpha is selected using GroupKFold inside the outer training
partition, preserving the validation partition for calibration.

This experiment is exploratory. Repeated splits reuse some questions
and must not be treated as independent external validation datasets.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

from src.utils.data_split import question_level_split
from src.models.tfidf_baseline import TfidfBaseline
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.nli_baseline import NLIBaseline
from src.models.concept_matcher import ConceptMatcher
from src.grading.feature_extractor import FeatureExtractor


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "OS": Path(
        Path.home() / "Downloads" / "revised_sample_os.csv"
    ),
    "DBMS": Path(
        Path.home() / "Downloads" / "revised_sample_dbms.csv"
    ),
}

RANDOM_STATES = [42, 52, 62, 72, 82]

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "exp_019_scorer_calibration"
    / "results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "question_text",
    "model_answer",
    "student_answer",
    "required_concepts",
    "label",
    "human_score",
]

SCORER_NAMES = [
    "Predicted-category mean",
    "Probability-weighted scorer",
    "Ridge regression",
]


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def safe_text(value):
    """Convert missing values to empty strings."""
    if pd.isna(value):
        return ""
    return str(value)


def build_features(frame, extractor):
    """
    Extract the existing TechGradeAI feature representation.

    This uses the project's existing 36-feature FeatureExtractor.
    It does not use human concept annotations as model inputs.
    """
    feature_rows = []

    for _, row in frame.iterrows():
        result = extractor.extract(
            question=safe_text(row["question_text"]),
            reference_answer=safe_text(row["model_answer"]),
            student_answer=safe_text(row["student_answer"]),
            required_concepts=safe_text(
                row["required_concepts"]
            ),
        )

        feature_rows.append(
            np.asarray(result.values, dtype=np.float32)
        )

    if not feature_rows:
        raise ValueError("Cannot extract features from an empty split.")

    return np.vstack(feature_rows)


def verify_group_separation(train, validation, test):
    """Ensure that one question cannot occur in multiple partitions."""
    train_questions = set(
        train["question_text"].astype(str)
    )
    val_questions = set(
        validation["question_text"].astype(str)
    )
    test_questions = set(
        test["question_text"].astype(str)
    )

    train_val = train_questions & val_questions
    train_test = train_questions & test_questions
    val_test = val_questions & test_questions

    if train_val or train_test or val_test:
        raise RuntimeError(
            "Question leakage detected.\n"
            f"Train-validation overlap: {train_val}\n"
            f"Train-test overlap: {train_test}\n"
            f"Validation-test overlap: {val_test}"
        )


def select_ridge_alpha(X, y, question_groups):
    """
    Select Ridge alpha with question-grouped cross-validation
    inside the outer training data only.

    This avoids using the calibration validation set to choose alpha.
    """
    unique_groups = np.unique(question_groups)

    n_splits = min(3, len(unique_groups))

    if n_splits < 2:
        return 1.0, pd.DataFrame()

    cv = GroupKFold(n_splits=n_splits)
    records = []

    for alpha in RIDGE_ALPHAS:
        fold_maes = []

        for fold, (inner_train, inner_valid) in enumerate(
            cv.split(X, y, groups=question_groups),
            start=1,
        ):
            model = Ridge(alpha=alpha)

            model.fit(
                X[inner_train],
                y[inner_train],
            )

            predictions = model.predict(X[inner_valid])

            fold_mae = mean_absolute_error(
                y[inner_valid],
                predictions,
            )

            fold_maes.append(float(fold_mae))

            records.append({
                "alpha": alpha,
                "fold": fold,
                "validation_mae": float(fold_mae),
                "train_rows": len(inner_train),
                "validation_rows": len(inner_valid),
                "train_questions": len(
                    set(question_groups[inner_train])
                ),
                "validation_questions": len(
                    set(question_groups[inner_valid])
                ),
            })

        print(
            f"    Ridge alpha={alpha:g}; "
            f"inner-CV mean MAE={np.mean(fold_maes):.4f}"
        )

    cv_results = pd.DataFrame(records)

    mean_by_alpha = (
        cv_results.groupby("alpha")["validation_mae"]
        .mean()
        .sort_values()
    )

    selected_alpha = float(mean_by_alpha.index[0])

    return selected_alpha, cv_results


def make_score_predictions(
    classifier,
    ridge_model,
    X,
    score_means,
    overall_training_mean,
):
    """
    Generate the three numerical scorers from the same feature matrix.
    """
    predicted_labels = classifier.predict(X)
    probabilities = classifier.predict_proba(X)

    # The classifier only predicts classes present in its training set.
    category_mean = np.asarray([
        score_means.get(label, overall_training_mean)
        for label in predicted_labels
    ], dtype=float)

    probability_weighted = np.zeros(len(X), dtype=float)

    for index, label in enumerate(classifier.classes_):
        label_mean = score_means.get(
            label,
            overall_training_mean,
        )

        probability_weighted += (
            probabilities[:, index] * label_mean
        )

    ridge_scores = ridge_model.predict(X)

    outputs = {
        "Global mean baseline": np.full(
            len(X),
            overall_training_mean,
            dtype=float,
        ),
        "Predicted-category mean": category_mean,
        "Probability-weighted scorer": probability_weighted,
        "Ridge regression": ridge_scores,
    }

    return outputs, predicted_labels


# ============================================================
# 3. CALIBRATION HELPERS
# ============================================================

def fit_calibrator(y_validation, raw_validation, method):
    """
    Fit calibration parameters from validation predictions only.

    bias:
        calibrated = raw + offset

    affine:
        calibrated = slope * raw + intercept

    The affine slope is constrained to be nonnegative so that
    calibration cannot reverse the order of predicted scores.
    """
    actual = np.asarray(y_validation, dtype=float)
    raw = np.asarray(raw_validation, dtype=float)

    if len(actual) != len(raw) or len(actual) == 0:
        raise ValueError(
            "Calibration inputs have incompatible lengths."
        )

    if method == "bias_correction":
        slope = 1.0
        intercept = float(np.mean(actual - raw))

    elif method == "affine_calibration":
        raw_mean = float(np.mean(raw))
        actual_mean = float(np.mean(actual))

        centered_raw = raw - raw_mean
        variance = float(np.mean(centered_raw ** 2))

        if variance < 1e-12:
            slope = 0.0
        else:
            covariance = float(
                np.mean(
                    centered_raw * (actual - actual_mean)
                )
            )

            slope = max(0.0, covariance / variance)

        intercept = actual_mean - slope * raw_mean

    else:
        raise ValueError(
            f"Unknown calibration method: {method}"
        )

    return {
        "method": method,
        "slope": float(slope),
        "intercept": float(intercept),
        "validation_rows": int(len(actual)),
        "validation_unique_questions": None,
    }


def apply_calibrator(raw_scores, parameters):
    """Apply learned calibration and keep scores inside 0–10."""
    raw = np.asarray(raw_scores, dtype=float)

    calibrated = (
        parameters["slope"] * raw
        + parameters["intercept"]
    )

    return np.clip(calibrated, 0.0, 10.0)


# ============================================================
# 4. METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - actual
    abs_errors = np.abs(errors)

    if (
        len(actual) > 1
        and np.std(actual) > 0
        and np.std(predicted) > 0
    ):
        spearman = pd.Series(actual).corr(
            pd.Series(predicted),
            method="spearman",
        )
    else:
        spearman = np.nan

    return {
        "n_test_rows": int(len(actual)),
        "MAE": float(np.mean(abs_errors)),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "mean_prediction_error": float(np.mean(errors)),
        "median_absolute_error": float(np.median(abs_errors)),
        "Spearman": float(spearman),
        "within_1_point_pct": float(
            np.mean(abs_errors <= 1.0) * 100
        ),
        "within_2_points_pct": float(
            np.mean(abs_errors <= 2.0) * 100
        ),
        "below_zero_pct": float(
            np.mean(predicted < 0) * 100
        ),
        "above_10_pct": float(
            np.mean(predicted > 10) * 100
        ),
    }


def make_markdown_table(frame, columns):
    """Create Markdown tables without requiring tabulate."""
    if frame.empty:
        return "_No data available._"

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in frame[columns].iterrows():
        values = []

        for column in columns:
            value = row[column]

            if pd.isna(value):
                values.append("—")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.3f}")
            elif isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            else:
                values.append(
                    str(value).replace("|", "\\|").replace("\n", " ")
                )

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


# ============================================================
# 5. MAIN EXPERIMENT
# ============================================================

def main():
    print("=" * 76)
    print("EXPERIMENT 019 — VALIDATION-ONLY SCORER CALIBRATION")
    print("=" * 76)

    run_metric_rows = []
    prediction_rows = []
    calibration_rows = []
    ridge_cv_rows = []
    split_rows = []

    # Reuse pretrained encoders across random states within a dataset.
    # The TF-IDF model is refitted for each training split.
    for dataset_name, csv_path in DATASETS.items():

        print("\n" + "=" * 76)
        print(f"DATASET: {dataset_name}")
        print("=" * 76)

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}"
            )

        df = pd.read_csv(csv_path)

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{dataset_name} dataset is missing columns: "
                f"{missing_columns}\n"
                f"Available columns: {list(df.columns)}"
            )

        df = df.copy().reset_index(drop=True)
        df["_source_row_id"] = np.arange(len(df))

        df["question_text"] = (
            df["question_text"].fillna("").astype(str)
        )
        df["model_answer"] = (
            df["model_answer"].fillna("").astype(str)
        )
        df["student_answer"] = (
            df["student_answer"].fillna("").astype(str)
        )
        df["required_concepts"] = (
            df["required_concepts"].fillna("").astype(str)
        )
        df["label"] = (
            df["label"].fillna("").astype(str).str.strip()
        )
        df["human_score"] = pd.to_numeric(
            df["human_score"],
            errors="coerce",
        )

        if df["human_score"].isna().any():
            raise ValueError(
                f"{dataset_name} contains missing or non-numeric "
                "human scores. Resolve the data issue before running."
            )

        if not df["human_score"].between(0, 10).all():
            raise ValueError(
                f"{dataset_name} contains human scores outside 0–10."
            )

        if (df["label"] == "").any():
            raise ValueError(
                f"{dataset_name} contains missing categorical labels."
            )

        print(f"Rows: {len(df)}")
        print(f"Unique questions: {df['question_text'].nunique()}")

        print("Loading embedding model...")
        embedding_model = EmbeddingBaseline()

        print("Loading NLI model...")
        nli_model = NLIBaseline()

        print("Loading concept matcher...")
        concept_matcher = ConceptMatcher()

        for seed in RANDOM_STATES:
            print("\n" + "-" * 76)
            print(f"{dataset_name} | random_state={seed}")
            print("-" * 76)

            train, validation, test = question_level_split(
                df,
                random_state=seed,
            )

            train = train.copy().reset_index(drop=True)
            validation = validation.copy().reset_index(drop=True)
            test = test.copy().reset_index(drop=True)

            verify_group_separation(
                train,
                validation,
                test,
            )

            print(
                f"Train: {len(train)} rows / "
                f"{train['question_text'].nunique()} questions"
            )
            print(
                f"Calibration validation: {len(validation)} rows / "
                f"{validation['question_text'].nunique()} questions"
            )
            print(
                f"Test: {len(test)} rows / "
                f"{test['question_text'].nunique()} questions"
            )
            print("Question leakage: NONE")

            split_rows.append({
                "dataset": dataset_name,
                "random_state": seed,
                "train_rows": len(train),
                "train_questions": train["question_text"].nunique(),
                "validation_rows": len(validation),
                "validation_questions": (
                    validation["question_text"].nunique()
                ),
                "test_rows": len(test),
                "test_questions": test["question_text"].nunique(),
                "question_leakage": False,
            })

            # ----------------------------------------------------
            # Fit TF-IDF strictly using this split's training rows.
            # ----------------------------------------------------

            tfidf_model = TfidfBaseline()

            tfidf_corpus = (
                train["question_text"].astype(str).tolist()
                + train["model_answer"].astype(str).tolist()
                + train["student_answer"].astype(str).tolist()
            )

            tfidf_model.fit(tfidf_corpus)

            extractor = FeatureExtractor(
                tfidf_model=tfidf_model,
                embedding_model=embedding_model,
                nli_model=nli_model,
                concept_matcher=concept_matcher,
            )

            # ----------------------------------------------------
            # Feature extraction
            # ----------------------------------------------------

            print("Extracting training features...")
            X_train = build_features(train, extractor)

            print("Extracting calibration-validation features...")
            X_validation = build_features(validation, extractor)

            print("Extracting test features...")
            X_test = build_features(test, extractor)

            y_train = train["human_score"].to_numpy(dtype=float)
            y_validation = validation["human_score"].to_numpy(
                dtype=float
            )
            y_test = test["human_score"].to_numpy(dtype=float)

            if not (
                len(X_train) == len(y_train)
                and len(X_validation) == len(y_validation)
                and len(X_test) == len(y_test)
            ):
                raise RuntimeError(
                    "Feature/target row count mismatch detected."
                )

            print(f"Feature count: {X_train.shape[1]}")

            # ----------------------------------------------------
            # Fit categorical classifier on training data only.
            # ----------------------------------------------------

            classifier = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            )

            classifier.fit(
                X_train,
                train["label"].to_numpy(),
            )

            # Training-only numerical score means by category.
            score_means = (
                train.groupby("label")["human_score"]
                .mean()
                .to_dict()
            )

            overall_training_mean = float(np.mean(y_train))

            # ----------------------------------------------------
            # Select Ridge alpha inside training data only.
            # ----------------------------------------------------

            selected_alpha, cv_results = select_ridge_alpha(
                X_train,
                y_train,
                train["question_text"].astype(str).to_numpy(),
            )

            print(f"Selected Ridge alpha: {selected_alpha:g}")

            if not cv_results.empty:
                cv_results["dataset"] = dataset_name
                cv_results["random_state"] = seed
                ridge_cv_rows.extend(
                    cv_results.to_dict(orient="records")
                )

            ridge_model = Ridge(alpha=selected_alpha)

            ridge_model.fit(
                X_train,
                y_train,
            )

            # ----------------------------------------------------
            # Generate calibration and test scores.
            # ----------------------------------------------------

            val_scores, val_labels = make_score_predictions(
                classifier,
                ridge_model,
                X_validation,
                score_means,
                overall_training_mean,
            )

            test_scores, test_labels = make_score_predictions(
                classifier,
                ridge_model,
                X_test,
                score_means,
                overall_training_mean,
            )

            # The global mean is kept only as an uncalibrated baseline.
            # The other three scorers receive validation-fitted
            # bias and affine calibration.
            all_methods = [
                "Global mean baseline",
                *SCORER_NAMES,
            ]

            variants = [
                "raw",
                "bias_correction",
                "affine_calibration",
            ]

            for method in all_methods:

                validation_raw = val_scores[method]
                test_raw = test_scores[method]

                if method == "Global mean baseline":
                    candidate_variants = ["raw"]
                    parameters_by_variant = {
                        "raw": {
                            "method": "raw",
                            "slope": 1.0,
                            "intercept": 0.0,
                        }
                    }
                else:
                    candidate_variants = variants

                    bias_parameters = fit_calibrator(
                        y_validation,
                        validation_raw,
                        "bias_correction",
                    )

                    affine_parameters = fit_calibrator(
                        y_validation,
                        validation_raw,
                        "affine_calibration",
                    )

                    bias_parameters["validation_unique_questions"] = (
                        validation["question_text"].nunique()
                    )

                    affine_parameters[
                        "validation_unique_questions"
                    ] = validation["question_text"].nunique()

                    parameters_by_variant = {
                        "raw": {
                            "method": "raw",
                            "slope": 1.0,
                            "intercept": 0.0,
                            "validation_rows": len(validation),
                            "validation_unique_questions": (
                                validation["question_text"].nunique()
                            ),
                        },
                        "bias_correction": bias_parameters,
                        "affine_calibration": affine_parameters,
                    }

                for variant in candidate_variants:
                    parameters = parameters_by_variant[variant]

                    if variant == "raw":
                        final_scores = np.asarray(
                            test_raw,
                            dtype=float,
                        )
                    else:
                        final_scores = apply_calibrator(
                            test_raw,
                            parameters,
                        )

                    metrics = calculate_metrics(
                        y_test,
                        final_scores,
                    )

                    run_metric_rows.append({
                        "dataset": dataset_name,
                        "random_state": seed,
                        "method": method,
                        "variant": variant,
                        "ridge_alpha": selected_alpha,
                        **metrics,
                    })

                    if method != "Global mean baseline":
                        calibration_rows.append({
                            "dataset": dataset_name,
                            "random_state": seed,
                            "method": method,
                            "variant": variant,
                            **parameters,
                        })

                    # Long-form held-out predictions.
                    for i, (_, row) in enumerate(test.iterrows()):
                        prediction_rows.append({
                            "dataset": dataset_name,
                            "random_state": seed,
                            "source_row_id": int(
                                row["_source_row_id"]
                            ),
                            "question_text": row["question_text"],
                            "student_answer": row["student_answer"],
                            "actual_label": row["label"],
                            "predicted_label": str(test_labels[i]),
                            "classification_correct": (
                                str(test_labels[i]) == str(row["label"])
                            ),
                            "actual_human_score": float(y_test[i]),
                            "method": method,
                            "variant": variant,
                            "raw_score": float(test_raw[i]),
                            "predicted_score": float(final_scores[i]),
                            "residual": float(
                                final_scores[i] - y_test[i]
                            ),
                            "absolute_error": float(
                                abs(final_scores[i] - y_test[i])
                            ),
                            "calibration_slope": float(
                                parameters["slope"]
                            ),
                            "calibration_intercept": float(
                                parameters["intercept"]
                            ),
                        })

                    print(
                        f"{method:30s} | {variant:20s} | "
                        f"test MAE={metrics['MAE']:.4f} | "
                        f"RMSE={metrics['RMSE']:.4f} | "
                        f"bias={metrics['mean_prediction_error']:+.4f}"
                    )

    # ========================================================
    # 6. SAVE RESULTS
    # ========================================================

    run_metrics = pd.DataFrame(run_metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    calibration_parameters = pd.DataFrame(calibration_rows)
    split_manifest = pd.DataFrame(split_rows)
    ridge_cv = pd.DataFrame(ridge_cv_rows)

    run_metrics.to_csv(
        OUTPUT_DIR / "exp019_run_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        OUTPUT_DIR / "exp019_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    calibration_parameters.to_csv(
        OUTPUT_DIR / "exp019_calibration_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_manifest.to_csv(
        OUTPUT_DIR / "exp019_split_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ridge_cv.to_csv(
        OUTPUT_DIR / "exp019_ridge_inner_cv.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 7. SUMMARY ACROSS RANDOM STATES
    # ========================================================

    metric_columns = [
        "MAE",
        "RMSE",
        "mean_prediction_error",
        "median_absolute_error",
        "Spearman",
        "within_1_point_pct",
        "within_2_points_pct",
        "below_zero_pct",
        "above_10_pct",
    ]

    summary_rows = []

    for (dataset, method, variant), group in run_metrics.groupby(
        ["dataset", "method", "variant"],
        sort=True,
    ):
        result = {
            "dataset": dataset,
            "method": method,
            "variant": variant,
            "repetitions": int(group["random_state"].nunique()),
        }

        for metric in metric_columns:
            result[f"{metric}_mean"] = group[metric].mean()
            result[f"{metric}_std"] = group[metric].std(ddof=1)

        summary_rows.append(result)

    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        ["dataset", "MAE_mean"],
        ascending=[True, True],
    ).reset_index(drop=True)

    summary.to_csv(
        OUTPUT_DIR / "exp019_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 76)
    print("EXP 019 SUMMARY — MEAN ACROSS QUESTION-LEVEL SPLITS")
    print("=" * 76)

    display_columns = [
        "dataset",
        "method",
        "variant",
        "repetitions",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
        "mean_prediction_error_mean",
        "Spearman_mean",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # ========================================================
    # 8. PLOT MAE
    # ========================================================

    datasets_found = sorted(summary["dataset"].unique())

    fig, axes = plt.subplots(
        1,
        len(datasets_found),
        figsize=(10 * len(datasets_found), 6),
        squeeze=False,
    )

    for index, dataset in enumerate(datasets_found):
        ax = axes[0, index]

        subset = summary[
            summary["dataset"] == dataset
        ].copy()

        subset = subset.sort_values(
            ["method", "variant"]
        )

        labels = [
            f"{row.method}\n{row.variant}"
            for row in subset.itertuples()
        ]

        means = subset["MAE_mean"].to_numpy(dtype=float)

        stds = (
            subset["MAE_std"]
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        x = np.arange(len(subset))

        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=35,
            ha="right",
            fontsize=8,
        )

        ax.set_ylabel("MAE (lower is better)")
        ax.set_title(f"{dataset}: Calibration comparison")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Experiment 019 — Validation-Only Scorer Calibration",
        y=1.03,
    )

    fig.tight_layout()

    plot_path = OUTPUT_DIR / "exp019_calibration_mae_comparison.png"

    fig.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # 9. REPORT
    # ========================================================

    report_lines = [
        "# Experiment 019 — Validation-Only Scorer Calibration",
        "",
        "## Objective",
        (
            "Determine whether validation-fitted bias correction or "
            "affine calibration improves numerical scoring on held-out "
            "questions."
        ),
        "",
        "## Methodology",
        "- Five question-level random-state splits were evaluated.",
        "- The existing concept-aware feature extractor was used.",
        "- The categorical classifier was trained on the outer training split.",
        "- Ridge alpha was selected using GroupKFold within outer training.",
        "- Calibration parameters were fitted on validation predictions only.",
        "- Test labels and scores were not used to fit calibration parameters.",
        "- Calibrated test scores were clipped to the 0–10 interval.",
        "- The global training mean is an uncalibrated baseline.",
        "",
        "## Mean metrics across splits",
        make_markdown_table(
            summary,
            [
                "dataset",
                "method",
                "variant",
                "repetitions",
                "MAE_mean",
                "MAE_std",
                "RMSE_mean",
                "RMSE_std",
                "mean_prediction_error_mean",
                "Spearman_mean",
            ],
        ),
        "",
        "## Interpretation cautions",
        (
            "The validation partitions contain only a small number of "
            "unique questions. Calibration parameters can therefore vary "
            "considerably across splits. Repeated splits reuse questions "
            "and are not independent external validation."
        ),
        (
            "The raw results in this experiment are evaluated on the same "
            "outer test questions as the calibrated versions, enabling "
            "within-split comparisons. The test results should not be used "
            "repeatedly to tune parameters or claim definitive superiority."
        ),
        (
            "Exp 019 uses inner cross-validation to select Ridge alpha, "
            "so its Ridge setup is not identical to the validation-tuned "
            "Ridge in Exp 016."
        ),
        "",
        "## Files generated",
        "- `exp019_run_metrics.csv`",
        "- `exp019_summary.csv`",
        "- `exp019_predictions.csv`",
        "- `exp019_calibration_parameters.csv`",
        "- `exp019_split_manifest.csv`",
        "- `exp019_ridge_inner_cv.csv`",
        "- `exp019_calibration_mae_comparison.png`",
    ]

    report_path = OUTPUT_DIR / "exp019_report.md"

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("EXPERIMENT 019 COMPLETED")
    print("=" * 76)
    print(f"Results: {OUTPUT_DIR}")
    print(f"Report: {report_path}")
    print(f"MAE plot: {plot_path}")
    print("\nNo source datasets or human annotations were modified.")
    print("Calibration was fitted from validation data only.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*constant.*",
        )
        main()