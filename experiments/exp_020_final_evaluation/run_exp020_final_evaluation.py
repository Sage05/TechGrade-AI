"""
TECHGRADE AI — EXPERIMENT 020
FINAL CONSOLIDATED EVALUATION

Purpose:
    Produce a reproducible consolidated evaluation of TechGradeAI's
    categorical classifier and numerical scoring methods.

Methods:
    1. Global mean baseline
    2. Predicted-category mean
    3. Probability-weighted scorer
    4. Ridge regression

Variants:
    - Raw
    - Validation-fitted bias correction
    - Validation-fitted affine calibration

Methodology:
    - Five question-level splits.
    - Ridge alpha selected by GroupKFold within training.
    - Calibration fitted on validation predictions only.
    - Test data used only for final evaluation.
    - Identical 0-10 clipping for every evaluated score.
    - Classification and numerical metrics reported separately.

Important:
    Repeated random splits reuse questions and some answers.
    These are exploratory repeated evaluations, not independent
    external validation datasets.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import GroupKFold

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

SCORE_MIN = 0.0
SCORE_MAX = 10.0

LABEL_ORDER = [
    "Correct",
    "Partially Correct",
    "Incorrect",
    "Irrelevant",
    "Contradictory",
]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "exp_020_final_evaluation"
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

SCORE_METHODS = [
    "Global mean baseline",
    "Predicted-category mean",
    "Probability-weighted scorer",
    "Ridge regression",
]

CALIBRATION_VARIANTS = [
    "raw",
    "bias_correction",
    "affine_calibration",
]


# ============================================================
# 2. HELPERS
# ============================================================

def safe_text(value):
    """Convert missing text values to empty strings."""
    if pd.isna(value):
        return ""
    return str(value)


def bounded_scores(scores):
    """
    Apply the same valid score interval to every scorer.

    Returns scores clipped to the 0-10 range.
    """
    values = np.asarray(scores, dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite numerical predictions were detected."
        )

    return np.clip(
        values,
        SCORE_MIN,
        SCORE_MAX,
    )


def build_features(frame, extractor):
    """Extract the project's existing 36-feature representation."""
    features = []

    for _, row in frame.iterrows():
        result = extractor.extract(
            question=safe_text(row["question_text"]),
            reference_answer=safe_text(row["model_answer"]),
            student_answer=safe_text(row["student_answer"]),
            required_concepts=safe_text(
                row["required_concepts"]
            ),
        )

        features.append(
            np.asarray(
                result.values,
                dtype=np.float32,
            )
        )

    if not features:
        raise ValueError(
            "Cannot extract features from an empty partition."
        )

    matrix = np.vstack(features)

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Feature extraction produced non-finite values."
        )

    return matrix


def verify_split(train, validation, test):
    """
    Verify question-level separation.

    The helper from the existing project is used in addition to
    explicit set checks.
    """
    no_leakage = verify_no_question_leakage(
        train,
        validation,
        test,
    )

    if not no_leakage:
        raise RuntimeError(
            "The existing split utility reported question leakage."
        )

    train_questions = set(
        train["question_text"].astype(str)
    )
    validation_questions = set(
        validation["question_text"].astype(str)
    )
    test_questions = set(
        test["question_text"].astype(str)
    )

    if train_questions & validation_questions:
        raise RuntimeError(
            "Training/validation question overlap detected."
        )

    if train_questions & test_questions:
        raise RuntimeError(
            "Training/test question overlap detected."
        )

    if validation_questions & test_questions:
        raise RuntimeError(
            "Validation/test question overlap detected."
        )


def choose_ridge_alpha(X, y, question_groups):
    """
    Select Ridge alpha using grouped CV inside the training partition.

    The outer validation and test data are not used to select alpha.
    """
    question_groups = np.asarray(
        question_groups,
        dtype=str,
    )

    unique_groups = np.unique(question_groups)

    n_splits = min(
        3,
        len(unique_groups),
    )

    if n_splits < 2:
        raise ValueError(
            "At least two unique training questions are required "
            "for grouped Ridge cross-validation."
        )

    cv = GroupKFold(
        n_splits=n_splits,
    )

    records = []

    for alpha in RIDGE_ALPHAS:
        fold_maes = []

        for fold, (train_indices, valid_indices) in enumerate(
            cv.split(
                X,
                y,
                groups=question_groups,
            ),
            start=1,
        ):
            model = Ridge(
                alpha=alpha,
            )

            model.fit(
                X[train_indices],
                y[train_indices],
            )

            predictions = model.predict(
                X[valid_indices]
            )

            mae = mean_absolute_error(
                y[valid_indices],
                predictions,
            )

            fold_maes.append(float(mae))

            records.append({
                "alpha": float(alpha),
                "fold": int(fold),
                "validation_mae": float(mae),
                "train_rows": len(train_indices),
                "validation_rows": len(valid_indices),
                "train_questions": len(
                    set(question_groups[train_indices])
                ),
                "validation_questions": len(
                    set(question_groups[valid_indices])
                ),
            })

        print(
            f"    alpha={alpha:g}: "
            f"inner grouped-CV MAE={np.mean(fold_maes):.4f}"
        )

    cv_frame = pd.DataFrame(records)

    mean_scores = (
        cv_frame.groupby("alpha")["validation_mae"]
        .mean()
        .sort_values()
    )

    selected_alpha = float(
        mean_scores.index[0]
    )

    return selected_alpha, cv_frame


# ============================================================
# 3. SCORER IMPLEMENTATION
# ============================================================

def generate_score_methods(
    classifier,
    ridge_model,
    feature_matrix,
    training_score_means,
    global_training_mean,
):
    """
    Generate the four scoring methods.

    The predicted-category and probability-weighted scorers use
    numerical means calculated from the training partition only.
    """
    predicted_labels = classifier.predict(
        feature_matrix
    )

    probabilities = classifier.predict_proba(
        feature_matrix
    )

    # Predicted-category mean.
    category_scores = np.asarray([
        training_score_means.get(
            label,
            global_training_mean,
        )
        for label in predicted_labels
    ], dtype=float)

    # Probability-weighted score.
    probability_scores = np.zeros(
        len(feature_matrix),
        dtype=float,
    )

    for index, label in enumerate(classifier.classes_):
        class_mean = training_score_means.get(
            label,
            global_training_mean,
        )

        probability_scores += (
            probabilities[:, index] * class_mean
        )

    # Ridge regression.
    ridge_scores = ridge_model.predict(
        feature_matrix
    )

    # Global mean baseline.
    global_scores = np.full(
        len(feature_matrix),
        global_training_mean,
        dtype=float,
    )

    return {
        "Global mean baseline": global_scores,
        "Predicted-category mean": category_scores,
        "Probability-weighted scorer": probability_scores,
        "Ridge regression": ridge_scores,
    }, predicted_labels, probabilities


# ============================================================
# 4. CALIBRATION
# ============================================================

def fit_calibration(
    y_validation,
    bounded_validation_scores,
    method,
):
    """
    Fit a calibration transform from validation data only.

    The inputs are already clipped to 0-10, matching the output
    range of the corresponding raw scorer during evaluation.
    """
    y = np.asarray(
        y_validation,
        dtype=float,
    )

    x = bounded_scores(
        bounded_validation_scores
    )

    if len(y) != len(x) or len(y) == 0:
        raise ValueError(
            "Calibration arrays have incompatible lengths."
        )

    if method == "bias_correction":
        slope = 1.0
        intercept = float(
            np.mean(y - x)
        )

    elif method == "affine_calibration":
        x_mean = float(
            np.mean(x)
        )
        y_mean = float(
            np.mean(y)
        )

        centered_x = x - x_mean

        variance = float(
            np.mean(centered_x ** 2)
        )

        if variance < 1e-12:
            slope = 0.0

        else:
            covariance = float(
                np.mean(
                    centered_x * (y - y_mean)
                )
            )

            # Do not allow calibration to reverse score ordering.
            slope = max(
                0.0,
                covariance / variance,
            )

        intercept = y_mean - slope * x_mean

    else:
        raise ValueError(
            f"Unknown calibration method: {method}"
        )

    return {
        "method": method,
        "slope": float(slope),
        "intercept": float(intercept),
        "validation_rows": int(len(y)),
    }


def apply_calibration(scores, parameters):
    """
    Apply calibration and then enforce the same 0-10 score bounds.
    """
    scores = bounded_scores(scores)

    transformed = (
        parameters["slope"] * scores
        + parameters["intercept"]
    )

    return bounded_scores(transformed)


# ============================================================
# 5. METRICS
# ============================================================

def numerical_metrics(actual, predicted, score_before_clip=None):
    """Numerical evaluation metrics, including clipping diagnostics."""
    y = np.asarray(actual, dtype=float)

    prediction = bounded_scores(
        predicted
    )

    errors = prediction - y
    absolute_errors = np.abs(errors)

    if (
        len(y) > 1
        and np.std(y) > 0
        and np.std(prediction) > 0
    ):
        spearman = float(
            pd.Series(y).corr(
                pd.Series(prediction),
                method="spearman",
            )
        )
    else:
        spearman = np.nan

    total_sum_squares = float(
        np.sum((y - np.mean(y)) ** 2)
    )

    if total_sum_squares > 0:
        r2 = float(
            1.0
            - np.sum(errors ** 2) / total_sum_squares
        )
    else:
        r2 = np.nan

    if score_before_clip is None:
        clipped_pct = 0.0
    else:
        before = np.asarray(
            score_before_clip,
            dtype=float,
        )

        clipped_pct = float(
            np.mean(
                (before < SCORE_MIN)
                | (before > SCORE_MAX)
            ) * 100
        )

    return {
        "n_test_rows": int(len(y)),
        "MAE": float(
            np.mean(absolute_errors)
        ),
        "RMSE": float(
            np.sqrt(np.mean(errors ** 2))
        ),
        "R2": r2,
        "Spearman": spearman,
        "mean_prediction_error": float(
            np.mean(errors)
        ),
        "median_absolute_error": float(
            np.median(absolute_errors)
        ),
        "within_1_point_pct": float(
            np.mean(absolute_errors <= 1.0) * 100
        ),
        "within_2_points_pct": float(
            np.mean(absolute_errors <= 2.0) * 100
        ),
        "predictions_at_zero_pct": float(
            np.mean(prediction == SCORE_MIN) * 100
        ),
        "predictions_at_ten_pct": float(
            np.mean(prediction == SCORE_MAX) * 100
        ),
        "score_clipped_pct": clipped_pct,
    }


def markdown_table(frame, columns):
    """Build a Markdown table without an optional tabulate package."""
    if frame.empty:
        return "_No rows available._"

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(
            ["---"] * len(columns)
        ) + " |",
    ]

    for _, row in frame[columns].iterrows():
        values = []

        for column in columns:
            value = row[column]

            if pd.isna(value):
                values.append("—")

            elif isinstance(
                value,
                (float, np.floating),
            ):
                values.append(
                    f"{value:.3f}"
                )

            elif isinstance(
                value,
                (int, np.integer),
            ):
                values.append(
                    str(int(value))
                )

            else:
                values.append(
                    str(value)
                    .replace("|", "\\|")
                    .replace("\n", " ")
                )

        lines.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(lines)


# ============================================================
# 6. MAIN EXPERIMENT
# ============================================================

def main():
    print("=" * 76)
    print("TECHGRADE AI — EXPERIMENT 020")
    print("FINAL CONSOLIDATED EVALUATION")
    print("=" * 76)

    run_metric_rows = []
    prediction_rows = []
    classification_rows = []
    per_class_rows = []
    confusion_matrices = {}
    calibration_rows = []
    ridge_cv_frames = []
    split_rows = []

    for dataset_name, csv_path in DATASETS.items():

        print("\n" + "=" * 76)
        print(f"DATASET: {dataset_name}")
        print("=" * 76)

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}"
            )

        df = pd.read_csv(
            csv_path
        ).copy()

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{dataset_name} is missing required columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        df = df.reset_index(
            drop=True
        )

        # Preserve a stable identifier across split repetitions.
        df["_source_row_id"] = np.arange(
            len(df)
        )

        for column in [
            "question_text",
            "model_answer",
            "student_answer",
            "required_concepts",
            "label",
        ]:
            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        df["human_score"] = pd.to_numeric(
            df["human_score"],
            errors="coerce",
        )

        if df[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError(
                f"{dataset_name} contains missing required values."
            )

        if not df["human_score"].between(
            SCORE_MIN,
            SCORE_MAX,
        ).all():
            raise ValueError(
                f"{dataset_name} has human scores outside 0-10."
            )

        if (df["label"] == "").any():
            raise ValueError(
                f"{dataset_name} has empty categorical labels."
            )

        print(f"Rows: {len(df)}")
        print(
            f"Unique questions: "
            f"{df['question_text'].nunique()}"
        )

        print("Loading embedding model...")
        embedding_model = EmbeddingBaseline()

        print("Loading NLI model...")
        nli_model = NLIBaseline()

        print("Loading concept matcher...")
        concept_matcher = ConceptMatcher()

        for seed in RANDOM_STATES:

            print("\n" + "-" * 76)
            print(
                f"{dataset_name} | random_state={seed}"
            )
            print("-" * 76)

            train, validation, test = question_level_split(
                df,
                random_state=seed,
            )

            train = train.copy().reset_index(drop=True)
            validation = validation.copy().reset_index(drop=True)
            test = test.copy().reset_index(drop=True)

            verify_split(
                train,
                validation,
                test,
            )

            print(
                f"Train: {len(train)} rows / "
                f"{train['question_text'].nunique()} questions"
            )
            print(
                f"Validation: {len(validation)} rows / "
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
                "train_questions": (
                    train["question_text"].nunique()
                ),
                "validation_rows": len(validation),
                "validation_questions": (
                    validation["question_text"].nunique()
                ),
                "test_rows": len(test),
                "test_questions": (
                    test["question_text"].nunique()
                ),
                "question_leakage": False,
            })

            # ------------------------------------------------
            # Fit TF-IDF using training data only.
            # ------------------------------------------------

            tfidf_model = TfidfBaseline()

            tfidf_corpus = (
                train["question_text"].tolist()
                + train["model_answer"].tolist()
                + train["student_answer"].tolist()
            )

            tfidf_model.fit(
                tfidf_corpus
            )

            extractor = FeatureExtractor(
                tfidf_model=tfidf_model,
                embedding_model=embedding_model,
                nli_model=nli_model,
                concept_matcher=concept_matcher,
            )

            # ------------------------------------------------
            # Extract features.
            # ------------------------------------------------

            print("Extracting training features...")
            X_train = build_features(
                train,
                extractor,
            )

            print("Extracting validation features...")
            X_validation = build_features(
                validation,
                extractor,
            )

            print("Extracting test features...")
            X_test = build_features(
                test,
                extractor,
            )

            if not (
                len(X_train) == len(train)
                and len(X_validation) == len(validation)
                and len(X_test) == len(test)
            ):
                raise RuntimeError(
                    "Feature and target row counts do not match."
                )

            print(
                f"Feature count: {X_train.shape[1]}"
            )

            y_train = train["human_score"].to_numpy(
                dtype=float
            )

            y_validation = validation["human_score"].to_numpy(
                dtype=float
            )

            y_test = test["human_score"].to_numpy(
                dtype=float
            )

            # ------------------------------------------------
            # Train the categorical classifier.
            # ------------------------------------------------

            classifier = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            )

            classifier.fit(
                X_train,
                train["label"].to_numpy(),
            )

            test_labels = classifier.predict(
                X_test
            )

            class_order = (
                LABEL_ORDER
                + sorted(
                    (
                        set(train["label"])
                        | set(test["label"])
                        | set(test_labels)
                    ) - set(LABEL_ORDER)
                )
            )

            # ------------------------------------------------
            # Score mapping from training partition only.
            # ------------------------------------------------

            training_score_means = (
                train.groupby("label")["human_score"]
                .mean()
                .to_dict()
            )

            global_training_mean = float(
                np.mean(y_train)
            )

            # ------------------------------------------------
            # Select Ridge alpha using grouped inner CV.
            # ------------------------------------------------

            print(
                "\nSelecting Ridge alpha within training..."
            )

            selected_alpha, cv_frame = choose_ridge_alpha(
                X_train,
                y_train,
                train["question_text"].astype(str).to_numpy(),
            )

            print(
                f"Selected Ridge alpha: {selected_alpha:g}"
            )

            if not cv_frame.empty:
                cv_frame["dataset"] = dataset_name
                cv_frame["random_state"] = seed
                ridge_cv_frames.append(
                    cv_frame
                )

            ridge_model = Ridge(
                alpha=selected_alpha
            )

            ridge_model.fit(
                X_train,
                y_train,
            )

            # ------------------------------------------------
            # Classification test metrics.
            # ------------------------------------------------

            accuracy = accuracy_score(
                test["label"],
                test_labels,
            )

            macro_f1 = f1_score(
                test["label"],
                test_labels,
                average="macro",
                labels=class_order,
                zero_division=0,
            )

            weighted_f1 = f1_score(
                test["label"],
                test_labels,
                average="weighted",
                labels=class_order,
                zero_division=0,
            )

            print(
                f"\nClassification accuracy: {accuracy:.4f}"
            )
            print(
                f"Classification macro F1: {macro_f1:.4f}"
            )
            print(
                f"Classification weighted F1: {weighted_f1:.4f}"
            )

            classification_rows.append({
                "dataset": dataset_name,
                "random_state": seed,
                "test_rows": len(test),
                "test_questions": (
                    test["question_text"].nunique()
                ),
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "classification_error_pct": float(
                    np.mean(
                        test["label"].to_numpy()
                        != test_labels
                    ) * 100
                ),
            })

            class_report = classification_report(
                test["label"],
                test_labels,
                labels=class_order,
                output_dict=True,
                zero_division=0,
            )

            for label in class_order:
                report_item = class_report.get(
                    label,
                    {},
                )

                per_class_rows.append({
                    "dataset": dataset_name,
                    "random_state": seed,
                    "label": label,
                    "precision": report_item.get(
                        "precision",
                        np.nan,
                    ),
                    "recall": report_item.get(
                        "recall",
                        np.nan,
                    ),
                    "f1_score": report_item.get(
                        "f1-score",
                        np.nan,
                    ),
                    "support": report_item.get(
                        "support",
                        0,
                    ),
                })

            cm = confusion_matrix(
                test["label"],
                test_labels,
                labels=class_order,
            )

            if dataset_name not in confusion_matrices:
                confusion_matrices[dataset_name] = {
                    "labels": class_order,
                    "counts": cm.astype(int),
                }
            else:
                confusion_matrices[dataset_name]["counts"] += cm

            # ------------------------------------------------
            # Generate unbounded scores before common clipping.
            # ------------------------------------------------

            validation_scores, _, _ = generate_score_methods(
                classifier,
                ridge_model,
                X_validation,
                training_score_means,
                global_training_mean,
            )

            test_scores, _, _ = generate_score_methods(
                classifier,
                ridge_model,
                X_test,
                training_score_means,
                global_training_mean,
            )

            # Every method's raw output is clipped by the same rule.
            # Calibration learns from bounded validation predictions.
            for method in SCORE_METHODS:

                raw_validation_unbounded = np.asarray(
                    validation_scores[method],
                    dtype=float,
                )

                raw_test_unbounded = np.asarray(
                    test_scores[method],
                    dtype=float,
                )

                raw_validation_bounded = bounded_scores(
                    raw_validation_unbounded
                )

                raw_test_bounded = bounded_scores(
                    raw_test_unbounded
                )

                if method == "Global mean baseline":
                    variants = ["raw"]

                    parameters_by_variant = {
                        "raw": {
                            "method": "raw",
                            "slope": 1.0,
                            "intercept": 0.0,
                            "validation_rows": len(validation),
                        }
                    }

                else:
                    variants = CALIBRATION_VARIANTS

                    bias_parameters = fit_calibration(
                        y_validation,
                        raw_validation_bounded,
                        "bias_correction",
                    )

                    affine_parameters = fit_calibration(
                        y_validation,
                        raw_validation_bounded,
                        "affine_calibration",
                    )

                    parameters_by_variant = {
                        "raw": {
                            "method": "raw",
                            "slope": 1.0,
                            "intercept": 0.0,
                            "validation_rows": len(validation),
                        },
                        "bias_correction": bias_parameters,
                        "affine_calibration": affine_parameters,
                    }

                for variant in variants:
                    parameters = parameters_by_variant[variant]

                    if variant == "raw":
                        before_clip = raw_test_unbounded
                        final_predictions = raw_test_bounded

                    else:
                        # Calibrators receive bounded raw scores so
                        # every method follows the same score interface.
                        before_clip = (
                            parameters["slope"] * raw_test_bounded
                            + parameters["intercept"]
                        )

                        final_predictions = bounded_scores(
                            before_clip
                        )

                    metrics = numerical_metrics(
                        y_test,
                        final_predictions,
                        score_before_clip=before_clip,
                    )

                    run_metric_rows.append({
                        "dataset": dataset_name,
                        "random_state": seed,
                        "method": method,
                        "variant": variant,
                        "ridge_alpha": selected_alpha,
                        "classification_accuracy": float(
                            accuracy
                        ),
                        "classification_macro_f1": float(
                            macro_f1
                        ),
                        "classification_weighted_f1": float(
                            weighted_f1
                        ),
                        **metrics,
                    })

                    calibration_rows.append({
                        "dataset": dataset_name,
                        "random_state": seed,
                        "method": method,
                        "variant": variant,
                        "slope": float(
                            parameters["slope"]
                        ),
                        "intercept": float(
                            parameters["intercept"]
                        ),
                        "validation_rows": int(
                            parameters["validation_rows"]
                        ),
                        "validation_questions": int(
                            validation["question_text"].nunique()
                        ),
                        "selected_ridge_alpha": selected_alpha,
                    })

                    for index, (_, row) in enumerate(
                        test.iterrows()
                    ):
                        prediction_rows.append({
                            "dataset": dataset_name,
                            "random_state": seed,
                            "source_row_id": int(
                                row["_source_row_id"]
                            ),
                            "question_text": row["question_text"],
                            "student_answer": row["student_answer"],
                            "actual_label": row["label"],
                            "predicted_label": str(
                                test_labels[index]
                            ),
                            "classification_correct": (
                                str(test_labels[index])
                                == str(row["label"])
                            ),
                            "actual_human_score": float(
                                y_test[index]
                            ),
                            "method": method,
                            "variant": variant,
                            "raw_score_unbounded": float(
                                raw_test_unbounded[index]
                            ),
                            "raw_score_bounded": float(
                                raw_test_bounded[index]
                            ),
                            "score_before_final_clipping": float(
                                before_clip[index]
                            ),
                            "predicted_score": float(
                                final_predictions[index]
                            ),
                            "residual": float(
                                final_predictions[index]
                                - y_test[index]
                            ),
                            "absolute_error": float(
                                abs(
                                    final_predictions[index]
                                    - y_test[index]
                                )
                            ),
                            "calibration_slope": float(
                                parameters["slope"]
                            ),
                            "calibration_intercept": float(
                                parameters["intercept"]
                            ),
                            "ridge_alpha": selected_alpha,
                        })

                    print(
                        f"{method:30s} | "
                        f"{variant:20s} | "
                        f"MAE={metrics['MAE']:.4f} | "
                        f"RMSE={metrics['RMSE']:.4f} | "
                        f"bias={metrics['mean_prediction_error']:+.4f}"
                    )

    # ========================================================
    # 7. SAVE RAW RESULT TABLES
    # ========================================================

    run_metrics = pd.DataFrame(
        run_metric_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    classification_metrics = pd.DataFrame(
        classification_rows
    )

    per_class_metrics = pd.DataFrame(
        per_class_rows
    )

    calibration_parameters = pd.DataFrame(
        calibration_rows
    )

    split_manifest = pd.DataFrame(
        split_rows
    )

    ridge_cv = (
        pd.concat(
            ridge_cv_frames,
            ignore_index=True,
        )
        if ridge_cv_frames
        else pd.DataFrame()
    )

    run_metrics.to_csv(
        OUTPUT_DIR / "exp020_run_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        OUTPUT_DIR / "exp020_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    classification_metrics.to_csv(
        OUTPUT_DIR / "exp020_classification_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_class_metrics.to_csv(
        OUTPUT_DIR / "exp020_per_class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    calibration_parameters.to_csv(
        OUTPUT_DIR / "exp020_calibration_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_manifest.to_csv(
        OUTPUT_DIR / "exp020_split_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ridge_cv.to_csv(
        OUTPUT_DIR / "exp020_ridge_inner_cv.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 8. CONFUSION MATRICES
    # ========================================================

    for dataset, record in confusion_matrices.items():
        labels = record["labels"]
        counts = record["counts"]

        count_df = pd.DataFrame(
            counts,
            index=labels,
            columns=labels,
        )

        count_df.index.name = "actual_label"
        count_df.columns.name = "predicted_label"

        safe_name = dataset.lower()

        count_df.to_csv(
            OUTPUT_DIR
            / f"exp020_confusion_counts_{safe_name}.csv",
            encoding="utf-8-sig",
        )

        row_sums = counts.sum(
            axis=1,
            keepdims=True,
        )

        row_percent = np.divide(
            counts,
            row_sums,
            out=np.zeros_like(
                counts,
                dtype=float,
            ),
            where=row_sums != 0,
        ) * 100

        percent_df = pd.DataFrame(
            row_percent,
            index=labels,
            columns=labels,
        )

        percent_df.index.name = "actual_label"
        percent_df.columns.name = "predicted_label"

        percent_df.to_csv(
            OUTPUT_DIR
            / f"exp020_confusion_row_percent_{safe_name}.csv",
            encoding="utf-8-sig",
        )

    # ========================================================
    # 9. MEAN AND STANDARD DEVIATION ACROSS SPLITS
    # ========================================================

    metric_columns = [
        "MAE",
        "RMSE",
        "R2",
        "Spearman",
        "mean_prediction_error",
        "median_absolute_error",
        "within_1_point_pct",
        "within_2_points_pct",
        "predictions_at_zero_pct",
        "predictions_at_ten_pct",
        "score_clipped_pct",
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
            "repetitions": int(
                group["random_state"].nunique()
            ),
        }

        for metric in metric_columns:
            result[f"{metric}_mean"] = group[metric].mean()
            result[f"{metric}_std"] = group[metric].std(
                ddof=1
            )

        summary_rows.append(result)

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        ["dataset", "MAE_mean"],
        ascending=[True, True],
    ).reset_index(drop=True)

    summary.to_csv(
        OUTPUT_DIR / "exp020_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    classification_summary = (
        classification_metrics.groupby(
            "dataset",
            sort=True,
        )
        .agg(
            repetitions=("random_state", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std"),
        )
        .reset_index()
    )

    classification_summary.to_csv(
        OUTPUT_DIR / "exp020_classification_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 10. PLOT NUMERICAL COMPARISON
    # ========================================================

    datasets_found = sorted(
        summary["dataset"].unique()
    )

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

        means = subset["MAE_mean"].to_numpy(
            dtype=float
        )

        stds = (
            subset["MAE_std"]
            .fillna(0)
            .to_numpy(dtype=float)
        )

        positions = np.arange(
            len(subset)
        )

        ax.bar(
            positions,
            means,
            yerr=stds,
            capsize=3,
        )

        ax.set_xticks(
            positions
        )

        ax.set_xticklabels(
            labels,
            rotation=40,
            ha="right",
            fontsize=8,
        )

        ax.set_ylabel(
            "MAE (lower is better)"
        )

        ax.set_title(
            f"{dataset}: mean MAE across splits"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

    fig.suptitle(
        "Exp 020 — Final Scorer Comparison",
        y=1.02,
    )

    fig.tight_layout()

    mae_plot = (
        OUTPUT_DIR
        / "exp020_final_mae_comparison.png"
    )

    fig.savefig(
        mae_plot,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # 11. SAVE CONSOLIDATED REPORT
    # ========================================================

    report_lines = [
        "# Experiment 020 — Final Consolidated Evaluation",
        "",
        "## Objective",
        (
            "Consolidate categorical classification and numerical "
            "scoring results from the current TechGradeAI datasets."
        ),
        "",
        "## Evaluation design",
        "- Datasets: Operating Systems and DBMS.",
        f"- Random states: {RANDOM_STATES}.",
        "- Splits are question-level; leakage checks passed for all runs.",
        "- Ridge alpha was selected using GroupKFold within the training partition.",
        "- Training-fold score means were used by category-dependent scorers.",
        "- Calibration parameters were fitted using validation predictions only.",
        "- All final numerical predictions were clipped to the same 0–10 interval.",
        "- Test human scores were not used to fit models or calibrators.",
        "",
        "## Categorical classification summary",
        markdown_table(
            classification_summary,
            [
                "dataset",
                "repetitions",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "weighted_f1_mean",
                "weighted_f1_std",
            ],
        ),
        "",
        "## Numerical scorer summary",
        markdown_table(
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
                "R2_mean",
                "Spearman_mean",
                "mean_prediction_error_mean",
                "score_clipped_pct_mean",
            ],
        ),
        "",
        "## Interpretation",
        (
            "Lower MAE and RMSE indicate smaller numerical errors. "
            "Mean prediction error is predicted score minus human score; "
            "positive values indicate overprediction and negative values "
            "indicate underprediction. Spearman correlation describes "
            "rank-order agreement."
        ),
        (
            "The method rows are pre-specified comparisons. The method "
            "with the lowest observed test MAE is not automatically "
            "declared a universally best model; repeated test comparisons "
            "can introduce selection bias."
        ),
        "",
        "## Important limitations",
        (
            "The datasets contain only 14 OS questions and 15 DBMS "
            "questions. The five repeated splits reuse questions and "
            "some answer rows. The resulting standard deviations describe "
            "split variability, not uncertainty from five independent "
            "external datasets."
        ),
        (
            "Calibration validation sets are small, so calibration "
            "parameters may be unstable. The results support exploratory "
            "comparisons but do not establish production readiness or "
            "generalisation to other subjects."
        ),
        "",
        "## Output files",
        "- `exp020_run_metrics.csv`",
        "- `exp020_summary.csv`",
        "- `exp020_predictions.csv`",
        "- `exp020_classification_metrics.csv`",
        "- `exp020_classification_summary.csv`",
        "- `exp020_per_class_metrics.csv`",
        "- `exp020_calibration_parameters.csv`",
        "- `exp020_split_manifest.csv`",
        "- `exp020_ridge_inner_cv.csv`",
        "- `exp020_confusion_counts_os.csv`",
        "- `exp020_confusion_counts_dbms.csv`",
        "- `exp020_confusion_row_percent_os.csv`",
        "- `exp020_confusion_row_percent_dbms.csv`",
        "- `exp020_final_mae_comparison.png`",
        "",
        "## Final note",
        (
            "This experiment concludes the current evaluation phase. "
            "Further work should prioritise expanding the number and "
            "diversity of questions, validating human-score agreement, "
            "and evaluating on genuinely independent questions."
        ),
    ]

    report_path = (
        OUTPUT_DIR
        / "exp020_final_report.md"
    )

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # 12. PRINT FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 76)
    print("EXP 020 — FINAL CLASSIFICATION SUMMARY")
    print("=" * 76)

    print(
        classification_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 76)
    print("EXP 020 — FINAL NUMERICAL SUMMARY")
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

    print("\n" + "=" * 76)
    print("EXPERIMENT 020 COMPLETED")
    print("=" * 76)

    print(f"Results directory: {OUTPUT_DIR}")
    print(f"Final report: {report_path}")
    print(f"MAE comparison: {mae_plot}")

    print("\nAll scoring methods use the same 0–10 output range.")
    print("Calibration was fitted on validation data only.")
    print("No source datasets or human annotations were modified.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*constant.*",
        )
        main()