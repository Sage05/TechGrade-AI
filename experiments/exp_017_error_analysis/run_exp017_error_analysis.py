"""
Experiment 017: Error Analysis and Scorer Calibration

Uses the saved predictions from Experiment 016.
Does not retrain models or modify source datasets.

Analyses:
1. Run-level and pooled scoring metrics.
2. Overprediction and underprediction.
3. Errors by actual categorical label.
4. Errors by predicted categorical label.
5. Errors by actual human-score band.
6. Errors grouped by question.
7. Repeated errors for individual answers across random splits.
8. Predictions outside the 0-10 scoring range.
9. Score compression and descriptive calibration diagnostics.
10. Worst consistently scored individual answers.

Residual convention:
    residual = predicted_score - actual_human_score

Positive residual = overprediction.
Negative residual = underprediction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "exp_016_scoring_robustness"
    / "results"
    / "exp016_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "exp_017_error_analysis"
    / "results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_COLUMNS = {
    "global_mean_baseline": "Global mean baseline",
    "predicted_category_mean": "Predicted-category mean",
    "probability_weighted_score": "Probability-weighted scorer",
    "ridge_regression_score": "Ridge regression",
}

SCORE_BAND_LABELS = [
    "0–2",
    ">2–4",
    ">4–6",
    ">6–8",
    ">8–10",
]

SCORE_BAND_EDGES = [-0.001, 2, 4, 6, 8, 10]


# ============================================================
# 2. LOAD AND VALIDATE INPUT
# ============================================================

print("=" * 72)
print("EXPERIMENT 017 — ERROR ANALYSIS AND SCORER CALIBRATION")
print("=" * 72)

print(f"\nInput file:\n{INPUT_PATH}")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "\nCould not find the Exp 016 predictions file.\n"
        f"Expected location: {INPUT_PATH}\n\n"
        "Make sure Experiment 016 completed successfully and that "
        "exp016_predictions.csv exists in its results folder."
    )

df = pd.read_csv(INPUT_PATH)

df.columns = [str(column).strip() for column in df.columns]

required_columns = [
    "dataset",
    "actual_human_score",
    *METHOD_COLUMNS.keys(),
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nThe Exp 016 prediction file is missing required columns:\n"
        + "\n".join(f" - {column}" for column in missing_columns)
        + "\n\nAvailable columns:\n"
        + ", ".join(df.columns)
    )

# Add optional columns if they are unavailable, so that the main
# numerical error analysis can still run.
OPTIONAL_DEFAULTS = {
    "random_state": -1,
    "source_csv_line": "",
    "question_text": "[Question unavailable]",
    "student_answer": "[Answer unavailable]",
    "actual_label": "[Label unavailable]",
    "predicted_label": "[Predicted label unavailable]",
}

for column, default_value in OPTIONAL_DEFAULTS.items():
    if column not in df.columns:
        df[column] = default_value

# Normalize the metadata used in grouping and reporting.
df["dataset"] = df["dataset"].fillna("Unknown dataset").astype(str)

df["question_text"] = (
    df["question_text"]
    .fillna("[Question unavailable]")
    .astype(str)
)

df["student_answer"] = (
    df["student_answer"]
    .fillna("[Answer unavailable]")
    .astype(str)
)

df["actual_label"] = (
    df["actual_label"]
    .fillna("[Label unavailable]")
    .astype(str)
)

df["predicted_label"] = (
    df["predicted_label"]
    .fillna("[Predicted label unavailable]")
    .astype(str)
)

df["actual_human_score"] = pd.to_numeric(
    df["actual_human_score"],
    errors="coerce",
)

df["random_state"] = pd.to_numeric(
    df["random_state"],
    errors="coerce",
).fillna(-1)

for column in METHOD_COLUMNS:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

invalid_actual_scores = (
    df["actual_human_score"].isna()
    | ~df["actual_human_score"].between(0, 10)
)

if invalid_actual_scores.any():
    print(
        "\nWARNING: Removing rows with missing or out-of-range "
        "actual human scores:"
    )
    print(f"Rows affected: {int(invalid_actual_scores.sum())}")

    df = df.loc[~invalid_actual_scores].copy()

if df.empty:
    raise ValueError(
        "No usable rows remain after validating actual human scores."
    )

# Build a stable identifier for the original answer across repeated
# random-state runs. Prefer the original source CSV row identifier.
source_line = (
    df["source_csv_line"]
    .fillna("")
    .astype(str)
    .str.strip()
)

has_source_line = (
    source_line.ne("")
    & source_line.str.lower().ne("nan")
)

fallback_key = (
    df["dataset"].astype(str)
    + "|question:"
    + df["question_text"].astype(str)
    + "|answer:"
    + df["student_answer"].astype(str)
)

df["_row_key"] = np.where(
    has_source_line,
    df["dataset"].astype(str) + "|line:" + source_line,
    fallback_key,
)

print(f"\nRows loaded: {len(df)}")
print(f"Datasets: {', '.join(sorted(df['dataset'].unique()))}")
print(f"Unique answer rows: {df['_row_key'].nunique()}")
print(f"Random states: {df['random_state'].nunique()}")

print("\nPrediction methods:")
for column, display_name in METHOD_COLUMNS.items():
    print(f"  - {display_name}: {column}")


# ============================================================
# 3. CREATE LONG-FORM PREDICTIONS AND RESIDUALS
# ============================================================

metadata_columns = [
    "dataset",
    "random_state",
    "source_csv_line",
    "_row_key",
    "question_text",
    "student_answer",
    "actual_label",
    "predicted_label",
    "actual_human_score",
]

long_parts = []

for prediction_column, method_name in METHOD_COLUMNS.items():
    part = df[metadata_columns + [prediction_column]].copy()

    part = part.rename(
        columns={prediction_column: "prediction"}
    )

    part["method"] = method_name

    part["prediction"] = pd.to_numeric(
        part["prediction"],
        errors="coerce",
    )

    invalid_predictions = part["prediction"].isna()

    if invalid_predictions.any():
        print(
            f"\nWARNING: {int(invalid_predictions.sum())} rows "
            f"for {method_name} have missing numerical predictions."
        )

    part = part.dropna(
        subset=["actual_human_score", "prediction"]
    ).copy()

    # Positive residual = the model assigns too high a score.
    part["residual"] = (
        part["prediction"] - part["actual_human_score"]
    )

    part["absolute_error"] = part["residual"].abs()

    part["squared_error"] = part["residual"] ** 2

    part["overprediction"] = part["residual"] > 0

    part["underprediction"] = part["residual"] < 0

    part["within_1_point"] = (
        part["absolute_error"] <= 1.0
    )

    part["within_2_points"] = (
        part["absolute_error"] <= 2.0
    )

    long_parts.append(part)

long_df = pd.concat(
    long_parts,
    ignore_index=True,
)

if long_df.empty:
    raise ValueError(
        "No valid numerical predictions were available for analysis."
    )

# Create score bands based on the actual human score.
long_df["actual_score_band"] = pd.cut(
    long_df["actual_human_score"],
    bins=SCORE_BAND_EDGES,
    labels=SCORE_BAND_LABELS,
    include_lowest=True,
).astype("string")

residuals_path = OUTPUT_DIR / "exp017_residuals_long.csv"

long_df.to_csv(
    residuals_path,
    index=False,
    encoding="utf-8-sig",
)

print(f"\nSaved residual-level data: {residuals_path}")


# ============================================================
# 4. METRIC FUNCTIONS
# ============================================================

def calculate_metrics(group):
    """
    Calculate scoring metrics for a group of predictions.

    R2 and Spearman are descriptive metrics and may be undefined for
    groups with no actual-score variation or no prediction variation.
    """

    actual = group["actual_human_score"].to_numpy(dtype=float)

    predicted = group["prediction"].to_numpy(dtype=float)

    errors = predicted - actual

    absolute_errors = np.abs(errors)

    squared_errors = errors ** 2

    actual_std = float(np.std(actual, ddof=0))

    predicted_std = float(np.std(predicted, ddof=0))

    total_sum_squares = float(
        np.sum((actual - np.mean(actual)) ** 2)
    )

    if total_sum_squares > 0:
        r2 = float(
            1 - np.sum(squared_errors) / total_sum_squares
        )
    else:
        r2 = np.nan

    spearman = pd.Series(actual).corr(
        pd.Series(predicted),
        method="spearman",
    )

    if actual_std > 0:
        spread_ratio = predicted_std / actual_std
    else:
        spread_ratio = np.nan

    return {
        "n_prediction_occurrences": int(len(group)),
        "unique_test_rows": int(group["_row_key"].nunique()),
        "unique_questions": int(group["question_text"].nunique()),
        "n_random_states": int(group["random_state"].nunique()),

        "MAE": float(np.mean(absolute_errors)),
        "RMSE": float(np.sqrt(np.mean(squared_errors))),
        "R2": float(r2),
        "Spearman": float(spearman),

        "mean_prediction_error": float(np.mean(errors)),
        "median_prediction_error": float(np.median(errors)),

        "overprediction_pct": float(
            np.mean(errors > 0) * 100
        ),
        "underprediction_pct": float(
            np.mean(errors < 0) * 100
        ),
        "exact_prediction_pct": float(
            np.mean(errors == 0) * 100
        ),

        "within_1_point_pct": float(
            np.mean(absolute_errors <= 1.0) * 100
        ),
        "within_2_points_pct": float(
            np.mean(absolute_errors <= 2.0) * 100
        ),

        "actual_mean": float(np.mean(actual)),
        "prediction_mean": float(np.mean(predicted)),

        "actual_std": actual_std,
        "prediction_std": predicted_std,
        "prediction_to_actual_std_ratio": float(spread_ratio),

        "actual_min": float(np.min(actual)),
        "actual_max": float(np.max(actual)),

        "prediction_min": float(np.min(predicted)),
        "prediction_max": float(np.max(predicted)),

        "predictions_below_zero_pct": float(
            np.mean(predicted < 0) * 100
        ),
        "predictions_above_10_pct": float(
            np.mean(predicted > 10) * 100
        ),
    }


def grouped_metrics(data, group_columns):
    """
    Group a dataframe and calculate metrics for every group.

    A manual grouping loop is used instead of newer pandas-specific
    groupby.apply behaviour, improving compatibility across versions.
    """

    records = []

    grouped = data.groupby(
        group_columns,
        dropna=False,
        sort=False,
    )

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        record = dict(zip(group_columns, group_key))

        record.update(calculate_metrics(group))

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 5. OVERALL POOLED PERFORMANCE
# ============================================================

pooled_summary = grouped_metrics(
    long_df,
    ["dataset", "method"],
)

pooled_summary = pooled_summary.sort_values(
    ["dataset", "MAE"],
    ascending=[True, True],
).reset_index(drop=True)

pooled_path = OUTPUT_DIR / "exp017_pooled_method_metrics.csv"

pooled_summary.to_csv(
    pooled_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 6. RUN-LEVEL PERFORMANCE
# ============================================================

# Each random state is evaluated separately first. We then calculate
# the mean and standard deviation across these runs.
#
# This distinguishes "average across runs" from metrics pooled across
# all test-prediction occurrences.

run_metrics = grouped_metrics(
    long_df,
    ["dataset", "random_state", "method"],
)

run_metrics = run_metrics.sort_values(
    ["dataset", "random_state", "method"]
).reset_index(drop=True)

run_metrics_path = OUTPUT_DIR / "exp017_run_level_metrics.csv"

run_metrics.to_csv(
    run_metrics_path,
    index=False,
    encoding="utf-8-sig",
)

run_summary_records = []

run_metric_columns = [
    "MAE",
    "RMSE",
    "R2",
    "Spearman",
    "mean_prediction_error",
    "overprediction_pct",
    "underprediction_pct",
    "within_1_point_pct",
    "within_2_points_pct",
    "prediction_to_actual_std_ratio",
    "predictions_below_zero_pct",
    "predictions_above_10_pct",
]

for (dataset_name, method_name), group in run_metrics.groupby(
    ["dataset", "method"],
    dropna=False,
    sort=False,
):
    record = {
        "dataset": dataset_name,
        "method": method_name,
        "repetitions": int(group["random_state"].nunique()),
    }

    for metric in run_metric_columns:
        record[f"{metric}_mean"] = group[metric].mean()

        record[f"{metric}_std"] = group[metric].std(ddof=1)

    run_summary_records.append(record)

run_summary = pd.DataFrame(run_summary_records)

run_summary = run_summary.sort_values(
    ["dataset", "MAE_mean"],
    ascending=[True, True],
).reset_index(drop=True)

run_summary_path = OUTPUT_DIR / "exp017_run_level_summary.csv"

run_summary.to_csv(
    run_summary_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 7. ERROR BY ACTUAL CATEGORICAL LABEL
# ============================================================

actual_label_summary = grouped_metrics(
    long_df,
    ["dataset", "method", "actual_label"],
)

actual_label_summary = actual_label_summary.sort_values(
    ["dataset", "actual_label", "MAE"],
    ascending=[True, True, False],
)

actual_label_path = (
    OUTPUT_DIR / "exp017_actual_label_error_summary.csv"
)

actual_label_summary.to_csv(
    actual_label_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 8. ERROR BY PREDICTED CATEGORICAL LABEL
# ============================================================

predicted_label_summary = grouped_metrics(
    long_df,
    ["dataset", "method", "predicted_label"],
)

predicted_label_summary = predicted_label_summary.sort_values(
    ["dataset", "predicted_label", "MAE"],
    ascending=[True, True, False],
)

predicted_label_path = (
    OUTPUT_DIR / "exp017_predicted_label_error_summary.csv"
)

predicted_label_summary.to_csv(
    predicted_label_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 9. ERROR BY ACTUAL HUMAN-SCORE BAND
# ============================================================

score_band_summary = grouped_metrics(
    long_df.dropna(subset=["actual_score_band"]),
    ["dataset", "method", "actual_score_band"],
)

score_band_summary = score_band_summary.sort_values(
    ["dataset", "method", "actual_score_band"],
)

score_band_path = (
    OUTPUT_DIR / "exp017_score_band_error_summary.csv"
)

score_band_summary.to_csv(
    score_band_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 10. ERROR BY QUESTION
# ============================================================

question_summary = grouped_metrics(
    long_df,
    ["dataset", "method", "question_text"],
)

question_summary = question_summary.sort_values(
    ["dataset", "method", "MAE"],
    ascending=[True, True, False],
)

question_path = (
    OUTPUT_DIR / "exp017_question_error_summary.csv"
)

question_summary.to_csv(
    question_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 11. REPEATED ERRORS FOR INDIVIDUAL ANSWERS
# ============================================================

# The same original answer may appear in the test set across multiple
# random-state runs. Aggregate its predictions before identifying
# consistently difficult answers.
#
# This avoids treating five predictions for one answer as five
# independent answers.

row_summary_records = []

row_group_columns = [
    "dataset",
    "method",
    "_row_key",
]

for (dataset_name, method_name, row_key), group in long_df.groupby(
    row_group_columns,
    dropna=False,
    sort=False,
):
    errors = group["residual"].to_numpy(dtype=float)
    predictions = group["prediction"].to_numpy(dtype=float)

    record = {
        "dataset": dataset_name,
        "method": method_name,
        "row_key": row_key,

        "source_csv_line": group["source_csv_line"].iloc[0],

        "question_text": group["question_text"].iloc[0],
        "student_answer": group["student_answer"].iloc[0],

        "actual_label": group["actual_label"].iloc[0],
        "predicted_label": group["predicted_label"].iloc[0],

        "actual_human_score": float(
            group["actual_human_score"].iloc[0]
        ),

        "test_occurrences": int(len(group)),
        "unique_random_states": int(
            group["random_state"].nunique()
        ),

        "mean_prediction": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions, ddof=0)),

        "mean_prediction_error": float(np.mean(errors)),
        "mean_absolute_error": float(np.mean(np.abs(errors))),
        "maximum_absolute_error": float(
            np.max(np.abs(errors))
        ),
    }

    row_summary_records.append(record)

row_summary = pd.DataFrame(row_summary_records)

row_summary = row_summary.sort_values(
    ["dataset", "method", "mean_absolute_error"],
    ascending=[True, True, False],
)

row_summary_path = (
    OUTPUT_DIR / "exp017_row_repeat_summary.csv"
)

row_summary.to_csv(
    row_summary_path,
    index=False,
    encoding="utf-8-sig",
)

# Keep the top 10 consistently difficult answers per method and dataset.
worst_errors = (
    row_summary
    .sort_values(
        ["dataset", "method", "mean_absolute_error"],
        ascending=[True, True, False],
    )
    .groupby(
        ["dataset", "method"],
        group_keys=False,
        sort=False,
    )
    .head(10)
    .reset_index(drop=True)
)

worst_errors_path = OUTPUT_DIR / "exp017_worst_errors.csv"

worst_errors.to_csv(
    worst_errors_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 12. PLOT: MAE BY METHOD AND DATASET
# ============================================================

datasets = sorted(run_summary["dataset"].unique())

n_datasets = len(datasets)

fig, axes = plt.subplots(
    1,
    n_datasets,
    figsize=(7 * n_datasets, 5),
    squeeze=False,
)

for index, dataset_name in enumerate(datasets):
    ax = axes[0, index]

    subset = run_summary[
        run_summary["dataset"] == dataset_name
    ].copy()

    method_order = [
        name
        for name in METHOD_COLUMNS.values()
        if name in subset["method"].values
    ]

    subset = subset.set_index("method")

    means = [
        subset.loc[name, "MAE_mean"]
        for name in method_order
    ]

    stds = [
        subset.loc[name, "MAE_std"]
        for name in method_order
    ]

    stds = [
        0.0 if pd.isna(value) else value
        for value in stds
    ]

    x_values = np.arange(len(method_order))

    ax.bar(
        x_values,
        means,
        yerr=stds,
        capsize=4,
    )

    ax.set_xticks(x_values)
    ax.set_xticklabels(
        method_order,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel("MAE (lower is better)")

    ax.set_title(f"{dataset_name}: MAE across random states")

    ax.grid(
        axis="y",
        alpha=0.25,
    )

fig.suptitle(
    "Exp 017 — Scorer Error Comparison",
    y=1.03,
)

fig.tight_layout()

mae_plot_path = OUTPUT_DIR / "exp017_mae_by_method.png"

fig.savefig(
    mae_plot_path,
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 13. PLOT: BIAS BY ACTUAL SCORE BAND
# ============================================================

fig, axes = plt.subplots(
    1,
    n_datasets,
    figsize=(8 * n_datasets, 5),
    squeeze=False,
)

band_positions = np.arange(len(SCORE_BAND_LABELS))

for index, dataset_name in enumerate(datasets):
    ax = axes[0, index]

    subset = score_band_summary[
        score_band_summary["dataset"] == dataset_name
    ].copy()

    for method_name in METHOD_COLUMNS.values():
        method_data = subset[
            subset["method"] == method_name
        ].copy()

        if method_data.empty:
            continue

        band_to_error = {}

        for _, record in method_data.iterrows():
            band_to_error[str(record["actual_score_band"])] = (
                record["mean_prediction_error"]
            )

        y_values = [
            band_to_error.get(band, np.nan)
            for band in SCORE_BAND_LABELS
        ]

        ax.plot(
            band_positions,
            y_values,
            marker="o",
            label=method_name,
        )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(band_positions)
    ax.set_xticklabels(SCORE_BAND_LABELS)

    ax.set_xlabel("Actual human-score band")

    ax.set_ylabel(
        "Mean prediction error: predicted − actual"
    )

    ax.set_title(f"{dataset_name}: Score-band bias")

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
    )

fig.suptitle(
    "Exp 017 — Overprediction and Underprediction by Score Band",
    y=1.03,
)

fig.tight_layout()

bias_plot_path = (
    OUTPUT_DIR / "exp017_error_by_actual_score_band.png"
)

fig.savefig(
    bias_plot_path,
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 14. MARKDOWN REPORT HELPERS
# ============================================================

def format_cell(value):
    """Format a value safely for a Markdown table."""

    if pd.isna(value):
        return "—"

    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    return (
        str(value)
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def markdown_table(data, columns):
    """Create a Markdown table without requiring tabulate."""

    if data.empty:
        return "_No rows available._"

    lines = []

    headers = [str(column) for column in columns]

    lines.append(
        "| " + " | ".join(headers) + " |"
    )

    lines.append(
        "| " + " | ".join("---" for _ in headers) + " |"
    )

    for _, record in data[columns].iterrows():
        values = [
            format_cell(record[column])
            for column in columns
        ]

        lines.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(lines)


# ============================================================
# 15. BUILD THE REPORT
# ============================================================

report_lines = []

report_lines.append("# Experiment 017 — Error Analysis and Scorer Calibration")
report_lines.append("")

report_lines.append(
    "## Objective"
)
report_lines.append(
    "Analyse the saved predictions from Experiment 016 to understand "
    "where each numerical scorer makes errors. This experiment does "
    "not retrain models, change labels, modify human scores, or "
    "automatically calibrate any predictions."
)
report_lines.append("")

report_lines.append(
    "## Data and interpretation"
)
report_lines.append(
    f"- Input prediction rows: {len(df)}."
)
report_lines.append(
    f"- Unique original answer rows: {df['_row_key'].nunique()}."
)
report_lines.append(
    f"- Datasets: {', '.join(sorted(df['dataset'].unique()))}."
)
report_lines.append(
    f"- Random states observed: {df['random_state'].nunique()}."
)
report_lines.append(
    "- Residual = predicted score − actual human score."
)
report_lines.append(
    "- Positive residual means overprediction; negative residual "
    "means underprediction."
)
report_lines.append(
    "- The same original answer can occur in multiple test splits. "
    "Repeated predictions are not independent answers or independent "
    "datasets."
)
report_lines.append("")

report_lines.append(
    "## 1. Mean performance across random-state runs"
)
report_lines.append(
    "The table below reports the mean and standard deviation of the "
    "metrics calculated separately for each random state. These are "
    "the more appropriate figures for comparing the repeated splits."
)
report_lines.append("")

run_table_columns = [
    "dataset",
    "method",
    "repetitions",
    "MAE_mean",
    "MAE_std",
    "RMSE_mean",
    "RMSE_std",
    "R2_mean",
    "Spearman_mean",
    "mean_prediction_error_mean",
    "prediction_to_actual_std_ratio_mean",
]

report_lines.append(
    markdown_table(
        run_summary,
        run_table_columns,
    )
)
report_lines.append("")

for dataset_name in datasets:
    subset = run_summary[
        run_summary["dataset"] == dataset_name
    ].copy()

    report_lines.append(f"### {dataset_name}")

    if subset.empty:
        report_lines.append("No results available.")
        report_lines.append("")
        continue

    mae_winner = subset.loc[
        subset["MAE_mean"].idxmin()
    ]

    rmse_winner = subset.loc[
        subset["RMSE_mean"].idxmin()
    ]

    spearman_valid = subset.dropna(
        subset=["Spearman_mean"]
    )

    report_lines.append(
        f"- Lowest mean MAE: **{mae_winner['method']}** "
        f"({mae_winner['MAE_mean']:.3f})."
    )

    report_lines.append(
        f"- Lowest mean RMSE: **{rmse_winner['method']}** "
        f"({rmse_winner['RMSE_mean']:.3f})."
    )

    if not spearman_valid.empty:
        rank_winner = spearman_valid.loc[
            spearman_valid["Spearman_mean"].idxmax()
        ]

        report_lines.append(
            f"- Highest mean Spearman correlation: "
            f"**{rank_winner['method']}** "
            f"({rank_winner['Spearman_mean']:.3f})."
        )

    # A standard-deviation ratio below 1 means that predictions vary
    # less than the human scores in the evaluated test occurrences.
    ratio_data = subset[
        [
            "method",
            "prediction_to_actual_std_ratio_mean",
        ]
    ].copy()

    ratio_data = ratio_data.sort_values(
        "prediction_to_actual_std_ratio_mean"
    )

    if not ratio_data.empty:
        report_lines.append(
            "- Prediction-to-actual standard-deviation ratios are "
            "listed in the main table. Ratios below 1 indicate a "
            "narrower spread of predicted scores than actual scores; "
            "this is a descriptive compression diagnostic, not proof "
            "of calibration quality."
        )

    report_lines.append("")

report_lines.append(
    "## 2. Pooled prediction metrics"
)
report_lines.append(
    "Pooled metrics combine all available test-prediction occurrences "
    "across random states. Since original answers can reappear across "
    "runs, these values are descriptive and are not independent-sample "
    "estimates."
)
report_lines.append("")

pooled_table_columns = [
    "dataset",
    "method",
    "n_prediction_occurrences",
    "unique_test_rows",
    "MAE",
    "RMSE",
    "mean_prediction_error",
    "overprediction_pct",
    "underprediction_pct",
    "within_1_point_pct",
    "prediction_to_actual_std_ratio",
    "predictions_below_zero_pct",
    "predictions_above_10_pct",
]

report_lines.append(
    markdown_table(
        pooled_summary,
        pooled_table_columns,
    )
)
report_lines.append("")

report_lines.append(
    "## 3. Error patterns by actual categorical label"
)
report_lines.append(
    "See `exp017_actual_label_error_summary.csv` for MAE, RMSE, "
    "bias and the number of evaluated occurrences for every "
    "dataset, scorer and actual label. Interpret small groups "
    "cautiously."
)
report_lines.append("")

report_lines.append(
    "## 4. Error patterns by predicted categorical label"
)
report_lines.append(
    "See `exp017_predicted_label_error_summary.csv` for numerical "
    "scoring errors grouped by the categorical label predicted by "
    "the classification model."
)
report_lines.append("")

report_lines.append(
    "## 5. Error patterns by actual score band"
)
report_lines.append(
    "See `exp017_score_band_error_summary.csv` and "
    "`exp017_error_by_actual_score_band.png`. The bias plot uses "
    "predicted score minus actual human score: values above zero "
    "indicate overprediction and values below zero indicate "
    "underprediction."
)
report_lines.append("")

report_lines.append(
    "## 6. Errors by question"
)
report_lines.append(
    "See `exp017_question_error_summary.csv` to identify questions "
    "where a scorer performs poorly or inconsistently. Some questions "
    "may have few student responses, so their metrics can be unstable."
)
report_lines.append("")

report_lines.append(
    "## 7. Repeated errors for individual answers"
)
report_lines.append(
    "`exp017_row_repeat_summary.csv` aggregates repeated predictions "
    "for each original answer and scorer. "
    "`exp017_worst_errors.csv` contains the ten highest mean absolute "
    "errors per dataset and scorer, based on the observed appearances "
    "of each answer in the test splits. Review the associated question, "
    "answer, actual label and human score before concluding that a "
    "scorer or human annotation is wrong."
)
report_lines.append("")

report_lines.append(
    "## 8. Interpretation cautions"
)
report_lines.append(
    "- These datasets contain a limited number of unique questions."
)
report_lines.append(
    "- Repeated random splits reuse questions and often reuse answer "
    "rows, so repetitions are not independent external validations."
)
report_lines.append(
    "- Error patterns can reflect model limitations, variation in "
    "answer difficulty, score-label alignment, rubric ambiguity, or "
    "human annotation decisions."
)
report_lines.append(
    "- This experiment does not automatically alter labels, human "
    "scores, model outputs, or calibration settings."
)
report_lines.append(
    "- Predictions outside the 0–10 range are reported rather than "
    "silently clipped."
)
report_lines.append("")

report_lines.append(
    "## Generated files"
)
report_files = [
    residuals_path,
    pooled_path,
    run_metrics_path,
    run_summary_path,
    actual_label_path,
    predicted_label_path,
    score_band_path,
    question_path,
    row_summary_path,
    worst_errors_path,
    mae_plot_path,
    bias_plot_path,
]

for path in report_files:
    report_lines.append(f"- `{path.name}`")

report_lines.append("")

report_path = OUTPUT_DIR / "exp017_report.md"

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# ============================================================
# 16. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("EXPERIMENT 017 COMPLETED")
print("=" * 72)

print("\nMean performance across random-state runs:")

display_columns = [
    "dataset",
    "method",
    "repetitions",
    "MAE_mean",
    "MAE_std",
    "RMSE_mean",
    "RMSE_std",
    "R2_mean",
    "Spearman_mean",
    "mean_prediction_error_mean",
    "prediction_to_actual_std_ratio_mean",
]

for _, record in run_summary[display_columns].iterrows():
    print(
        f"\nDataset: {record['dataset']}"
        f"\nMethod: {record['method']}"
        f"\n  Repetitions: {record['repetitions']}"
        f"\n  MAE: {record['MAE_mean']:.4f}"
        f"\n  MAE standard deviation: {record['MAE_std']:.4f}"
        f"\n  RMSE: {record['RMSE_mean']:.4f}"
        f"\n  RMSE standard deviation: {record['RMSE_std']:.4f}"
        f"\n  R2: {record['R2_mean']:.4f}"
        f"\n  Spearman: {record['Spearman_mean']:.4f}"
        f"\n  Mean prediction error: "
        f"{record['mean_prediction_error_mean']:.4f}"
        f"\n  Prediction/actual spread ratio: "
        f"{record['prediction_to_actual_std_ratio_mean']:.4f}"
    )

print("\n\nOutput directory:")
print(OUTPUT_DIR)

print("\nImportant files to inspect first:")
print(f"  1. {report_path.name}")
print(f"  2. {run_summary_path.name}")
print(f"  3. {actual_label_path.name}")
print(f"  4. {score_band_path.name}")
print(f"  5. {worst_errors_path.name}")

print("\nNo models were retrained.")
print("No source datasets or human annotations were modified.")
print("\nExperiment 017 finished successfully.")