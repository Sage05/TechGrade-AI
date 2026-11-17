"""Exp 018: Classification Error Attribution for TechGradeAI.

Uses Exp 016 predictions and per-random-state training-only label-score means.
No models are retrained and no source labels/scores are changed.
The oracle-category score uses the true test label only as a diagnostic; it is
NOT a deployable scorer and must not be reported as normal test performance.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "experiments" / "exp_016_scoring_robustness" / "results"
PREDICTIONS_PATH = INPUT_DIR / "exp016_predictions.csv"
TRAIN_MEANS_PATH = INPUT_DIR / "exp016_training_label_means.csv"
OUTPUT_DIR = ROOT / "experiments" / "exp_018_classification_error_attribution" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["Correct", "Partially Correct", "Incorrect", "Irrelevant", "Contradictory"]
STATE_COLUMNS = ["random_state", "seed", "split_seed", "random_seed"]
LABEL_COLUMNS = ["label", "actual_label", "category", "class_label", "training_label"]
MEAN_COLUMNS = [
    "training_mean_score", "training_label_mean_score", "training_label_mean",
    "train_mean_score", "mean_training_score", "mean_human_score", "mean_score",
    "score_mean", "label_mean_score", "training_mean",
]
PRED_MEAN_COLUMNS = [
    "predicted_category_mean", "predicted_category_mean_score",
    "predicted_category_mean_prediction", "category_mean_score",
]


def norm_col(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def norm_value(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))


def find_col(frame, candidates, description):
    lookup = {norm_col(c): c for c in frame.columns}
    for candidate in candidates:
        if norm_col(candidate) in lookup:
            return lookup[norm_col(candidate)]
    raise ValueError(
        f"Could not find {description}. Tried {candidates}. Available columns: {list(frame.columns)}"
    )


def state_key(value):
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(value) else int(value)


def score_metrics(frame, score_col):
    x = frame.dropna(subset=["actual_human_score", score_col])
    if x.empty:
        return {"prediction_occurrences": 0, "unique_test_rows": 0, "unique_questions": 0,
                "MAE": np.nan, "RMSE": np.nan, "mean_prediction_error": np.nan,
                "within_1_point_pct": np.nan, "within_2_points_pct": np.nan}
    err = x[score_col].to_numpy(float) - x["actual_human_score"].to_numpy(float)
    return {
        "prediction_occurrences": len(x),
        "unique_test_rows": x["_row_key"].nunique(),
        "unique_questions": x["question_text"].nunique(),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "mean_prediction_error": float(np.mean(err)),
        "within_1_point_pct": float(np.mean(np.abs(err) <= 1) * 100),
        "within_2_points_pct": float(np.mean(np.abs(err) <= 2) * 100),
    }


def md_table(frame, columns):
    if frame.empty:
        return "_No rows available._"
    def fmt(v):
        if pd.isna(v): return "—"
        if isinstance(v, (float, np.floating)): return f"{v:.3f}"
        if isinstance(v, (int, np.integer)): return str(int(v))
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, r in frame[columns].iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in columns) + " |")
    return "\n".join(lines)


print("=" * 72)
print("EXPERIMENT 018 — CLASSIFICATION ERROR ATTRIBUTION")
print("=" * 72)
print(f"Predictions: {PREDICTIONS_PATH}")
print(f"Training means: {TRAIN_MEANS_PATH}")
if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(f"Missing {PREDICTIONS_PATH}. Make sure Exp 016 completed.")
if not TRAIN_MEANS_PATH.exists():
    raise FileNotFoundError(
        f"Missing {TRAIN_MEANS_PATH}. Exp 018 requires per-split training-only means; "
        "it will not calculate oracle scores from test human scores."
    )

p = pd.read_csv(PREDICTIONS_PATH)
m = pd.read_csv(TRAIN_MEANS_PATH)
p.columns = [str(c).strip() for c in p.columns]
m.columns = [str(c).strip() for c in m.columns]
required = ["dataset", "actual_label", "predicted_label", "actual_human_score"]
missing = [c for c in required if c not in p.columns]
if missing:
    raise ValueError(f"Exp 016 predictions file is missing {missing}. Found: {list(p.columns)}")

pred_mean_col = find_col(p, PRED_MEAN_COLUMNS, "predicted-category mean score column")
p_state_col = find_col(p, STATE_COLUMNS, "random-state column in predictions")
m_state_col = find_col(m, STATE_COLUMNS, "random-state column in training means")
m_label_col = find_col(m, LABEL_COLUMNS, "label column in training means")
m_score_col = find_col(m, MEAN_COLUMNS, "training-only mean score column")
m_dataset_col = find_col(m, ["dataset", "subject", "dataset_name"], "dataset column in training means")

for c in ["source_csv_line", "question_text", "student_answer"]:
    if c not in p.columns:
        p[c] = "[unavailable]"
for c in ["dataset", "actual_label", "predicted_label", "question_text", "student_answer"]:
    p[c] = p[c].fillna("[missing]").astype(str).str.strip()
p["actual_human_score"] = pd.to_numeric(p["actual_human_score"], errors="coerce")
p[pred_mean_col] = pd.to_numeric(p[pred_mean_col], errors="coerce")
p["random_state"] = p[p_state_col].map(state_key)
bad_score = p["actual_human_score"].isna() | ~p["actual_human_score"].between(0, 10)
if bad_score.any():
    print(f"WARNING: Excluding {int(bad_score.sum())} rows with invalid human scores.")
    p = p.loc[~bad_score].copy()
if p.empty:
    raise ValueError("No valid prediction rows remain after validating actual human scores.")

# Track the same original answer across repeated random-state test splits.
line = p["source_csv_line"].fillna("").astype(str).str.strip()
has_line = line.ne("") & line.str.lower().ne("nan")
fallback_key = p["dataset"] + "|Q:" + p["question_text"] + "|A:" + p["student_answer"]
p["_row_key"] = np.where(has_line, p["dataset"] + "|line:" + line, fallback_key)

# Make a lookup table for training-only score means keyed by dataset, split and label.
m["_dataset_key"] = m[m_dataset_col].map(norm_value)
m["_label_key"] = m[m_label_col].map(norm_value)
m["_state_key"] = m[m_state_col].map(state_key)
m["_mean"] = pd.to_numeric(m[m_score_col], errors="coerce")
m = m.dropna(subset=["_state_key", "_mean"]).copy()
key_cols = ["_dataset_key", "_state_key", "_label_key"]
if m.duplicated(key_cols).any():
    dup = m.loc[m.duplicated(key_cols, keep=False), [m_dataset_col, m_state_col, m_label_col, m_score_col]]
    raise ValueError("Duplicate dataset/random-state/label training means:\n" + dup.to_string(index=False))
means_lookup = {
    (r["_dataset_key"], int(r["_state_key"]), r["_label_key"]): float(r["_mean"])
    for _, r in m.iterrows()
}
p["_dataset_key"] = p["dataset"].map(norm_value)
p["_state_key"] = p["random_state"].map(state_key)
p["_actual_key"] = p["actual_label"].map(norm_value)
p["_predicted_key"] = p["predicted_label"].map(norm_value)
p["oracle_category_mean_score"] = [
    means_lookup.get((d, s, lab), np.nan)
    for d, s, lab in zip(p["_dataset_key"], p["_state_key"], p["_actual_key"])
]
p["training_mean_for_predicted_label"] = [
    means_lookup.get((d, s, lab), np.nan)
    for d, s, lab in zip(p["_dataset_key"], p["_state_key"], p["_predicted_key"])
]
if p["oracle_category_mean_score"].isna().any():
    miss = p.loc[p["oracle_category_mean_score"].isna(), ["dataset", "random_state", "actual_label"]].drop_duplicates()
    raise ValueError("Could not find training-fold score mean for these actual labels:\n" + miss.to_string(index=False))

p["classification_correct"] = p["_actual_key"] == p["_predicted_key"]
p["predicted_category_error"] = p[pred_mean_col] - p["actual_human_score"]
p["oracle_category_error"] = p["oracle_category_mean_score"] - p["actual_human_score"]
p["predicted_category_absolute_error"] = p["predicted_category_error"].abs()
p["oracle_category_absolute_error"] = p["oracle_category_error"].abs()
# Positive value means the oracle diagnostic has lower absolute error.
p["oracle_mae_reduction"] = p["predicted_category_absolute_error"] - p["oracle_category_absolute_error"]
p["oracle_improved"] = p["oracle_mae_reduction"] > 0
mapping_diff = (p[pred_mean_col] - p["training_mean_for_predicted_label"]).abs().median()

# Classification metrics pooled across test appearances and for each split.
class_rows = []
for dataset, d in p.groupby("dataset", sort=True):
    groups = [("POOLED_ALL_RUNS", d)] + [(int(seed), g) for seed, g in d.groupby("random_state", sort=True)]
    for seed, g in groups:
        class_rows.append({
            "dataset": dataset, "random_state": seed,
            "prediction_occurrences": len(g), "unique_test_rows": g["_row_key"].nunique(),
            "unique_questions": g["question_text"].nunique(),
            "accuracy": accuracy_score(g["actual_label"], g["predicted_label"]),
            "macro_f1": f1_score(g["actual_label"], g["predicted_label"], average="macro", zero_division=0),
            "weighted_f1": f1_score(g["actual_label"], g["predicted_label"], average="weighted", zero_division=0),
            "classification_error_pct": (~g["classification_correct"]).mean() * 100,
        })
class_metrics = pd.DataFrame(class_rows)
class_metrics.to_csv(OUTPUT_DIR / "exp018_classification_metrics.csv", index=False, encoding="utf-8-sig")

# Confusion matrices and per-class precision/recall/F1, pooled by dataset.
class_rows = []
confusion_counts = {}
for dataset, d in p.groupby("dataset", sort=True):
    observed = sorted(set(d["actual_label"]) | set(d["predicted_label"]))
    labels = [x for x in LABEL_ORDER if x in observed] + [x for x in observed if x not in LABEL_ORDER]
    cr = classification_report(d["actual_label"], d["predicted_label"], labels=labels, output_dict=True, zero_division=0)
    for label in labels:
        r = cr.get(label, {})
        class_rows.append({"dataset": dataset, "label": label, "precision": r.get("precision", np.nan),
                           "recall": r.get("recall", np.nan), "f1_score": r.get("f1-score", np.nan),
                           "support_occurrences": r.get("support", 0)})
    cm = pd.crosstab(d["actual_label"], d["predicted_label"]).reindex(index=labels, columns=labels, fill_value=0)
    confusion_counts[dataset] = cm
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", dataset)
    cm.to_csv(OUTPUT_DIR / f"exp018_confusion_counts_{safe}.csv", encoding="utf-8-sig")
    (cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) * 100).to_csv(
        OUTPUT_DIR / f"exp018_confusion_row_percent_{safe}.csv", encoding="utf-8-sig")
pd.DataFrame(class_rows).to_csv(OUTPUT_DIR / "exp018_per_class_report.csv", index=False, encoding="utf-8-sig")

# Numerical scoring comparison: deployed-style category prediction vs diagnostic oracle category.
score_rows = []
for dataset, d in p.groupby("dataset", sort=True):
    for name, col in [("Predicted-category mean (current)", pred_mean_col),
                      ("Oracle-category mean (diagnostic only)", "oracle_category_mean_score")]:
        r = {"dataset": dataset, "scorer": name}
        r.update(score_metrics(d, col))
        score_rows.append(r)
score_comparison = pd.DataFrame(score_rows)
score_comparison.to_csv(OUTPUT_DIR / "exp018_predicted_vs_oracle_score_metrics.csv", index=False, encoding="utf-8-sig")

# How metrics differ when classification is correct versus incorrect.
attr_rows = []
for (dataset, correct), d in p.groupby(["dataset", "classification_correct"], sort=True):
    current = score_metrics(d, pred_mean_col)
    oracle = score_metrics(d, "oracle_category_mean_score")
    attr_rows.append({
        "dataset": dataset,
        "classification_status": "Correct category" if correct else "Incorrect category",
        "prediction_occurrences": len(d), "unique_test_rows": d["_row_key"].nunique(),
        "current_category_mean_MAE": current["MAE"], "oracle_category_mean_MAE": oracle["MAE"],
        "mean_absolute_error_reduction_with_oracle": current["MAE"] - oracle["MAE"],
        "oracle_improvement_occurrence_pct": d["oracle_improved"].mean() * 100,
        "current_category_mean_bias": current["mean_prediction_error"],
        "oracle_category_mean_bias": oracle["mean_prediction_error"],
    })
attribution = pd.DataFrame(attr_rows)
attribution.to_csv(OUTPUT_DIR / "exp018_error_by_classification_correctness.csv", index=False, encoding="utf-8-sig")

# Which actual -> predicted category confusions are common and numerically costly?
transition_rows = []
for (dataset, actual, predicted), d in p.groupby(["dataset", "actual_label", "predicted_label"], sort=True, dropna=False):
    transition_rows.append({
        "dataset": dataset, "actual_label": actual, "predicted_label": predicted,
        "classification_correct": norm_value(actual) == norm_value(predicted),
        "prediction_occurrences": len(d), "unique_test_rows": d["_row_key"].nunique(),
        "actual_score_mean": d["actual_human_score"].mean(),
        "predicted_category_mean_MAE": d["predicted_category_absolute_error"].mean(),
        "oracle_category_mean_MAE": d["oracle_category_absolute_error"].mean(),
        "mean_absolute_error_reduction_with_oracle": d["oracle_mae_reduction"].mean(),
    })
transitions = pd.DataFrame(transition_rows).sort_values(["dataset", "prediction_occurrences"], ascending=[True, False])
transitions.to_csv(OUTPUT_DIR / "exp018_label_transition_errors.csv", index=False, encoding="utf-8-sig")

# Row-level results and the clearest examples among misclassified answers.
cols = ["dataset", "random_state", "source_csv_line", "_row_key", "question_text", "student_answer",
        "actual_label", "predicted_label", "classification_correct", "actual_human_score", pred_mean_col,
        "oracle_category_mean_score", "predicted_category_error", "oracle_category_error",
        "predicted_category_absolute_error", "oracle_category_absolute_error", "oracle_mae_reduction", "oracle_improved"]
row_level = p[cols].rename(columns={"_row_key": "answer_key", pred_mean_col: "predicted_category_mean_score"})
row_level.to_csv(OUTPUT_DIR / "exp018_answer_level_attribution.csv", index=False, encoding="utf-8-sig")
mismatch = p.loc[~p["classification_correct"]].copy()
example_cols = ["dataset", "random_state", "source_csv_line", "question_text", "student_answer", "actual_label",
                "predicted_label", "actual_human_score", pred_mean_col, "oracle_category_mean_score",
                "predicted_category_absolute_error", "oracle_category_absolute_error", "oracle_mae_reduction"]
for ascending, filename in [(False, "exp018_misclassifications_oracle_helps_most.csv"),
                            (True, "exp018_misclassifications_oracle_helps_least.csv")]:
    examples = mismatch.sort_values("oracle_mae_reduction", ascending=ascending).groupby("dataset", group_keys=False).head(20)
    examples[example_cols].rename(columns={pred_mean_col: "predicted_category_mean_score"}).to_csv(
        OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")

# Plot confusion matrices (counts across the repeated test runs).
fig, axes = plt.subplots(1, len(confusion_counts), figsize=(7 * len(confusion_counts), 6), squeeze=False)
for i, (dataset, cm) in enumerate(confusion_counts.items()):
    ax = axes[0, i]
    arr = cm.to_numpy(dtype=float)
    im = ax.imshow(arr, interpolation="nearest", aspect="auto")
    ax.set_title(f"{dataset}: confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(np.arange(len(cm.columns))); ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cm.index))); ax.set_yticklabels(cm.index)
    threshold = np.nanmax(arr) / 2 if arr.size else 0
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            ax.text(c, r, str(int(arr[r, c])), ha="center", va="center",
                    color="white" if arr[r, c] > threshold else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Test occurrences")
fig.suptitle("Exp 018 — Classification Error Attribution", y=1.02)
fig.tight_layout()
plot_path = OUTPUT_DIR / "exp018_confusion_matrices.png"
fig.savefig(plot_path, dpi=180, bbox_inches="tight")
plt.close(fig)

# Write a compact report.
report = [
    "# Experiment 018 — Classification Error Attribution", "",
    "## Purpose",
    "Compare the current predicted-category mean with a diagnostic oracle-category mean. The oracle uses the true test label to select a score mean calculated only from that split's training fold. It is not a deployable scorer.", "",
    "## Data and caveats",
    f"- Test prediction occurrences: {len(p)}.",
    f"- Unique answer rows across test runs: {p['_row_key'].nunique()}.",
    f"- Datasets: {', '.join(sorted(p['dataset'].unique()))}.",
    f"- Random states: {', '.join(str(x) for x in sorted(p['random_state'].dropna().unique()))}.",
    f"- Median absolute difference between saved predicted-category means and training-mean lookup: {mapping_diff:.6f}.",
    "- Repeated test appearances reuse some answers and are not independent samples.",
    "- Oracle scores are a diagnostic only; never report them as ordinary model test scores.", "",
    "## 1. Classification metrics (pooled rows and per split)",
    md_table(class_metrics, ["dataset", "random_state", "prediction_occurrences", "unique_test_rows", "accuracy", "macro_f1", "weighted_f1", "classification_error_pct"]), "",
    "## 2. Current versus oracle-category score mapping",
    md_table(score_comparison, ["dataset", "scorer", "prediction_occurrences", "unique_test_rows", "MAE", "RMSE", "mean_prediction_error", "within_1_point_pct", "within_2_points_pct"]), "",
    "## 3. Attribution by classification correctness",
    md_table(attribution, ["dataset", "classification_status", "prediction_occurrences", "unique_test_rows", "current_category_mean_MAE", "oracle_category_mean_MAE", "mean_absolute_error_reduction_with_oracle", "oracle_improvement_occurrence_pct"]), "",
    "A positive MAE reduction means the oracle diagnostic reduced absolute error. This is not a causal estimate: the oracle uses the true test category by design, and score differences also reflect the relationship between labels and human scores.", "",
    "## Key output files",
    "- `exp018_classification_metrics.csv`: accuracy and F1 for each split and pooled results.",
    "- `exp018_per_class_report.csv`: precision, recall and F1 by class.",
    "- `exp018_confusion_counts_*.csv` and `exp018_confusion_row_percent_*.csv`: confusion matrices.",
    "- `exp018_predicted_vs_oracle_score_metrics.csv`: current vs diagnostic scoring metrics.",
    "- `exp018_error_by_classification_correctness.csv`: score errors for correct/incorrect classifications.",
    "- `exp018_label_transition_errors.csv`: error impact for actual-to-predicted category pairs.",
    "- `exp018_answer_level_attribution.csv`: row-level diagnostic.",
    "- `exp018_misclassifications_oracle_helps_most.csv` / `exp018_misclassifications_oracle_helps_least.csv`: examples to inspect.",
    "- `exp018_confusion_matrices.png`: confusion-matrix plot.", "",
    "## Limitations",
    "There are few unique questions; repeated random splits are not independent external validation. No source data, human annotations, labels, or models were modified."
]
report_path = OUTPUT_DIR / "exp018_report.md"
report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

print("\nClassification metrics:")
print(class_metrics[class_metrics["random_state"] == "POOLED_ALL_RUNS"].to_string(index=False))
print("\nPredicted-category mean vs oracle diagnostic:")
print(score_comparison.to_string(index=False))
print("\nError by classification correctness:")
print(attribution.to_string(index=False))
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Main report: {report_path}")
print(f"Confusion plot: {plot_path}")
print("\nExp 018 finished. No models were retrained and no source data were modified.")
