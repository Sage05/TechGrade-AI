"""TechGrade AI — Experiment 013: Human Score Analysis and Categorical Alignment.

This is a descriptive/audit experiment. It does not train a scorer, modify the
source CSV files, or automatically correct labels/scores.

Run from the TechGrade-AI repository root:
    python -m experiments.exp_013_score_analysis.run_experiment

Override dataset paths if needed:
    python -m experiments.exp_013_score_analysis.run_experiment \
        --os-csv "C:\\path\\to\\revised_sample_os.csv" \
        --dbms-csv "C:\\path\\to\\revised_sample_dbms.csv"

Outputs are written to experiments/exp_013_score_analysis/results by default.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_LABELS = [
    "Correct",
    "Partially Correct",
    "Incorrect",
    "Irrelevant",
    "Contradictory",
]

# These bands come from the project's dataset revision guidelines. They are
# suggested review ranges, NOT strict rules and must not be used to overwrite scores.
SUGGESTED_SCORE_BANDS: dict[str, tuple[float, float]] = {
    "Correct": (8.0, 10.0),
    "Partially Correct": (4.0, 7.5),
    "Incorrect": (1.0, 3.5),
    "Contradictory": (0.0, 3.0),
    "Irrelevant": (0.0, 2.0),
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "question_text": ("question_text", "question", "prompt", "question_text "),
    "student_answer": ("student_answer", "answer", "student response", "response"),
    "model_answer": ("model_answer", "reference_answer", "reference answer", "model answer"),
    "label": ("label", "class", "category", "evaluation_label", "grade_label"),
    "human_score": ("human_score", "score", "human score", "score_0_10", "marks"),
    "annotator_id": ("annotator_id", "annotator", "grader_id", "evaluator_id"),
}

LABEL_NORMALIZATION = {
    "correct": "Correct",
    "partially correct": "Partially Correct",
    "partial correct": "Partially Correct",
    "incorrect": "Incorrect",
    "irrelevant": "Irrelevant",
    "contradictory": "Contradictory",
}


def resolve_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename known column variants to canonical project column names."""
    out = df.copy()
    original_by_normalized: dict[str, str] = {}
    for col in out.columns:
        normalized = " ".join(str(col).strip().lower().replace("_", " ").split())
        original_by_normalized[normalized] = str(col)

    rename: dict[str, str] = {}
    found: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        candidates = [" ".join(a.strip().lower().replace("_", " ").split()) for a in aliases]
        # Ordered aliases make canonical column names win deterministically when
        # a file happens to contain both a canonical name and a legacy alias.
        for candidate in candidates:
            if candidate in original_by_normalized:
                actual = original_by_normalized[candidate]
                found[canonical] = actual
                if actual != canonical:
                    rename[actual] = canonical
                break
    return out.rename(columns=rename), found


def normalize_label(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = " ".join(str(value).strip().lower().replace("_", " ").split())
    if not text:
        return None
    return LABEL_NORMALIZATION.get(text, f"UNKNOWN: {str(value).strip()}")


def safe_text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    return df[column].astype("string").str.strip().replace("", pd.NA)


def load_dataset(path: Path, dataset_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    # Preserve source values in memory; no source file is ever overwritten.
    raw = pd.read_csv(path)
    df, found_columns = resolve_columns(raw)
    df = df.copy()
    df.insert(0, "_source_row", np.arange(2, len(df) + 2))  # CSV line number, including header
    df.insert(0, "_dataset", dataset_name)

    for col in ("question_text", "student_answer", "model_answer", "label", "human_score", "annotator_id"):
        if col not in df.columns:
            df[col] = pd.NA

    df["_label_raw"] = df["label"].copy()
    df["label_normalized"] = df["label"].map(normalize_label)
    df["_score_raw"] = df["human_score"].copy()
    # Coerce scores for analysis without changing the original source file.
    score_text = df["human_score"].astype("string").str.strip()
    df["human_score_numeric"] = pd.to_numeric(score_text, errors="coerce")
    df.loc[score_text.isna() | score_text.eq(""), "human_score_numeric"] = np.nan

    df["_question_clean"] = safe_text_series(df, "question_text")
    df["_answer_clean"] = safe_text_series(df, "student_answer")

    nonempty_score_text = score_text.notna() & score_text.ne("")
    score_non_numeric = nonempty_score_text & df["human_score_numeric"].isna()
    score_out_of_range = df["human_score_numeric"].notna() & ~df["human_score_numeric"].between(0, 10)
    missing_label = safe_text_series(df, "label").isna()
    unknown_label = df["label_normalized"].astype("string").str.startswith("UNKNOWN:", na=False)
    missing_question = df["_question_clean"].isna()
    missing_answer = df["_answer_clean"].isna()
    missing_reference = safe_text_series(df, "model_answer").isna()

    # Exact duplicate key is a flag for review only, not an instruction to delete.
    q_norm = df["_question_clean"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True)
    a_norm = df["_answer_clean"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True)
    duplicate_mask = (q_norm.ne("") & a_norm.ne("")) & pd.DataFrame({"q": q_norm, "a": a_norm}).duplicated(keep=False)

    suggested_band_low = df["label_normalized"].map({k: v[0] for k, v in SUGGESTED_SCORE_BANDS.items()})
    suggested_band_high = df["label_normalized"].map({k: v[1] for k, v in SUGGESTED_SCORE_BANDS.items()})
    outside_suggested_band = (
        df["human_score_numeric"].notna()
        & suggested_band_low.notna()
        & ((df["human_score_numeric"] < suggested_band_low) | (df["human_score_numeric"] > suggested_band_high))
    )

    flags: list[list[str]] = []
    for idx in df.index:
        row_flags: list[str] = []
        if missing_question.loc[idx]: row_flags.append("missing_question_text")
        if missing_answer.loc[idx]: row_flags.append("missing_student_answer")
        if missing_reference.loc[idx]: row_flags.append("missing_model_answer")
        if missing_label.loc[idx]: row_flags.append("missing_label")
        if unknown_label.loc[idx]: row_flags.append("unknown_label")
        if pd.isna(df.loc[idx, "human_score_numeric"]):
            row_flags.append("missing_or_non_numeric_human_score" if score_non_numeric.loc[idx] else "missing_human_score")
        if score_out_of_range.loc[idx]: row_flags.append("human_score_out_of_range")
        if duplicate_mask.loc[idx]: row_flags.append("possible_exact_duplicate_question_answer")
        if outside_suggested_band.loc[idx]: row_flags.append("outside_suggested_label_score_band_review_only")
        flags.append(row_flags)
    df["quality_flags"] = [";".join(x) for x in flags]

    noncanonical_labels = sorted({
        str(v) for v in df["label_normalized"].dropna().unique()
        if v not in EXPECTED_LABELS
    })
    audit = {
        "dataset": dataset_name,
        "source_path": str(path),
        "rows": int(len(df)),
        "unique_questions": int(df["_question_clean"].nunique(dropna=True)),
        "missing_question_text": int(missing_question.sum()),
        "missing_student_answer": int(missing_answer.sum()),
        "missing_model_answer": int(missing_reference.sum()),
        "missing_label": int(missing_label.sum()),
        "unknown_label": int(unknown_label.sum()),
        "missing_human_score": int((~nonempty_score_text & df["human_score_numeric"].isna()).sum()),
        "non_numeric_human_score": int(score_non_numeric.sum()),
        "out_of_range_human_score": int(score_out_of_range.sum()),
        "possible_duplicate_question_answer_rows": int(duplicate_mask.sum()),
        "outside_suggested_label_score_band_rows_review_only": int(outside_suggested_band.sum()),
        "annotator_id_column_present": bool(found_columns.get("annotator_id")),
        "unique_annotator_ids": int(safe_text_series(df, "annotator_id").nunique(dropna=True)),
        "unrecognized_label_values": "; ".join(noncanonical_labels),
    }
    return df, audit


def describe_by_label(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name in list(df["_dataset"].dropna().unique()):
        dataset_df = df[df["_dataset"] == dataset_name]
        labels = EXPECTED_LABELS + sorted(
            x for x in dataset_df["label_normalized"].dropna().unique()
            if x not in EXPECTED_LABELS
        )
        for label in labels:
            group_all = dataset_df[dataset_df["label_normalized"] == label]
            group = group_all["human_score_numeric"].dropna()
            if len(group_all) == 0:
                continue
            band = SUGGESTED_SCORE_BANDS.get(label)
            valid_in_band = int(group.between(band[0], band[1]).sum()) if band else np.nan
            rows.append({
                "dataset": dataset_name,
                "label": label,
                "answer_count": int(len(group_all)),
                "valid_numeric_score_count": int(len(group)),
                "missing_or_invalid_score_count": int(len(group_all) - len(group)),
                "mean": float(group.mean()) if len(group) else np.nan,
                "median": float(group.median()) if len(group) else np.nan,
                "std_dev": float(group.std(ddof=1)) if len(group) > 1 else np.nan,
                "minimum": float(group.min()) if len(group) else np.nan,
                "q1": float(group.quantile(0.25)) if len(group) else np.nan,
                "q3": float(group.quantile(0.75)) if len(group) else np.nan,
                "maximum": float(group.max()) if len(group) else np.nan,
                "suggested_band_low": band[0] if band else np.nan,
                "suggested_band_high": band[1] if band else np.nan,
                "count_within_suggested_band": valid_in_band,
                "percent_within_suggested_band": (100.0 * valid_in_band / len(group)) if band and len(group) else np.nan,
                "band_note": "Suggested range; flag for review only, never an automatic correction." if band else "No suggested band available.",
            })
    return pd.DataFrame(rows)


def score_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_name in list(df["_dataset"].dropna().unique()):
        subset = df[df["_dataset"] == dataset_name]
        scores = subset["human_score_numeric"].dropna()
        rows.append({
            "dataset": dataset_name,
            "rows": int(len(subset)),
            "unique_questions": int(subset["_question_clean"].nunique(dropna=True)),
            "numeric_score_count": int(len(scores)),
            "missing_or_invalid_score_count": int(len(subset) - len(scores)),
            "mean": float(scores.mean()) if len(scores) else np.nan,
            "median": float(scores.median()) if len(scores) else np.nan,
            "std_dev": float(scores.std(ddof=1)) if len(scores) > 1 else np.nan,
            "minimum": float(scores.min()) if len(scores) else np.nan,
            "q1": float(scores.quantile(0.25)) if len(scores) else np.nan,
            "q3": float(scores.quantile(0.75)) if len(scores) else np.nan,
            "maximum": float(scores.max()) if len(scores) else np.nan,
        })
    return pd.DataFrame(rows)


def run_kruskal_wallis(df: pd.DataFrame) -> pd.DataFrame:
    """Optional exploratory test of score differences across categorical labels."""
    rows: list[dict[str, Any]] = []
    try:
        from scipy.stats import kruskal
    except Exception:
        return pd.DataFrame([{
            "dataset": "ALL",
            "test": "Kruskal-Wallis",
            "status": "not run: scipy is unavailable",
            "groups_included": "",
            "statistic_H": np.nan,
            "p_value": np.nan,
            "interpretation_note": "Install scipy to run the exploratory test; descriptive summaries are still produced.",
        }])

    for dataset_name in list(df["_dataset"].dropna().unique()):
        subset = df[df["_dataset"] == dataset_name]
        groups: dict[str, np.ndarray] = {}
        for label in EXPECTED_LABELS:
            arr = subset.loc[subset["label_normalized"] == label, "human_score_numeric"].dropna().to_numpy(dtype=float)
            if len(arr) >= 2:
                groups[label] = arr
        if len(groups) < 2:
            rows.append({
                "dataset": dataset_name,
                "test": "Kruskal-Wallis",
                "status": "not run: fewer than two labels with at least two numeric scores",
                "groups_included": "; ".join(groups),
                "statistic_H": np.nan,
                "p_value": np.nan,
                "interpretation_note": "Descriptive only; small numbers of unique questions limit generalization.",
            })
            continue
        try:
            statistic, p_value = kruskal(*groups.values())
            status = "completed"
            note = ("Exploratory omnibus test for differences in score distributions across labels; "
                    "it does not establish label correctness or prove that a scoring model will generalize.")
            rows.append({
                "dataset": dataset_name,
                "test": "Kruskal-Wallis",
                "status": status,
                "groups_included": "; ".join(groups),
                "statistic_H": float(statistic),
                "p_value": float(p_value),
                "interpretation_note": note,
            })
        except Exception as exc:
            rows.append({
                "dataset": dataset_name,
                "test": "Kruskal-Wallis",
                "status": f"failed: {type(exc).__name__}: {exc}",
                "groups_included": "; ".join(groups),
                "statistic_H": np.nan,
                "p_value": np.nan,
                "interpretation_note": "Inspect the input data before interpreting this test.",
            })
    return pd.DataFrame(rows)


def plot_score_distribution(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    generated: list[Path] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib unavailable; skipping plots.")
        return generated

    valid = df[df["human_score_numeric"].notna()].copy()
    if valid.empty:
        print("[WARN] No valid numerical scores; skipping plots.")
        return generated

    ordered_labels = [x for x in EXPECTED_LABELS if x in set(valid["label_normalized"])]
    ordered_labels += sorted(x for x in valid["label_normalized"].dropna().unique() if x not in EXPECTED_LABELS)
    groups = [valid.loc[valid["label_normalized"] == label, "human_score_numeric"].to_numpy() for label in ordered_labels]
    nonempty = [(label, group) for label, group in zip(ordered_labels, groups) if len(group)]
    if nonempty:
        labels, values = zip(*nonempty)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(values, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=20)
        ax.set_title("Human score distribution by categorical label")
        ax.set_xlabel("Categorical label")
        ax.set_ylabel("Human score (0–10)")
        ax.set_ylim(-0.5, 10.5)
        fig.tight_layout()
        path = out_dir / "score_distribution_by_label.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)

    for dataset_name in valid["_dataset"].dropna().unique():
        subset = valid[valid["_dataset"] == dataset_name]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(subset["human_score_numeric"].to_numpy(), bins=np.arange(-0.25, 10.76, 0.5), edgecolor="black")
        ax.set_title(f"Human score distribution — {dataset_name}")
        ax.set_xlabel("Human score (0–10)")
        ax.set_ylabel("Number of answers")
        ax.set_xlim(-0.5, 10.5)
        fig.tight_layout()
        safe_name = "".join(c.lower() if c.isalnum() else "_" for c in str(dataset_name)).strip("_")
        path = out_dir / f"score_histogram_{safe_name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)
    return generated


def write_report(
    out_path: Path,
    audits: pd.DataFrame,
    overall: pd.DataFrame,
    by_label: pd.DataFrame,
    tests: pd.DataFrame,
    flags: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# Experiment 013 — Human Score Analysis and Categorical Alignment",
        "",
        "## Purpose",
        "Describe the human-score distributions in the existing datasets, inspect score variation within each categorical label, and flag data-quality items for review before implementing a numerical scorer.",
        "",
        "## Scope and safeguards",
        "- This experiment is descriptive; it does not train a scoring model.",
        "- Source CSV files are not modified.",
        "- Scores outside suggested label bands are flagged for review only, not treated as automatic errors.",
        "- The suggested bands are documented project guidelines, not rigid rules: Correct 8–10; Partially Correct 4–7.5; Incorrect 1–3.5; Contradictory 0–3; Irrelevant 0–2.",
        "- Categorical labels are not converted into a numeric ordinal scale because Contradictory and Irrelevant are distinct failure types, not necessarily ordered relative to each other.",
        "",
        "## Dataset audit",
        audits.to_markdown(index=False) if not audits.empty else "No audit summary generated.",
        "",
        "## Overall score summaries",
        overall.to_markdown(index=False, floatfmt=".3f") if not overall.empty else "No valid overall summaries generated.",
        "",
        "## Score distribution within categorical labels",
        by_label.to_markdown(index=False, floatfmt=".3f") if not by_label.empty else "No label-level summaries generated.",
        "",
        "## Exploratory Kruskal–Wallis tests",
        tests.to_markdown(index=False, floatfmt=".5g") if not tests.empty else "No statistical tests were run.",
        "",
        "## Rows requiring review",
        f"Rows with at least one flag: {int((flags['quality_flags'].fillna('').str.len() > 0).sum()) if not flags.empty else 0}",
        "See `row_quality_flags.csv` for row-level flags. Review flags do not automatically imply that a row is wrong.",
        "",
        "## Interpretation guidance",
        "1. Check score variation within each label. Substantial variation suggests that the label alone cannot determine a 0–10 score.",
        "2. Check the suggested-band alignment as an annotation QA aid, not a rigid pass/fail rule.",
        "3. Treat statistical tests as exploratory: there are relatively few unique questions, and answers to the same question are not independent observations.",
        "4. Before training a regression scorer, decide how to evaluate predictions (e.g. MAE and RMSE) and preserve question-level separation to reduce leakage.",
        "5. A score distribution analysis does not prove inter-rater reliability. That requires independent annotations for the same answers.",
        "",
        "## Decision for the next experiment",
        "Use these findings to design Exp 014 (an interpretable deterministic scoring baseline) and Exp 015 (a learned regression baseline). Do not begin scoring-model training until the data-quality flags and score rubric have been reviewed.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 013: analyze human scores and categorical alignment.")
    parser.add_argument("--os-csv", type=Path, default=Path(Path.home() / "Downloads" / "revised_sample_os.csv"), help="Path to the Operating Systems CSV.")
    parser.add_argument("--dbms-csv", type=Path, default=Path(Path.home() / "Downloads" / "revised_sample_dbms.csv"), help="Path to the DBMS CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/exp_013_score_analysis/results"), help="Directory for reports, CSV summaries, flags, and plots.")
    args = parser.parse_args()

    try:
        os_df, os_audit = load_dataset(args.os_csv, "OS")
        dbms_df, dbms_audit = load_dataset(args.dbms_csv, "DBMS")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print("Pass the correct files with --os-csv and --dbms-csv.", file=sys.stderr)
        return 2

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([os_df, dbms_df], ignore_index=True, sort=False)

    # Main row-level table identifies the source file and row for traceability.
    quality_flags = combined[["_dataset", "_source_row", "question_text", "student_answer", "label", "_score_raw", "human_score_numeric", "quality_flags"]].copy()
    quality_flags = quality_flags.rename(columns={"_dataset": "dataset", "_source_row": "source_csv_line", "_score_raw": "human_score_original"})
    quality_flags = quality_flags[quality_flags["quality_flags"].fillna("").str.len() > 0]

    audit_df = pd.DataFrame([os_audit, dbms_audit])
    overall = score_overall_summary(combined)
    by_label = describe_by_label(combined)
    tests = run_kruskal_wallis(combined)

    audit_df.to_csv(out_dir / "dataset_audit.csv", index=False)
    overall.to_csv(out_dir / "overall_score_summary.csv", index=False)
    by_label.to_csv(out_dir / "label_score_summary.csv", index=False)
    tests.to_csv(out_dir / "statistical_tests.csv", index=False)
    quality_flags.to_csv(out_dir / "row_quality_flags.csv", index=False)

    plots = plot_score_distribution(combined, out_dir)
    write_report(out_dir / "exp_013_report.md", audit_df, overall, by_label, tests, quality_flags)

    print("=" * 72)
    print("TECHGRADE AI — EXPERIMENT 013")
    print("HUMAN SCORE ANALYSIS AND CATEGORICAL ALIGNMENT")
    print("=" * 72)
    print(f"\nOS dataset:   {args.os_csv}")
    print(f"DBMS dataset: {args.dbms_csv}")
    print(f"\nResults saved to: {out_dir.resolve()}")
    print("\n[DATASET AUDIT]")
    display_cols = ["dataset", "rows", "unique_questions", "missing_label", "unknown_label", "missing_human_score", "non_numeric_human_score", "out_of_range_human_score", "possible_duplicate_question_answer_rows"]
    print(audit_df[display_cols].to_string(index=False))
    print("\n[OVERALL HUMAN SCORE SUMMARY]")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n[LABEL-LEVEL SCORE SUMMARY]")
    summary_cols = ["dataset", "label", "answer_count", "valid_numeric_score_count", "mean", "median", "std_dev", "minimum", "maximum", "percent_within_suggested_band"]
    print(by_label[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n[EXPLORATORY STATISTICAL TESTS]")
    print(tests.to_string(index=False))
    print(f"\nRows flagged for review: {len(quality_flags)}")
    print("Flagging does not mean the row is definitely incorrect; inspect the report.")
    print("Plots generated:")
    for path in plots:
        print(f"  - {path.name}")
    print("\nEXPERIMENT 013 COMPLETE")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
