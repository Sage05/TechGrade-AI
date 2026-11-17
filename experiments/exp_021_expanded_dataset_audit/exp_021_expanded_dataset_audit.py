
from pathlib import Path
from collections import Counter
import re
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# EXPERIMENT 021: EXPANDED DATASET AUDIT
# Read-only audit. Does not modify the input CSV files.
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DOWNLOADS = Path.home() / "Downloads"

INPUTS = {
    "OS": DOWNLOADS / "expanded_quality_os_2500.csv",
    "DBMS": DOWNLOADS / "expanded_quality_dbms_2500.csv",
}

OUTPUT_DIR = BASE_DIR / "experiments" / "exp_021_expanded_dataset_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIMILARITY_THRESHOLD = 0.82

REQUIRED_COLUMNS = [
    "question_text",
    "model_answer",
    "student_answer",
    "label",
    "human_score",
]

SOURCE_COLUMNS = [
    "entry_id",
    "answer_source",
    "annotator_id",
    "annotation_confidence",
    "annotator_1_label",
    "annotator_1_score",
    "annotator_2_label",
    "annotator_2_score",
    "agreement_status",
    "disagreement_reason",
    "annotator_note",
]


def clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_question(value):
    """Remove common presentation wrappers without deleting source text."""
    text = clean_text(value)

    # Repeatedly strip common prompt wrappers.
    patterns = [
        r"^give a direct answer to\s*",
        r"^provide a direct answer to\s*",
        r"^answer the following question\s*",
        r"^how would you answer this in a viva\s*",
        r"^in a viva explain\s*",
        r"^in an operating systems viva explain\s*",
        r"^in a database viva explain\s*",
        r"^explain the principle involved in\s*",
        r"^a junior developer asks\s*",
        r"^please explain\s*",
        r"^explain\s*",
        r"^question\s*",
    ]

    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            updated = re.sub(pattern, "", text).strip()
            if updated != text:
                text = updated
                changed = True

    return text


def load_dataset(subject, path):
    if not path.exists():
        raise FileNotFoundError(
            f"{subject} dataset not found: {path}\n"
            "Check the filename and Downloads folder."
        )

    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{subject}: missing required columns: {missing}")

    # Preserve original data; audit fields are added only in memory.
    df["_normalized_question"] = df["question_text"].map(normalize_question)
    df["_normalized_student_answer"] = df["student_answer"].map(clean_text)
    df["_normalized_model_answer"] = df["model_answer"].map(clean_text)

    return df


def label_score_audit(df):
    rows = []

    for label, group in df.groupby("label", dropna=False):
        scores = pd.to_numeric(group["human_score"], errors="coerce")
        rows.append({
            "label": str(label),
            "count": int(len(group)),
            "score_count": int(scores.notna().sum()),
            "score_mean": scores.mean(),
            "score_std": scores.std(),
            "score_min": scores.min(),
            "score_max": scores.max(),
            "unique_scores": int(scores.nunique()),
        })

    return pd.DataFrame(rows)


def exact_duplicate_report(df, subject):
    reports = []

    for column in [
        "_normalized_question",
        "_normalized_student_answer",
        "_normalized_model_answer",
    ]:
        counts = df[column].value_counts(dropna=False)
        duplicated = counts[counts > 1]

        for text, count in duplicated.items():
            reports.append({
                "subject": subject,
                "field": column.replace("_normalized_", ""),
                "duplicate_count": int(count),
                "text": text,
            })

    return pd.DataFrame(reports)


def near_duplicate_questions(df, subject):
    """
    Generate candidate pairs only. Similarity is not proof of equivalence.
    """
    questions = (
        df[["_normalized_question", "question_text"]]
        .drop_duplicates(subset=["_normalized_question"])
        .reset_index(drop=True)
    )

    texts = questions["_normalized_question"].fillna("").tolist()
    texts = [text if text else "empty_question" for text in texts]

    if len(texts) < 2:
        return pd.DataFrame(columns=[
            "subject", "similarity", "question_a", "question_b"
        ])

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(matrix, dense_output=True)

    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = float(similarity[i, j])
            if score >= SIMILARITY_THRESHOLD:
                pairs.append({
                    "subject": subject,
                    "similarity": round(score, 4),
                    "question_a": questions.iloc[i]["question_text"],
                    "question_b": questions.iloc[j]["question_text"],
                    "normalized_a": texts[i],
                    "normalized_b": texts[j],
                })

    result = pd.DataFrame(pairs)
    if not result.empty:
        result = result.sort_values("similarity", ascending=False)

    return result


def audit_one(subject, path):
    print(f"\nAuditing {subject}: {path}")
    df = load_dataset(subject, path)

    # Basic quality and diversity statistics.
    stats = {
        "subject": subject,
        "file": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns) - 3),
        "missing_question": int(df["question_text"].isna().sum()),
        "missing_student_answer": int(df["student_answer"].isna().sum()),
        "missing_model_answer": int(df["model_answer"].isna().sum()),
        "missing_label": int(df["label"].isna().sum()),
        "missing_score": int(df["human_score"].isna().sum()),
        "distinct_raw_questions": int(df["question_text"].nunique()),
        "distinct_normalized_questions": int(df["_normalized_question"].nunique()),
        "distinct_student_answers": int(df["_normalized_student_answer"].nunique()),
        "distinct_model_answers": int(df["_normalized_model_answer"].nunique()),
        "distinct_labels": int(df["label"].nunique()),
    }

    scores = pd.to_numeric(df["human_score"], errors="coerce")
    stats["score_mean"] = float(scores.mean()) if scores.notna().any() else None
    stats["score_min"] = float(scores.min()) if scores.notna().any() else None
    stats["score_max"] = float(scores.max()) if scores.notna().any() else None

    stats_path = OUTPUT_DIR / f"exp021_{subject.lower()}_summary.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Full label/score relationship.
    label_report = label_score_audit(df)
    label_report.to_csv(
        OUTPUT_DIR / f"exp021_{subject.lower()}_label_score_audit.csv",
        index=False,
    )

    # Data source distribution, when the field exists.
    if "answer_source" in df.columns:
        source_report = (
            df["answer_source"].fillna("[MISSING]")
            .value_counts()
            .rename_axis("answer_source")
            .reset_index(name="count")
        )
        source_report.to_csv(
            OUTPUT_DIR / f"exp021_{subject.lower()}_source_distribution.csv",
            index=False,
        )

    # Human annotation coverage.
    annotation_fields = [
        col for col in [
            "annotator_1_label",
            "annotator_1_score",
            "annotator_2_label",
            "annotator_2_score",
            "annotator_id",
            "annotation_confidence",
        ] if col in df.columns
    ]

    annotation_rows = []
    for col in annotation_fields:
        annotation_rows.append({
            "field": col,
            "non_missing_count": int(df[col].notna().sum()),
            "coverage_percent": round(
                float(df[col].notna().mean() * 100), 2
            ),
        })

    pd.DataFrame(annotation_rows).to_csv(
        OUTPUT_DIR / f"exp021_{subject.lower()}_annotation_coverage.csv",
        index=False,
    )

    # Exact duplicates.
    duplicates = exact_duplicate_report(df, subject)
    duplicates.to_csv(
        OUTPUT_DIR / f"exp021_{subject.lower()}_duplicate_texts.csv",
        index=False,
    )

    # Candidate near-duplicate question pairs.
    near_duplicates = near_duplicate_questions(df, subject)
    near_duplicates.to_csv(
        OUTPUT_DIR / f"exp021_{subject.lower()}_near_duplicate_questions.csv",
        index=False,
    )

    # Inspect whether a normalized question has conflicting labels/scores.
    conflict_rows = []
    for question, group in df.groupby("_normalized_question", dropna=False):
        labels = sorted(group["label"].dropna().astype(str).unique().tolist())
        group_scores = pd.to_numeric(group["human_score"], errors="coerce")
        unique_scores = sorted(group_scores.dropna().unique().tolist())

        if len(labels) > 1 or len(unique_scores) > 1:
            conflict_rows.append({
                "normalized_question": question,
                "row_count": int(len(group)),
                "distinct_labels": " | ".join(labels),
                "distinct_scores": " | ".join(map(str, unique_scores)),
                "example_questions": " || ".join(
                    group["question_text"].dropna().astype(str).head(5)
                ),
            })

    pd.DataFrame(conflict_rows).to_csv(
        OUTPUT_DIR / f"exp021_{subject.lower()}_question_label_score_variation.csv",
        index=False,
    )

    print(json.dumps(stats, indent=2))
    print(f"Candidate near-duplicate pairs: {len(near_duplicates)}")
    print(f"Reports saved in: {OUTPUT_DIR}")

    return stats


def main():
    all_stats = []

    for subject, path in INPUTS.items():
        all_stats.append(audit_one(subject, path))

    pd.DataFrame(all_stats).to_csv(
        OUTPUT_DIR / "exp021_combined_summary.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("EXP 021 AUDIT COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Original datasets were not modified.")
    print("Near-duplicate pairs are candidates for review, not confirmed duplicates.")
    print("=" * 65)


if __name__ == "__main__":
    main()
