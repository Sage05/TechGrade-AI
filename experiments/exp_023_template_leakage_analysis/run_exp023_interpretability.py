from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
OUT.mkdir(parents=True, exist_ok=True)

paths = {
    "OS": Path.home() / "Downloads" / "OS_1000_synthetic_v2.csv",
    "DBMS": Path.home() / "Downloads" / "DBMS_1000_synthetic_v2.csv",
}

frames = []
for subject, path in paths.items():
    df = pd.read_csv(path).fillna("")
    df["subject"] = subject
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
data["answer"] = (
    data["student_answer"]
    .astype(str).str.lower().str.split().str.join(" ")
)

# Use word patterns for human-readable interpretation.
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=3,
    max_features=20000,
)

X = vectorizer.fit_transform(data["answer"])
y = data["label"]

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)
model.fit(X, y)

terms = vectorizer.get_feature_names_out()
rows = []

for i, label in enumerate(model.classes_):
    strongest = model.coef_[i].argsort()[-15:][::-1]
    print(f"\n{'=' * 55}")
    print(f"LABEL: {label}")
    print(f"{'=' * 55}")

    for j in strongest:
        term = terms[j]
        weight = float(model.coef_[i, j])
        print(f"{term:<35} {weight:.4f}")
        rows.append({
            "label": label,
            "term": term,
            "coefficient": weight,
        })

pd.DataFrame(rows).to_csv(
    OUT / "exp023_label_predictive_terms.csv",
    index=False,
)

print("\nSaved label-predictive terms.")
