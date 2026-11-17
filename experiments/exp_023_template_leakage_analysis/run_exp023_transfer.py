from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

PATHS = {
    "OS": Path.home() / "Downloads" / "OS_1000_synthetic_v2.csv",
    "DBMS": Path.home() / "Downloads" / "DBMS_1000_synthetic_v2.csv",
}

LABELS = [
    "Correct", "Partially Correct", "Incorrect",
    "Contradictory", "Irrelevant",
]

def load(path):
    df = pd.read_csv(path).fillna("")
    df = df[
        df["student_answer"].astype(str).str.strip().ne("")
        & df["question_id"].astype(str).str.strip().ne("")
        & df["label"].astype(str).str.strip().ne("")
    ].copy()
    df["answer_norm"] = (
        df["student_answer"].astype(str)
        .str.lower().str.split().str.join(" ")
    )
    return df.reset_index(drop=True)

def make_model():
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2
        ),
        LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=42
        ),
    )

def scores(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1_5_classes": f1_score(
            y, pred, labels=LABELS,
            average="macro", zero_division=0
        ),
    }

def within_domain(subject, df):
    results = []
    cv = GroupKFold(n_splits=5)

    for fold, (tr, te) in enumerate(
        cv.split(df, df["label"], df["question_id"]), 1
    ):
        train = df.iloc[tr]
        test = df.iloc[te].copy()

        model = make_model()
        model.fit(train["answer_norm"], train["label"])

        seen = set(train["answer_norm"])
        test["seen"] = test["answer_norm"].isin(seen)
        pred = model.predict(test["answer_norm"])

        results.append({
            "subject": subject,
            "fold": fold,
            "test_rows": len(test),
            "unseen_rows": int((~test["seen"]).sum()),
            **{
                f"all_{k}": v
                for k, v in scores(test["label"], pred).items()
            },
        })

        unseen = ~test["seen"].to_numpy()
        if unseen.any():
            results[-1].update({
                f"unseen_{k}": v
                for k, v in scores(
                    test.loc[unseen, "label"],
                    pred[unseen]
                ).items()
            })

    return results

def cross_domain(source, target, datasets):
    train = datasets[source]
    test = datasets[target]

    model = make_model()
    model.fit(train["answer_norm"], train["label"])
    pred = model.predict(test["answer_norm"])

    unseen = ~test["answer_norm"].isin(
        set(train["answer_norm"])
    ).to_numpy()

    result = {
        "source": source,
        "target": target,
        "train_rows": len(train),
        "test_rows": len(test),
        "unseen_test_rows": int(unseen.sum()),
        **scores(test["label"], pred),
    }

    if unseen.any():
        result.update({
            f"unseen_{k}": v
            for k, v in scores(
                test.loc[unseen, "label"],
                pred[unseen]
            ).items()
        })

    return result

def main():
    datasets = {k: load(v) for k, v in PATHS.items()}

    within = []
    for subject, df in datasets.items():
        within.extend(within_domain(subject, df))

    within_df = pd.DataFrame(within)
    within_df.to_csv(
        OUT / "exp023_unseen_answer_evaluation.csv",
        index=False,
    )

    print("\nUNSEEN ANSWERS — GROUPED CV")
    print(within_df.round(4).to_string(index=False))

    transfer = pd.DataFrame([
        cross_domain("OS", "DBMS", datasets),
        cross_domain("DBMS", "OS", datasets),
    ])
    transfer.to_csv(
        OUT / "exp023_cross_domain_evaluation.csv",
        index=False,
    )

    print("\nCROSS-DOMAIN TRANSFER")
    print(transfer.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
