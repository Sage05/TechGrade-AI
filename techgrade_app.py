
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.utils.data_split import question_level_split
from src.models.tfidf_baseline import TfidfBaseline
from src.models.embedding_baseline import EmbeddingBaseline
from src.models.nli_baseline import NLIBaseline
from src.models.concept_matcher import ConceptMatcher
from src.grading.feature_extractor import FeatureExtractor
from src.grading.hybrid_grader import HybridGrader


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TechGrade AI | Answer Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DOWNLOADS = Path.home() / "Downloads"

DATASETS = {
    "Operating Systems": DOWNLOADS / "revised_sample_os.csv",
    "DBMS": DOWNLOADS / "revised_sample_dbms.csv",
}

REQUIRED_COLUMNS = [
    "question_text",
    "model_answer",
    "student_answer",
    "required_concepts",
    "label",
]


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263246;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] {
        color: #111827 !important;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.8px;
        margin-bottom: 0;
    }

    .brand-accent {
        color: #8b9cff;
    }

    .sidebar-caption {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 3px;
        margin-bottom: 22px;
    }

    .hero {
        background: linear-gradient(115deg, #172554 0%, #312e81 56%, #4f46e5 100%);
        padding: 30px 34px;
        border-radius: 20px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(49, 46, 129, 0.15);
    }

    .hero-kicker {
        color: #c7d2fe;
        font-size: 0.76rem;
        letter-spacing: 2px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: white;
        font-size: 2.25rem;
        line-height: 1.12;
        letter-spacing: -1px;
        margin: 0 0 12px 0;
    }

    .hero p {
        color: #e0e7ff;
        font-size: 1rem;
        margin: 0;
        max-width: 800px;
    }

    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        color: #172033;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 5px 0 12px 0;
    }

    .panel {
        background: #ffffff;
        padding: 22px;
        border: 1px solid #e7eaf1;
        border-radius: 16px;
        margin-bottom: 16px;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.85rem;
    }

    .result-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: #312e81;
        overflow-wrap: anywhere;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: #4f46e5;
        color: white;
        border: 1px solid #4f46e5;
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #3730a3;
        border-color: #3730a3;
        color: white;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e7eaf1;
        padding: 16px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #6b7280;
    }

    div[data-testid="stMetricValue"] {
        color: #172033;
    }

    div[data-testid="stTextArea"] textarea {
        border: 1px solid #dbe1ec;
        border-radius: 10px;
        background: #fbfcfe;
        color: #172033;
    }

    .footer {
        color: #8a94a6;
        font-size: 0.8rem;
        text-align: center;
        padding-top: 18px;
    }

    /* Keep form labels readable against the light canvas */
[data-testid="stAppViewContainer"] .main
[data-testid="stWidgetLabel"],
[data-testid="stAppViewContainer"] .main
[data-testid="stWidgetLabel"] *,
[data-testid="stAppViewContainer"] .main
[data-testid="stMetricLabel"],
[data-testid="stAppViewContainer"] .main
[data-testid="stMetricLabel"] * {
    color: #344054 !important;
}

/* Make text inputs readable */
[data-testid="stAppViewContainer"] .main textarea {
    color: #172033 !important;
    background-color: #ffffff !important;
}

/* Fix the sidebar subject and example dropdowns */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #dbe1ec !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #172033 !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_text(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


# ============================================================
# MODEL INITIALIZATION
# ============================================================

@st.cache_resource(show_spinner=False)
def build_model(
    subject: str,
    csv_path: str,
    file_mtime: float,
):
    """
    Build and cache the Exp 008 hybrid grader.

    The classifier is fitted in memory on the training split.
    It does not load a saved classifier checkpoint.
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) < 10:
        raise ValueError("The selected dataset has too few rows.")

    train_df, validation_df, test_df = question_level_split(
        df,
        random_state=42,
    )

    if train_df.empty:
        raise ValueError("The question-level training split is empty.")

    # Fit TF-IDF vocabulary using training data only.
    train_corpus = (
        train_df["question_text"].map(safe_text).tolist()
        + train_df["model_answer"].map(safe_text).tolist()
        + train_df["student_answer"].map(safe_text).tolist()
    )

    tfidf = TfidfBaseline()
    tfidf.fit(train_corpus)

    embedding = EmbeddingBaseline()
    nli = NLIBaseline()
    concept_matcher = ConceptMatcher()

    extractor = FeatureExtractor(
        tfidf_model=tfidf,
        embedding_model=embedding,
        nli_model=nli,
        concept_matcher=concept_matcher,
    )

    feature_rows = []
    labels = []

    progress = st.progress(
        0,
        text=f"Preparing {subject} training features...",
    )

    total = len(train_df)

    for i, (_, row) in enumerate(train_df.iterrows(), start=1):
        features = extractor.extract(
            question=safe_text(row["question_text"]),
            reference_answer=safe_text(row["model_answer"]),
            student_answer=safe_text(row["student_answer"]),
            required_concepts=safe_text(row["required_concepts"]),
        )

        feature_rows.append(features.values)
        labels.append(safe_text(row["label"]))

        if i % 5 == 0 or i == total:
            progress.progress(
                min(i / total, 1.0),
                text=f"Preparing training features: {i}/{total}",
            )

    progress.progress(1.0, text="Training the hybrid classifier...")

    X_train = np.vstack(feature_rows)

    grader = HybridGrader(extractor)
    grader.fit(X=X_train, labels=labels)

    progress.empty()

    return {
        "subject": subject,
        "df": df,
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df.reset_index(drop=True),
        "grader": grader,
    }


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "question_input": "",
    "reference_input": "",
    "student_input": "",
    "concepts_input": "",
    "last_result": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">TECHGRADE<span class="brand-accent"> AI</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-caption">Technical Answer Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Workspace")

    subject = st.selectbox(
        "Select subject",
        options=list(DATASETS.keys()),
    )

    dataset_path = DATASETS[subject]

    st.divider()

    st.markdown("### Demo examples")
    st.caption("Examples below come from the held-out split.")

    model_bundle = None

    if not dataset_path.exists():
        st.error(f"Dataset not found:\n{dataset_path}")
        st.stop()

    try:
        with st.spinner(
            "Initializing the hybrid model. First load may take a few minutes..."
        ):
            model_bundle = build_model(
                subject,
                str(dataset_path),
                dataset_path.stat().st_mtime,
            )
    except Exception as exc:
        st.error("The grading pipeline could not be initialized.")
        st.exception(exc)
        st.stop()

    test_df = model_bundle["test_df"]

    if not test_df.empty:
        example_indices = list(range(len(test_df)))

        def format_example(i):
            row = test_df.iloc[i]
            question = safe_text(row["question_text"])
            if len(question) > 65:
                question = question[:62] + "..."
            return f"Example {i + 1}: {question}"

        selected_example = st.selectbox(
            "Choose a held-out question",
            options=example_indices,
            format_func=format_example,
        )

        if st.button("Load selected example", use_container_width=True):
            example_row = test_df.iloc[selected_example]

            st.session_state["question_input"] = safe_text(
                example_row["question_text"]
            )
            st.session_state["reference_input"] = safe_text(
                example_row["model_answer"]
            )
            st.session_state["student_input"] = safe_text(
                example_row["student_answer"]
            )
            st.session_state["concepts_input"] = safe_text(
                example_row["required_concepts"]
            )
            st.session_state["last_result"] = None
            st.rerun()
    else:
        st.info("No held-out examples are available in this split.")

    st.divider()

    st.markdown("### Current pipeline")
    st.markdown(
        """
        - TF-IDF similarity
        - Sentence embeddings
        - Natural language inference
        - Concept matching
        - Logistic Regression classifier
        """
    )

    st.caption("Exp 008 · Concept-aware hybrid grader")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Research prototype · Exp 008</div>
        <h1>Understand the answer.<br>Evaluate the knowledge.</h1>
        <p>
            Evaluate technical student responses using a hybrid NLP pipeline
            that combines semantic similarity, entailment signals and
            concept-level evidence.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Summary metrics from selected dataset.
df = model_bundle["df"]

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Dataset records", f"{len(df):,}")
metric2.metric(
    "Distinct question strings",
    f"{df['question_text'].nunique():,}",
)
metric3.metric(
    "Available labels",
    f"{df['label'].nunique():,}",
)

st.markdown("")

input_col, result_col = st.columns([1.08, 0.92], gap="large")


# ============================================================
# INPUT PANEL
# ============================================================

with input_col:
    st.markdown(
        '<div class="section-heading">01 / Answer submission</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="small-muted">
            Provide the question, reference answer and student response.
            Then run the current hybrid grading pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    with st.form("grading_form", clear_on_submit=False):
        st.text_area(
            "Technical question",
            key="question_input",
            height=105,
            placeholder="Enter the question the student was asked...",
        )

        st.text_area(
            "Reference / model answer",
            key="reference_input",
            height=145,
            placeholder="Enter the expected answer or marking reference...",
        )

        st.text_area(
            "Student answer",
            key="student_input",
            height=180,
            placeholder="Paste the student's response here...",
        )

        st.text_area(
            "Required concepts",
            key="concepts_input",
            height=85,
            placeholder="Example: virtual memory | paging | address space",
            help="Use the separator format expected by your existing concept matcher.",
        )

        submitted = st.form_submit_button(
            "Evaluate student answer",
            use_container_width=True,
        )

    if submitted:
        question = safe_text(st.session_state["question_input"])
        reference = safe_text(st.session_state["reference_input"])
        student = safe_text(st.session_state["student_input"])
        concepts = safe_text(st.session_state["concepts_input"])

        if not question or not reference or not student:
            st.session_state["last_result"] = None
            st.warning(
                "Please provide a question, reference answer and student answer."
            )
        else:
            try:
                with st.spinner("Evaluating answer..."):
                    result = model_bundle["grader"].predict(
                        question=question,
                        reference_answer=reference,
                        student_answer=student,
                        required_concepts=concepts,
                    )

                st.session_state["last_result"] = {
                    "subject": subject,
                    "label": result.label,
                    "confidence": float(result.confidence),
                    "probabilities": result.probabilities,
                    "concept_predictions": [
                        {
                            "concept": p.concept,
                            "decision": p.decision,
                            "semantic_similarity": float(
                                p.semantic_similarity
                            ),
                            "entailment": float(p.entailment),
                            "contradiction": float(p.contradiction),
                        }
                        for p in result.concept_predictions
                    ],
                }

            except Exception as exc:
                st.session_state["last_result"] = None
                st.error("Evaluation failed.")
                st.exception(exc)


# ============================================================
# RESULTS PANEL
# ============================================================

with result_col:
    st.markdown(
        '<div class="section-heading">02 / Evaluation results</div>',
        unsafe_allow_html=True,
    )

    last_result = st.session_state.get("last_result")

    if not last_result:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size:2rem;margin-bottom:10px;">◈</div>
                <div style="font-size:1.2rem;font-weight:700;color:#172033;">
                    Ready to evaluate
                </div>
                <div style="color:#6b7280;margin-top:8px;line-height:1.6;">
                    Submit an answer or load a held-out example from the
                    sidebar. The model's prediction, class probabilities
                    and concept evidence will appear here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif last_result["subject"] != subject:
        st.info(
            f"The previous result belongs to {last_result['subject']}. "
            "Submit a new answer to evaluate it with the selected subject."
        )

    else:
        st.markdown(
            """
            <div class="panel">
                <div class="small-muted">PREDICTED CATEGORY</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="result-label">{last_result["label"]}</div>',
            unsafe_allow_html=True,
        )

        confidence_col, subject_col = st.columns(2)

        confidence_col.metric(
            "Model confidence",
            f"{last_result['confidence']:.1%}",
        )
        subject_col.metric("Subject", subject)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Class probabilities")

        probability_items = sorted(
            last_result["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for label, probability in probability_items:
            st.progress(
                max(0.0, min(1.0, float(probability))),
                text=f"{label}  ·  {probability:.1%}",
            )

        st.markdown("")
        st.markdown("#### Concept-level evidence")

        concepts = last_result["concept_predictions"]

        if concepts:
            concept_df = pd.DataFrame(concepts).rename(
                columns={
                    "concept": "Concept",
                    "decision": "Decision",
                    "semantic_similarity": "Similarity",
                    "entailment": "Entailment",
                    "contradiction": "Contradiction",
                }
            )

            for col in [
                "Similarity",
                "Entailment",
                "Contradiction",
            ]:
                concept_df[col] = concept_df[col].round(3)

            st.dataframe(
                concept_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No concept evidence was returned. Enter required concepts "
                "in the format expected by the concept matcher."
            )

        with st.expander("How should I interpret this result?"):
            st.write(
                "The predicted category is the class chosen by the current "
                "hybrid classifier. The confidence value is the maximum "
                "predicted class probability; it is not a guarantee of "
                "correctness or a validated measure of grading reliability."
            )
            st.write(
                "Concept decisions and similarity/entailment values are "
                "signals from the existing feature pipeline, not independent "
                "proof that an answer is correct."
            )


# ============================================================
# DATASET DETAILS
# ============================================================

st.divider()

with st.expander("Dataset and split details"):
    st.write(f"**Selected dataset:** {DATASETS[subject]}")
    st.write(f"**Total rows:** {len(df)}")
    st.write(f"**Training rows:** {len(model_bundle['train_df'])}")
    st.write(f"**Validation rows:** {len(model_bundle['validation_df'])}")
    st.write(f"**Held-out rows:** {len(model_bundle['test_df'])}")

    st.markdown("**Dataset label distribution**")
    distribution = (
        df["label"]
        .fillna("[MISSING]")
        .value_counts()
        .rename_axis("Label")
        .reset_index(name="Rows")
    )

    st.dataframe(
        distribution,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "This demonstration fits the classifier on its training split in "
        "memory. It does not load a saved classifier checkpoint. "
        "Results are experimental and depend on dataset quality."
    )

st.markdown(
    """
    <div class="footer">
        TECHGRADE AI · Concept-Aware Hybrid NLP Grader · Research Prototype
    </div>
    """,
    unsafe_allow_html=True,
)
