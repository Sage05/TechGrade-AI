"""
TechGrade AI — Presentation Demo

Purpose
-------
A simple local demonstration interface for the current TechGrade AI
grading pipeline.

This file intentionally does NOT modify the research model or any
project source files. It trains the current Exp 008 pipeline on the
training portion of the selected dataset and lets the presenter:

    1. Load real annotated examples.
    2. Evaluate a custom question/answer.
    3. Inspect label, confidence, class probabilities and concept evidence.

Run from the TechGrade-AI project root:

    python techgrade_demo.py

Optional dataset paths can be changed in DATASETS below.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

import pandas as pd

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

DATASETS = {
    "Operating Systems": Path(
        Path.home() / "Downloads" / "revised_sample_os.csv"
    ),
    "DBMS": Path(
        Path.home() / "Downloads" / "revised_sample_dbms.csv"
    ),
}


# ============================================================
# MODEL
# ============================================================

class TechGradeDemoModel:
    """Loads the existing TechGrade pipeline once and reuses it."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{csv_path}"
            )

        self.df = pd.read_csv(csv_path)

        train_df, _, _ = question_level_split(
            self.df,
            random_state=42,
        )

        self.tfidf = TfidfBaseline()

        self.tfidf.fit(
            (
                train_df["question_text"].astype(str)
                + " "
                + train_df["model_answer"].astype(str)
            ).tolist()
        )

        self.embedding = EmbeddingBaseline()
        self.nli = NLIBaseline()
        self.concept_matcher = ConceptMatcher()

        self.extractor = FeatureExtractor(
            tfidf_model=self.tfidf,
            embedding_model=self.embedding,
            nli_model=self.nli,
            concept_matcher=self.concept_matcher,
        )

        X_train = []
        y_train = []

        for _, row in train_df.iterrows():

            features = self.extractor.extract(
                question=row["question_text"],
                reference_answer=row["model_answer"],
                student_answer=row["student_answer"],
                required_concepts=row["required_concepts"],
            )

            X_train.append(features.values)
            y_train.append(row["label"])

        self.grader = HybridGrader(self.extractor)

        self.grader.fit(
            X_train,
            y_train,
        )

    def evaluate(
        self,
        question: str,
        reference_answer: str,
        student_answer: str,
        required_concepts: str,
    ):
        return self.grader.predict(
            question=question,
            reference_answer=reference_answer,
            student_answer=student_answer,
            required_concepts=required_concepts,
        )

    def get_examples(self) -> dict[str, dict]:
        """
        Return one high-confidence correctly classified example
        for each available label.
        """
        examples = {}

        for _, row in self.df.iterrows():

            features = self.extractor.extract(
                question=row["question_text"],
                reference_answer=row["model_answer"],
                student_answer=row["student_answer"],
                required_concepts=row["required_concepts"],
            )

            result = self.grader.predict_features_with_details(
                features
            )

            actual = str(row["label"])

            if actual != result.label:
                continue

            current = examples.get(actual)

            if (
                current is None
                or result.confidence > current["confidence"]
            ):
                examples[actual] = {
                    "question": str(row["question_text"]),
                    "reference": str(row["model_answer"]),
                    "student": str(row["student_answer"]),
                    "required_concepts": str(
                        row["required_concepts"]
                    ),
                    "result": result,
                }

        return examples


# ============================================================
# APPLICATION
# ============================================================

class TechGradeDemoApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(
            "TechGrade AI — Technical Answer Evaluation"
        )
        self.root.geometry("1400x900")
        self.root.minsize(1150, 760)
        self.root.configure(bg="#eef1f5")

        self.models: dict[str, TechGradeDemoModel] = {}
        self.examples: dict[str, dict[str, dict]] = {}

        self.build_styles()
        self.build_header()
        self.build_controls()
        self.build_input_area()
        self.build_result_area()
        self.build_footer()

        self.load_dataset("Operating Systems")

    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------

    def build_styles(self):

        style = ttk.Style()

        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "Demo.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 8),
        )

        style.configure(
            "Dataset.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 7),
        )

        style.configure(
            "TCombobox",
            padding=5,
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    def build_header(self):

        frame = tk.Frame(
            self.root,
            bg="#ffffff",
        )

        frame.pack(
            fill="x",
            padx=18,
            pady=(18, 8),
        )

        tk.Label(
            frame,
            text="TECHGRADE AI",
            bg="#ffffff",
            fg="#16202a",
            font=("Segoe UI", 25, "bold"),
        ).pack(
            anchor="w",
            padx=24,
            pady=(18, 0),
        )

        tk.Label(
            frame,
            text="Automated Technical Answer Evaluation using NLP",
            bg="#ffffff",
            fg="#63717d",
            font=("Segoe UI", 11),
        ).pack(
            anchor="w",
            padx=24,
            pady=(2, 18),
        )

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    def build_controls(self):

        outer = tk.Frame(
            self.root,
            bg="#eef1f5",
        )

        outer.pack(
            fill="x",
            padx=18,
            pady=6,
        )

        tk.Label(
            outer,
            text="DATASET",
            bg="#eef1f5",
            fg="#374151",
            font=("Segoe UI", 9, "bold"),
        ).pack(
            side="left",
            padx=(0, 8),
        )

        self.dataset_var = tk.StringVar(
            value="Operating Systems"
        )

        dataset_box = ttk.Combobox(
            outer,
            textvariable=self.dataset_var,
            values=list(DATASETS.keys()),
            state="readonly",
            width=22,
        )

        dataset_box.pack(
            side="left",
            padx=(0, 12),
        )

        ttk.Button(
            outer,
            text="Load Dataset",
            command=self.on_dataset_change,
            style="Dataset.TButton",
        ).pack(
            side="left",
        )

        tk.Label(
            outer,
            text="   |   DEMO EXAMPLES",
            bg="#eef1f5",
            fg="#374151",
            font=("Segoe UI", 9, "bold"),
        ).pack(
            side="left",
            padx=(20, 8),
        )

        self.demo_var = tk.StringVar(
            value="Select example"
        )

        self.demo_box = ttk.Combobox(
            outer,
            textvariable=self.demo_var,
            values=[],
            state="readonly",
            width=22,
        )

        self.demo_box.pack(
            side="left",
            padx=(0, 10),
        )

        ttk.Button(
            outer,
            text="Load Example",
            command=self.load_selected_example,
            style="Dataset.TButton",
        ).pack(
            side="left",
        )

    # --------------------------------------------------------
    # Input area
    # --------------------------------------------------------

    def build_input_area(self):

        container = tk.Frame(
            self.root,
            bg="#eef1f5",
        )

        container.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=6,
        )

        left = tk.Frame(
            container,
            bg="#ffffff",
        )

        left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 6),
        )

        right = tk.Frame(
            container,
            bg="#ffffff",
            width=450,
        )

        right.pack(
            side="right",
            fill="y",
            padx=(6, 0),
        )

        right.pack_propagate(False)

        self.question_box = self.add_textbox(
            left,
            "QUESTION",
            height=4,
        )

        self.reference_box = self.add_textbox(
            left,
            "REFERENCE / MODEL ANSWER",
            height=7,
        )

        self.student_box = self.add_textbox(
            left,
            "STUDENT ANSWER",
            height=8,
        )

        self.concepts_box = self.add_textbox(
            left,
            "REQUIRED CONCEPTS  (separate with |)",
            height=3,
        )

        evaluate_frame = tk.Frame(
            left,
            bg="#ffffff",
        )

        evaluate_frame.pack(
            fill="x",
            padx=20,
            pady=(4, 20),
        )

        ttk.Button(
            evaluate_frame,
            text="EVALUATE ANSWER",
            command=self.evaluate_current,
            style="Demo.TButton",
        ).pack(
            side="left",
        )

        ttk.Button(
            evaluate_frame,
            text="CLEAR",
            command=self.clear_inputs,
            style="Demo.TButton",
        ).pack(
            side="left",
            padx=8,
        )

    def add_textbox(
        self,
        parent,
        title,
        height,
    ):

        frame = tk.Frame(
            parent,
            bg="#ffffff",
        )

        frame.pack(
            fill="x",
            padx=20,
            pady=(12, 2),
        )

        tk.Label(
            frame,
            text=title,
            bg="#ffffff",
            fg="#263238",
            font=("Segoe UI", 9, "bold"),
        ).pack(
            anchor="w",
            pady=(0, 5),
        )

        text = tk.Text(
            frame,
            height=height,
            wrap="word",
            font=("Segoe UI", 10),
            bg="#f7f8fa",
            fg="#202124",
            relief="flat",
            padx=12,
            pady=9,
        )

        text.pack(
            fill="x",
        )

        return text

    # --------------------------------------------------------
    # Result area
    # --------------------------------------------------------

    def build_result_area(self):

        # Find the right-hand frame created above.
        # It is the second child of the main container.
        container = self.root.winfo_children()[-1]

        self.result_panel = container.winfo_children()[1]

        tk.Label(
            self.result_panel,
            text="EVALUATION RESULT",
            bg="#ffffff",
            fg="#263238",
            font=("Segoe UI", 10, "bold"),
        ).pack(
            anchor="w",
            padx=20,
            pady=(20, 8),
        )

        self.result_label = tk.Label(
            self.result_panel,
            text="READY",
            bg="#ffffff",
            fg="#16202a",
            font=("Segoe UI", 25, "bold"),
        )

        self.result_label.pack(
            anchor="w",
            padx=20,
        )

        self.confidence_label = tk.Label(
            self.result_panel,
            text="Confidence: —",
            bg="#ffffff",
            fg="#63717d",
            font=("Segoe UI", 11),
        )

        self.confidence_label.pack(
            anchor="w",
            padx=20,
            pady=(2, 18),
        )

        tk.Label(
            self.result_panel,
            text="CLASS PROBABILITIES",
            bg="#ffffff",
            fg="#263238",
            font=("Segoe UI", 9, "bold"),
        ).pack(
            anchor="w",
            padx=20,
            pady=(0, 8),
        )

        self.probability_text = tk.Text(
            self.result_panel,
            height=8,
            wrap="word",
            font=("Consolas", 9),
            bg="#f7f8fa",
            fg="#202124",
            relief="flat",
            padx=12,
            pady=10,
        )

        self.probability_text.pack(
            fill="x",
            padx=20,
        )

        tk.Label(
            self.result_panel,
            text="CONCEPT EVIDENCE",
            bg="#ffffff",
            fg="#263238",
            font=("Segoe UI", 9, "bold"),
        ).pack(
            anchor="w",
            padx=20,
            pady=(18, 8),
        )

        concept_scroll_frame = tk.Frame(
            self.result_panel,
            bg="#ffffff",
        )

        concept_scroll_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=(0, 20),
        )

        self.concept_text = tk.Text(
            concept_scroll_frame,
            wrap="word",
            font=("Segoe UI", 9),
            bg="#f7f8fa",
            fg="#202124",
            relief="flat",
            padx=12,
            pady=10,
        )

        scrollbar = ttk.Scrollbar(
            concept_scroll_frame,
            orient="vertical",
            command=self.concept_text.yview,
        )

        self.concept_text.configure(
            yscrollcommand=scrollbar.set
        )

        self.concept_text.pack(
            side="left",
            fill="both",
            expand=True,
        )

        scrollbar.pack(
            side="right",
            fill="y",
        )

        self.set_readonly(
            self.probability_text
        )

        self.set_readonly(
            self.concept_text
        )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    def build_footer(self):

        tk.Label(
            self.root,
            text="TechGrade AI  •  Hybrid NLP Grading Pipeline",
            bg="#eef1f5",
            fg="#7b8794",
            font=("Segoe UI", 9),
        ).pack(
            pady=(2, 10),
        )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def set_readonly(widget):

        widget.configure(
            state="disabled"
        )

    @staticmethod
    def set_text(widget, value):

        widget.configure(
            state="normal"
        )

        widget.delete(
            "1.0",
            "end",
        )

        widget.insert(
            "1.0",
            value,
        )

        widget.configure(
            state="disabled"
        )

    def get_text(self, widget):

        return widget.get(
            "1.0",
            "end",
        ).strip()

    # --------------------------------------------------------
    # Dataset loading
    # --------------------------------------------------------

    def load_dataset(self, dataset_name):

        path = DATASETS[dataset_name]

        try:
            self.root.title(
                f"TechGrade AI — Loading {dataset_name}..."
            )

            self.root.update_idletasks()

            if dataset_name not in self.models:

                self.models[dataset_name] = (
                    TechGradeDemoModel(path)
                )

                self.examples[dataset_name] = (
                    self.models[dataset_name].get_examples()
                )

            available = sorted(
                self.examples[dataset_name].keys()
            )

            self.demo_box["values"] = available

            if available:
                self.demo_var.set(
                    "Select example"
                )

            self.root.title(
                "TechGrade AI — Technical Answer Evaluation"
            )

        except Exception as exc:

            messagebox.showerror(
                "TechGrade AI — Dataset Error",
                str(exc),
            )

    def on_dataset_change(self):

        self.load_dataset(
            self.dataset_var.get()
        )

    # --------------------------------------------------------
    # Example loading
    # --------------------------------------------------------

    def load_selected_example(self):

        dataset = self.dataset_var.get()
        label = self.demo_var.get()

        if (
            not label
            or label == "Select example"
        ):
            return

        example = self.examples[dataset][label]

        self.set_text(
            self.question_box,
            example["question"],
        )

        self.set_text(
            self.reference_box,
            example["reference"],
        )

        self.set_text(
            self.student_box,
            example["student"],
        )

        self.set_text(
            self.concepts_box,
            example["required_concepts"],
        )

        self.display_result(
            example["result"]
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def evaluate_current(self):

        question = self.get_text(
            self.question_box
        )

        reference = self.get_text(
            self.reference_box
        )

        student = self.get_text(
            self.student_box
        )

        concepts = self.get_text(
            self.concepts_box
        )

        if not question:
            messagebox.showwarning(
                "Missing Input",
                "Please enter a question.",
            )
            return

        if not reference:
            messagebox.showwarning(
                "Missing Input",
                "Please enter the reference answer.",
            )
            return

        if not student:
            messagebox.showwarning(
                "Missing Input",
                "Please enter the student answer.",
            )
            return

        try:

            dataset = self.dataset_var.get()

            model = self.models[dataset]

            result = model.evaluate(
                question=question,
                reference_answer=reference,
                student_answer=student,
                required_concepts=concepts,
            )

            self.display_result(result)

        except Exception as exc:

            messagebox.showerror(
                "Evaluation Error",
                str(exc),
            )

    # --------------------------------------------------------
    # Result display
    # --------------------------------------------------------

    def display_result(self, result):

        label = result.label.upper()

        self.result_label.config(
            text=label,
        )

        self.confidence_label.config(
            text=(
                f"Confidence: "
                f"{result.confidence * 100:.2f}%"
            ),
        )

        probability_lines = []

        for name, probability in sorted(
            result.probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        ):

            probability_lines.append(
                f"{name:<20} "
                f"{probability * 100:>7.2f}%"
            )

        self.set_text(
            self.probability_text,
            "\n".join(probability_lines),
        )

        concept_lines = []

        if not result.concept_predictions:

            concept_lines.append(
                "No required concepts supplied."
            )

        else:

            for prediction in result.concept_predictions:

                if prediction.decision == "COVERED":
                    marker = "✓"
                elif prediction.decision == "INCORRECT":
                    marker = "!"
                else:
                    marker = "○"

                concept_lines.append(
                    f"{marker} {prediction.concept}"
                )

                concept_lines.append(
                    f"   Decision      : "
                    f"{prediction.decision}"
                )

                concept_lines.append(
                    f"   Similarity    : "
                    f"{prediction.semantic_similarity:.4f}"
                )

                concept_lines.append(
                    f"   Entailment    : "
                    f"{prediction.entailment:.4f}"
                )

                concept_lines.append(
                    f"   Contradiction : "
                    f"{prediction.contradiction:.4f}"
                )

                concept_lines.append("")

        self.set_text(
            self.concept_text,
            "\n".join(concept_lines),
        )

    # --------------------------------------------------------
    # Clear
    # --------------------------------------------------------

    def clear_inputs(self):

        for widget in [
            self.question_box,
            self.reference_box,
            self.student_box,
            self.concepts_box,
        ]:

            widget.delete(
                "1.0",
                "end",
            )

        self.result_label.config(
            text="READY"
        )

        self.confidence_label.config(
            text="Confidence: —"
        )

        self.set_text(
            self.probability_text,
            "",
        )

        self.set_text(
            self.concept_text,
            "",
        )


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    root = tk.Tk()

    TechGradeDemoApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()
