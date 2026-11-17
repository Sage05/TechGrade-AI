# TechGrade AI — Experiment 013
## Human Score Analysis and Categorical Alignment

### Purpose
This experiment analyses the existing OS and DBMS human scores before we design either a deterministic 0–10 scoring baseline (Exp 014) or a learned regression baseline (Exp 015).

It:
- audits required fields, labels, numerical score validity, duplicates, and missing values;
- describes the overall score distribution by dataset;
- describes score distributions within each of the five categorical labels;
- flags scores outside the project's suggested label-specific bands for human review;
- runs an exploratory Kruskal–Wallis test when SciPy is available;
- produces CSV summaries, a Markdown report, and plots.

**It does not** train a model, automatically correct annotations, enforce suggested bands, or overwrite the original CSVs.

### Add to the project
Copy the entire `experiments/exp_013_score_analysis` folder into the existing `M:\TechGrade-AI\experiments` directory. Keep the existing experiments unchanged.

The resulting path should be:

```text
M:\TechGrade-AI\experiments\exp_013_score_analysis\run_experiment.py
```

### Run from the project root
Activate your existing virtual environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run:

```powershell
python -m experiments.exp_013_score_analysis.run_experiment
```

The script defaults to the dataset paths previously used by the project:

```text
C:\Users\Ayuj Shingte\Downloads\revised_sample_os.csv
C:\Users\Ayuj Shingte\Downloads\revised_sample_dbms.csv
```

If the CSVs are elsewhere, supply the paths explicitly:

```powershell
python -m experiments.exp_013_score_analysis.run_experiment `
  --os-csv "C:\path\to\revised_sample_os.csv" `
  --dbms-csv "C:\path\to\revised_sample_dbms.csv"
```

To choose another output location:

```powershell
python -m experiments.exp_013_score_analysis.run_experiment `
  --os-csv "C:\path\to\revised_sample_os.csv" `
  --dbms-csv "C:\path\to\revised_sample_dbms.csv" `
  --output-dir "experiments\exp_013_score_analysis\results"
```

### Output files
Created in `experiments/exp_013_score_analysis/results/`:

- `dataset_audit.csv` — row counts, distinct question counts, missing values, invalid scores/labels, and duplicate candidates.
- `overall_score_summary.csv` — score count, mean, median, standard deviation, quartiles, and range by dataset.
- `label_score_summary.csv` — the same descriptive statistics by dataset and categorical label, plus suggested-band alignment.
- `row_quality_flags.csv` — row-specific issues to review; the script does not correct them automatically.
- `statistical_tests.csv` — exploratory Kruskal–Wallis results, or an explanation if a test could not run.
- `exp_013_report.md` — consolidated report and interpretation cautions.
- `score_distribution_by_label.png` and dataset-specific score histograms, where Matplotlib is available.

### Interpretation safeguards
The project's dataset revision guidelines suggest the following score ranges: Correct 8–10, Partially Correct 4–7.5, Incorrect 1–3.5, Contradictory 0–3, Irrelevant 0–2. These are guidelines, not strict boundaries. Scores outside them are only flagged for review; they are not automatically treated as wrong.

A label is not converted into a numeric rank because Contradictory and Irrelevant represent different failure types. The Kruskal–Wallis test is exploratory, not proof of model performance. Since the dataset has relatively few unique questions, observations within the same question are not fully independent; broader validation and a human-agreement study remain future work.

### Exp 013 completion checklist
- [ ] Run successfully on both original CSVs.
- [ ] Inspect `dataset_audit.csv` and `row_quality_flags.csv`.
- [ ] Review score distributions within each label.
- [ ] Record conclusions and unresolved data-quality questions.
- [ ] Do not start Exp 014 until the score analysis has been reviewed.
