# Experiment 013 — Human Score Analysis and Categorical Alignment

## Purpose
Describe the human-score distributions in the existing datasets, inspect score variation within each categorical label, and flag data-quality items for review before implementing a numerical scorer.

## Scope and safeguards
- This experiment is descriptive; it does not train a scoring model.
- Source CSV files are not modified.
- Scores outside suggested label bands are flagged for review only, not treated as automatic errors.
- The suggested bands are documented project guidelines, not rigid rules: Correct 8–10; Partially Correct 4–7.5; Incorrect 1–3.5; Contradictory 0–3; Irrelevant 0–2.
- Categorical labels are not converted into a numeric ordinal scale because Contradictory and Irrelevant are distinct failure types, not necessarily ordered relative to each other.

## Dataset audit
| dataset   | source_path                                             |   rows |   unique_questions |   missing_question_text |   missing_student_answer |   missing_model_answer |   missing_label |   unknown_label |   missing_human_score |   non_numeric_human_score |   out_of_range_human_score |   possible_duplicate_question_answer_rows |   outside_suggested_label_score_band_rows_review_only | annotator_id_column_present   |   unique_annotator_ids | unrecognized_label_values   |
|:----------|:--------------------------------------------------------|-------:|-------------------:|------------------------:|-------------------------:|-----------------------:|----------------:|----------------:|----------------------:|--------------------------:|---------------------------:|------------------------------------------:|------------------------------------------------------:|:------------------------------|-----------------------:|:----------------------------|
| OS        | C:\Users\Ayuj Shingte\Downloads\revised_sample_os.csv   |     84 |                 14 |                       0 |                        0 |                      0 |               0 |               0 |                     0 |                         0 |                          0 |                                         0 |                                                     8 | True                          |                      2 |                             |
| DBMS      | C:\Users\Ayuj Shingte\Downloads\revised_sample_dbms.csv |     92 |                 15 |                       0 |                        0 |                      0 |               0 |               0 |                     0 |                         0 |                          0 |                                         0 |                                                    10 | True                          |                      2 |                             |

## Overall score summaries
| dataset   |   rows |   unique_questions |   numeric_score_count |   missing_or_invalid_score_count |   mean |   median |   std_dev |   minimum |    q1 |    q3 |   maximum |
|:----------|-------:|-------------------:|----------------------:|---------------------------------:|-------:|---------:|----------:|----------:|------:|------:|----------:|
| OS        |     84 |                 14 |                    84 |                                0 |  3.887 |    2.500 |     3.795 |     0.000 | 0.500 | 8.000 |     9.500 |
| DBMS      |     92 |                 15 |                    92 |                                0 |  4.136 |    3.250 |     3.714 |     0.000 | 0.500 | 8.000 |    10.000 |

## Score distribution within categorical labels
| dataset   | label             |   answer_count |   valid_numeric_score_count |   missing_or_invalid_score_count |   mean |   median |   std_dev |   minimum |    q1 |    q3 |   maximum |   suggested_band_low |   suggested_band_high |   count_within_suggested_band |   percent_within_suggested_band | band_note                                                             |
|:----------|:------------------|---------------:|----------------------------:|---------------------------------:|-------:|---------:|----------:|----------:|------:|------:|----------:|---------------------:|----------------------:|------------------------------:|--------------------------------:|:----------------------------------------------------------------------|
| OS        | Correct           |             28 |                          28 |                                0 |  8.750 |    8.750 |     0.764 |     8.000 | 8.000 | 9.500 |     9.500 |                8.000 |                10.000 |                            28 |                         100.000 | Suggested range; flag for review only, never an automatic correction. |
| OS        | Partially Correct |             14 |                          14 |                                0 |  4.464 |    4.500 |     0.664 |     3.500 | 4.000 | 5.000 |     5.500 |                4.000 |                 7.500 |                            12 |                          85.714 | Suggested range; flag for review only, never an automatic correction. |
| OS        | Incorrect         |             14 |                          14 |                                0 |  0.786 |    1.000 |     0.671 |     0.000 | 0.000 | 1.500 |     1.500 |                1.000 |                 3.500 |                             8 |                          57.143 | Suggested range; flag for review only, never an automatic correction. |
| OS        | Irrelevant        |             14 |                          14 |                                0 |  0.000 |    0.000 |     0.000 |     0.000 | 0.000 | 0.000 |     0.000 |                0.000 |                 2.000 |                            14 |                         100.000 | Suggested range; flag for review only, never an automatic correction. |
| OS        | Contradictory     |             14 |                          14 |                                0 |  0.571 |    0.500 |     0.267 |     0.000 | 0.500 | 0.500 |     1.000 |                0.000 |                 3.000 |                            14 |                         100.000 | Suggested range; flag for review only, never an automatic correction. |
| DBMS      | Correct           |             32 |                          32 |                                0 |  8.797 |    9.000 |     0.792 |     7.500 | 8.000 | 9.500 |    10.000 |                8.000 |                10.000 |                            31 |                          96.875 | Suggested range; flag for review only, never an automatic correction. |
| DBMS      | Partially Correct |             15 |                          15 |                                0 |  4.300 |    4.500 |     0.676 |     3.000 | 4.000 | 4.750 |     5.500 |                4.000 |                 7.500 |                            12 |                          80.000 | Suggested range; flag for review only, never an automatic correction. |
| DBMS      | Incorrect         |             20 |                          20 |                                0 |  1.325 |    1.500 |     0.674 |     0.500 | 0.500 | 2.000 |     2.500 |                1.000 |                 3.500 |                            14 |                          70.000 | Suggested range; flag for review only, never an automatic correction. |
| DBMS      | Irrelevant        |             12 |                          12 |                                0 |  0.000 |    0.000 |     0.000 |     0.000 | 0.000 | 0.000 |     0.000 |                0.000 |                 2.000 |                            12 |                         100.000 | Suggested range; flag for review only, never an automatic correction. |
| DBMS      | Contradictory     |             13 |                          13 |                                0 |  0.615 |    0.500 |     0.300 |     0.000 | 0.500 | 1.000 |     1.000 |                0.000 |                 3.000 |                            13 |                         100.000 | Suggested range; flag for review only, never an automatic correction. |

## Exploratory Kruskal–Wallis tests
| dataset   | test           | status    | groups_included                                                  |   statistic_H |    p_value | interpretation_note                                                                                                                                                   |
|:----------|:---------------|:----------|:-----------------------------------------------------------------|--------------:|-----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OS        | Kruskal-Wallis | completed | Correct; Partially Correct; Incorrect; Irrelevant; Contradictory |        75.386 | 1.6513e-15 | Exploratory omnibus test for differences in score distributions across labels; it does not establish label correctness or prove that a scoring model will generalize. |
| DBMS      | Kruskal-Wallis | completed | Correct; Partially Correct; Incorrect; Irrelevant; Contradictory |        84.227 | 2.2123e-17 | Exploratory omnibus test for differences in score distributions across labels; it does not establish label correctness or prove that a scoring model will generalize. |

## Rows requiring review
Rows with at least one flag: 18
See `row_quality_flags.csv` for row-level flags. Review flags do not automatically imply that a row is wrong.

## Interpretation guidance
1. Check score variation within each label. Substantial variation suggests that the label alone cannot determine a 0–10 score.
2. Check the suggested-band alignment as an annotation QA aid, not a rigid pass/fail rule.
3. Treat statistical tests as exploratory: there are relatively few unique questions, and answers to the same question are not independent observations.
4. Before training a regression scorer, decide how to evaluate predictions (e.g. MAE and RMSE) and preserve question-level separation to reduce leakage.
5. A score distribution analysis does not prove inter-rater reliability. That requires independent annotations for the same answers.

## Decision for the next experiment
Use these findings to design Exp 014 (an interpretable deterministic scoring baseline) and Exp 015 (a learned regression baseline). Do not begin scoring-model training until the data-quality flags and score rubric have been reviewed.
