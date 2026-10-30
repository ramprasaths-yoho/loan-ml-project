# Loan Default Prediction - ML Workflow

Predict loan default (classification), estimate the requested loan amount (regression)
and segment customers (K-Means). Built for the one-day ML workflow assessment.

## How to run
```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
# 1. put the trainer's CSV in data/ and edit src/config.py (file name, column names)
python src/task1_splits_leakage.py
python src/task2_pipeline.py
python src/task3_models.py
python src/task4_evaluation_tuning.py
```
Plots and tables are saved in `outputs/`, the saved pipeline in `models/`.

## Folder structure
- `data/` dataset (not committed if large)  - `src/` code  - `outputs/` plots/tables  - `models/` joblib pipeline

## Decisions (fill in with your own words)
- Split: 60/20/20 stratified, random_state=42; test set used once at the very end.
- Missing values: median (numeric), most frequent (categorical); KNNImputer compared in Task 2.
- Encoding: OneHotEncoder for unordered columns, OrdinalEncoder for `education`.
- New features: date parts, income_to_loan ratio, log of income.
- Everything preprocessing-related lives in a Pipeline/ColumnTransformer, so no leakage in CV.

## Results (paste from your run)
| Model | Validation accuracy | Validation ROC-AUC |
|---|---|---|
| ... | ... | ... |

Final test result: ...

## Customer segments (K-Means)
Describe each segment here.
