# Explainable Diabetes Risk Prediction

B.Sc. Data Science, TY mini project. A machine-learning app that predicts diabetes risk from basic health
measurements **and explains why** using SHAP (Explainable AI). Built with Python, scikit-learn, SHAP and Streamlit.

> **Educational/awareness use only. This is not medical advice and not a diagnosis.**

## Features

| Requirement (FRD) | Where |
|---|---|
| FR-1 Preprocessing (zeros treated as missing, median imputation, scaling, stratified 80/20 split) | `src/preprocessing.py`, `src/training.py` |
| FR-2 Train and compare Logistic Regression, Decision Tree, Random Forest (+ XGBoost if installed), Accuracy/Precision/Recall/F1/ROC-AUC | `src/training.py` -> `reports/` |
| FR-3 Input form with validation and inline errors | `app.py`, `src/predict.py` |
| FR-4 Risk class + probability | `app.py`, `src/predict.py` |
| FR-5 Per-prediction SHAP chart, global importance, plain-language summary | `src/explain.py`, `app.py` |
| FR-6 One-screen result, permanent disclaimer, PDF download | `app.py`, `src/report.py` |
| FR-7 No user data stored | (nothing is persisted) |
| EH-1..EH-4 Error handling (missing input, out of range, model load failure, SHAP failure) | `app.py` |

## Project structure

```
app.py                Streamlit app
train.py              Trains models, writes models/ and reports/
requirements.txt
src/
  config.py           paths, feature metadata, constants
  data.py             load / validate / download the dataset
  preprocessing.py    zero->NaN, imputer + scaler pipeline
  training.py         model comparison, tuning, evaluation, global SHAP
  predict.py          input validation, single prediction
  explain.py          SHAP helpers, contribution table, chart, summary
  report.py           PDF report
tests/                unit tests
data/diabetes.csv     dataset (created/downloaded, see below)
models/               saved model (created by train.py)
reports/              metrics + plots (created by train.py)
docs/                 PRD, FRD, TRD, Directory
```

## 1. Run it on your computer

Requires Python 3.10 or 3.11 (3.12 also works).

```bash
# 1) create a virtual environment
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) install
pip install -r requirements.txt

# 3) get the dataset (either way works)
#    a) do nothing: train.py downloads a public copy into data/diabetes.csv, or
#    b) download "diabetes.csv" from
#       https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
#       and put it at data/diabetes.csv

# 4) train (1-3 minutes). Creates models/diabetes_model.joblib and reports/*
python train.py

# 5) start the app
streamlit run app.py
```

Run the tests: `python -m unittest discover -s tests -t . -v` (or `pytest`).

## 2. Put it on GitHub

Create an empty repository on github.com first (no README), then in the project folder:

```bash
git init
git add .
git commit -m "Explainable diabetes risk prediction"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

**Before pushing, make sure these exist and are committed:** `data/diabetes.csv`, `models/diabetes_model.joblib`
and the `reports/` files (all created by `python train.py`). They are small, and the deployed app needs them.

## 3. Deploy on Streamlit Community Cloud (free)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Create a new app, choose your repository, branch `main`, and main file path `app.py`.
3. Open *Advanced settings* and pick Python **3.11** (safest for SHAP and scikit-learn).
4. Deploy. The first start takes a few minutes while packages install.

If the saved model cannot be loaded (for example it was trained with a different scikit-learn version), the app
trains a fresh model automatically on start-up. This needs `data/diabetes.csv` in the repo or internet access, and takes about a minute.

## How it works

1. **Cleaning:** in the PIMA data, a 0 in Glucose, BloodPressure, SkinThickness, Insulin or BMI is impossible, so it is treated as missing and filled with the training median.
2. **Modelling:** each candidate model sits in a pipeline (imputer, scaler, model) and is tuned with 5-fold cross-validation on the training split (class weights balanced). The model with the best cross-validated ROC-AUC is chosen; the 20% test set is used **once** for the reported metrics, so there is no test-set leakage.
3. **Explaining:** SHAP `TreeExplainer` (tree models) or `LinearExplainer` (Logistic Regression) gives each feature's push on the estimate. Tree-model values are in probability points; Logistic Regression values are in log-odds (the app labels which).
4. **Sanity check:** `reports/metrics.json` records whether Glucose is among the top SHAP features (it should be, clinically).

## Known limitations (mention these in your viva)

* Small dataset (768 rows, adult women of Pima heritage): results do not generalise to everyone.
* SHAP explains the model's behaviour, not medical causation.
* The default threshold is 0.5; for real screening you would tune it towards higher recall.
* Optional features from the PRD not included: LLM-written explanations, SQLite logging.

## Data source and licence

PIMA Indians Diabetes Database (UCI / Kaggle, originally from the National Institute of Diabetes and Digestive and Kidney Diseases). Code is MIT-licensed (replace `<YOUR NAME>` in `LICENSE`).
