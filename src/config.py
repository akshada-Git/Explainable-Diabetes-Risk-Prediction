"""Central configuration: paths, feature metadata, constants.

Everything that other modules need to agree on lives here so that training,
prediction, the UI and the tests can never drift apart.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_FILENAME = "diabetes_model.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

TARGET = "Outcome"

# Order matters: the model is trained and queried with columns in this order.
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# In the PIMA dataset a 0 in these columns is a physiologically impossible
# value and really means "not measured" (a disguised missing value).
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
RISK_THRESHOLD = 0.5      # probability >= threshold  ->  "High Risk"
MIN_TEST_ROC_AUC = 0.70   # sanity gate used by the tests (TR-12.2)

# Public copies of the PIMA Indians Diabetes dataset, tried in order if
# data/diabetes.csv is not present. (Kaggle needs a login, these do not.)
DATA_URLS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
]

DISCLAIMER = (
    "This tool is for educational/awareness purposes only and is not a "
    "substitute for professional medical advice."
)

# Metadata used to build the input form, validate input and label charts.
# `optional=True` means the user may tick "I don't know" and the value is
# imputed with the training-set median (exactly like the training pipeline).
FEATURE_INFO = {
    "Pregnancies": dict(
        label="Pregnancies", unit="", min=0, max=20, default=1, step=1,
        kind="int", optional=False,
        help="Number of times pregnant. Use 0 if not applicable.",
    ),
    "Glucose": dict(
        label="Glucose", unit="mg/dL", min=40, max=300, default=110, step=1,
        kind="int", optional=False,
        help="Plasma glucose concentration (2-hour oral glucose tolerance test in the dataset).",
    ),
    "BloodPressure": dict(
        label="Blood pressure", unit="mmHg", min=30, max=140, default=72, step=1,
        kind="int", optional=False,
        help="Diastolic blood pressure (the lower number).",
    ),
    "SkinThickness": dict(
        label="Skin thickness", unit="mm", min=5, max=99, default=23, step=1,
        kind="int", optional=True,
        help="Triceps skin-fold thickness. Tick 'I don't know' if you have no measurement.",
    ),
    "Insulin": dict(
        label="Insulin", unit="\u00b5U/mL", min=10, max=900, default=80, step=1,
        kind="int", optional=True,
        help="2-hour serum insulin. Tick 'I don't know' if you have no measurement.",
    ),
    "BMI": dict(
        label="BMI", unit="kg/m\u00b2", min=10.0, max=70.0, default=27.0, step=0.1,
        kind="float", optional=False,
        help="Body mass index = weight (kg) / height (m) squared.",
    ),
    "DiabetesPedigreeFunction": dict(
        label="Diabetes pedigree function", unit="", min=0.05, max=2.5, default=0.4, step=0.01,
        kind="float", optional=False,
        help="A score of diabetes family history. Higher = stronger family history. Typical range 0.1 - 1.0.",
    ),
    "Age": dict(
        label="Age", unit="years", min=18, max=100, default=30, step=1,
        kind="int", optional=False,
        help="Age in years. The training data covers ages 21-81.",
    ),
}
