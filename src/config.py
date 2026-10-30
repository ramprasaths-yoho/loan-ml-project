from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# --- target ---
TARGET = "Churn"
POSITIVE_LABEL = "Yes"

# --- columns ---
DROP_COLS = ["customerID"]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
               "PaymentMethod"]
ORDINAL = {"Contract": ["Month-to-month", "One year", "Two year"]}

# --- feature engineering (this dataset has no date column) ---
DATE_COL = None
INCOME_COL = "MonthlyCharges"   # numerator of the ratio feature
LOAN_COL = "tenure"             # denominator of the ratio feature
SKEWED_COL = "MonthlyCharges"   # gets the log transform

# --- regression: predict this numeric column ---
REG_TARGET = "TotalCharges"

# --- K-Means ---
KMEANS_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]