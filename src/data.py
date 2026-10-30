"""Loading the CSV and making the 60/20/20 stratified split."""
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C


def load_data():
    df = pd.read_csv(C.DATA_PATH)
    df = df.drop(columns=[c for c in C.DROP_COLS if c in df.columns])
    for c in C.NUMERIC:  # text like " " becomes NaN instead of crashing
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if C.DATE_COL:
        df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL], errors="coerce")
    y = (df[C.TARGET] == C.POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[C.TARGET])
    return X, y


def get_splits():
    """60% train / 20% validation / 20% test, stratified, random_state=42."""
    X, y = load_data()
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=C.RANDOM_STATE)
    # 25% of the remaining 80% = 20% of the whole data
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=C.RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test
