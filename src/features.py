"""Feature engineering + preprocessing (Task 2). Everything is inside a Pipeline,
so during cross-validation it is fitted on the training folds only (no leakage)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

import config as C


def add_features(df, use_ratio=True):
    """Creates the new features from raw columns."""
    df = df.copy()
    if C.DATE_COL:                                   # 1) date decomposition
        d = pd.to_datetime(df[C.DATE_COL], errors="coerce")
        df["app_year"] = d.dt.year
        df["app_month"] = d.dt.month
        df["app_dayofweek"] = d.dt.dayofweek
        df = df.drop(columns=[C.DATE_COL])
    if use_ratio:                                    # 2) interaction feature
        df["income_to_loan"] = df[C.INCOME_COL] / df[C.LOAN_COL].replace(0, np.nan)
    df["log_" + C.SKEWED_COL] = np.log1p(df[C.SKEWED_COL].clip(lower=0))  # 3) log
    if not C.DATE_COL:                               # flag feature (no date column here)
        df["is_new_customer"] = (df[C.LOAN_COL] <= 6).astype(int)
    return df


def engineered_names(use_ratio=True):
    names = ["app_year", "app_month", "app_dayofweek"] if C.DATE_COL else ["is_new_customer"]
    if use_ratio:
        names.append("income_to_loan")
    names.append("log_" + C.SKEWED_COL)
    return names


def build_preprocessor(numeric_imputer="median", drop=()):
    """numeric_imputer: 'median' (SimpleImputer) or 'knn' (KNNImputer).
    drop: raw columns to leave out (used when one of them is the regression target)."""
    use_ratio = C.LOAN_COL not in drop
    num_cols = [c for c in C.NUMERIC if c not in drop] + engineered_names(use_ratio)
    cat_cols = [c for c in C.CATEGORICAL if c not in drop]
    ord_cols = [c for c in C.ORDINAL if c not in drop]

    num_imp = (SimpleImputer(strategy="median") if numeric_imputer == "median"
               else KNNImputer(n_neighbors=5))
    numeric = Pipeline([("imp", num_imp), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ordinal = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(categories=[C.ORDINAL[c] for c in ord_cols],
                               handle_unknown="use_encoded_value", unknown_value=-1))])

    ct = ColumnTransformer([("num", numeric, num_cols),
                            ("cat", categorical, cat_cols),
                            ("ord", ordinal, ord_cols)])
    return Pipeline([("features", FunctionTransformer(add_features,
                                                      kw_args={"use_ratio": use_ratio})),
                     ("ct", ct)])


def build_model(estimator, **prep_kwargs):
    """Full pipeline: raw rows in -> prediction out."""
    return Pipeline([("pre", build_preprocessor(**prep_kwargs)), ("model", estimator)])


def feature_names(fitted_pipe):
    return fitted_pipe.named_steps["pre"].named_steps["ct"].get_feature_names_out()