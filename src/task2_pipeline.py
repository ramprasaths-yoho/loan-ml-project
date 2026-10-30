"""Task 2 - Preprocessing, features and Pipeline.   Run: python src/task2_pipeline.py"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import config as C
from data import get_splits
from features import add_features, build_model, engineered_names

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

print("Missing values per column (train):")
print(X_train.isna().sum()[X_train.isna().sum() > 0])

print("\nNew features (first 3 rows):")
print(add_features(X_train.head(3))[engineered_names()])

# ---- Imputer comparison: median (SimpleImputer) vs KNNImputer on numeric columns ----
print("\n--- Pipeline CV accuracy (5-fold, imputation happens inside each fold) ---")
for name in ["median", "knn"]:
    pipe = build_model(LogisticRegression(max_iter=1000), numeric_imputer=name)
    acc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Numeric imputer = {name:6}: {acc.mean():.2f} +/- {acc.std():.2f}")

# ---- Fit, save with joblib, load, predict on 3 raw rows ----
pipe = build_model(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
joblib.dump(pipe, C.MODEL_DIR / "pipeline.joblib")
print(f"\nSaved pipeline to {C.MODEL_DIR / 'pipeline.joblib'}")

loaded = joblib.load(C.MODEL_DIR / "pipeline.joblib")
# 3 raw rows. Here we take them from the validation set; you can also type your own:
# new_rows = pd.DataFrame([{"age": 35, "income": 50000, ...}])  (use the raw column names)
new_rows = X_val.head(3)
print("Raw input rows:")
print(new_rows)
print("Predictions for 3 new rows:", loaded.predict(new_rows))
