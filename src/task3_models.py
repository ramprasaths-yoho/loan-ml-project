"""Task 3 - Models: supervised and unsupervised.   Run: python src/task3_models.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import config as C
from features import build_model, feature_names
from models import regression_sets, train_classifiers
from data import get_splits

pd.set_option("display.width", 150)
X_train, X_val, X_test, y_train, y_val, y_test = get_splits()

# ---- 1. Four classifiers + results table ----
print("--- Training classifiers ---")
models = train_classifiers(X_train, y_train, X_val, y_val)
rows = []
for name, m in models.items():
    rows.append({"Model": name,
                 "Validation accuracy": round(accuracy_score(y_val, m.predict(X_val)), 4),
                 "Validation ROC-AUC": round(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]), 4)})
results = pd.DataFrame(rows)
results.to_csv(C.OUT_DIR / "task3_results_table.csv", index=False)
print("\n", results.to_string(index=False))

# ---- 2. predict_proba vs predict (logistic regression, 5 rows) ----
lr = models["LogisticRegression"]
sample = X_val.head(5)
proba = lr.predict_proba(sample)
print("\n--- predict_proba vs predict (5 rows) ---")
print(pd.DataFrame({"P(class 0)": proba[:, 0].round(3), "P(class 1)": proba[:, 1].round(3),
                    "predict": lr.predict(sample)}))
print("predict() simply returns class 1 when P(class 1) > 0.5, otherwise class 0.")

# ---- 3. Top-5 important features ----
print("\n--- Top 5 features ---")
for name in ["RandomForest", "XGBoost"]:
    m = models[name]
    imp = pd.Series(m.named_steps["model"].feature_importances_, index=feature_names(m))
    print(f"\n{name}:\n{imp.sort_values(ascending=False).head(5).round(4)}")

# ---- 4. Regression: predict REG_TARGET with Linear / Ridge / Lasso ----
print(f"\n--- Regression: predicting '{C.REG_TARGET}' ---")
Xr_tr, yr_tr, Xr_va, yr_va = regression_sets(X_train, X_val)
reg_models = {"LinearRegression": LinearRegression(),
              "Ridge": Ridge(alpha=10.0),
              # Lasso's alpha is relative to the size of the target, so we scale it by its std
              "Lasso": Lasso(alpha=0.01 * yr_tr.std(), max_iter=10000)}
coefs, fitted_reg = {}, {}
for name, est in reg_models.items():
    # drop=(target,) so the target and features derived from it are not used as inputs
    pipe = build_model(est, drop=(C.REG_TARGET,)).fit(Xr_tr, yr_tr)
    fitted_reg[name] = pipe
    coefs[name] = pipe.named_steps["model"].coef_
    print(f"{name:17} validation R2: {pipe.score(Xr_va, yr_va):.4f}")
coef_table = pd.DataFrame(coefs, index=feature_names(fitted_reg["Lasso"])).round(2)
print("\nCoefficients:\n", coef_table)
print(f"\nLasso coefficients that are exactly zero: {(coefs['Lasso'] == 0).sum()} of {len(coefs['Lasso'])}")
print("Ridge shrinks coefficients towards zero; Lasso can set some exactly to zero (feature selection).")

# ---- 5. K-Means segmentation ----
print("\n--- K-Means (k = 2..8) ---")
raw = X_train[C.KMEANS_COLS]
raw = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(raw), columns=raw.columns)
Xk = StandardScaler().fit_transform(raw)
ks, inertias, sils = list(range(2, 9)), [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(Xk)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xk, km.labels_, sample_size=min(10000, len(Xk)),
                                 random_state=C.RANDOM_STATE))
    print(f"k={k}: inertia {inertias[-1]:.1f}  silhouette {sils[-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertias, marker="o"); ax[0].set_title("Elbow curve")
ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia")
ax[1].plot(ks, sils, marker="o", color="green"); ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k"); ax[1].set_ylabel("silhouette")
plt.tight_layout(); plt.savefig(C.OUT_DIR / "task3_elbow_silhouette.png"); plt.close()

best_k = ks[int(np.argmax(sils))]
labels = KMeans(n_clusters=best_k, n_init=10, random_state=C.RANDOM_STATE).fit(Xk).labels_
seg = raw.assign(segment=labels, target_rate=y_train.values).groupby("segment").agg(
    **{c: (c, "mean") for c in C.KMEANS_COLS}, size=("segment", "size"),
    target_rate=("target_rate", "mean")).round(2)
print(f"\nBest k by silhouette = {best_k}. Segment profile (means):\n", seg)
print("Describe each segment in your README, e.g. 'young, low income, low score, high default rate'.")
