"""Task 4 - Evaluation and tuning.   Run: python src/task4_evaluation_tuning.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

import config as C
from data import get_splits
from features import build_model
from models import regression_sets, train_classifiers

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
models = train_classifiers(X_train, y_train, X_val, y_val)


def report(name, y_true, pred):
    print(f"\n{name}: accuracy {accuracy_score(y_true, pred):.3f} | "
          f"precision {precision_score(y_true, pred, zero_division=0):.3f} | "
          f"recall {recall_score(y_true, pred):.3f} | F1 {f1_score(y_true, pred):.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_true, pred))


def clean(params):
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
            for k, v in params.items()}


def pick_threshold(y_true, proba, min_recall=0.75):
    """Highest-precision threshold that still keeps recall >= min_recall."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    ok = np.where(rec[:-1] >= min_recall)[0]
    i = ok[np.argmax(prec[:-1][ok])]
    return thr[i], prec[i], rec[i]


# ---- 1. Metrics for every classifier ----
print("=== 1. Classification metrics (validation set) ===")
for name, m in models.items():
    report(name, y_val, m.predict(X_val))
majority = max(y_val.mean(), 1 - y_val.mean())
print(f"\nClass balance: {y_val.mean():.1%} positives. A model that always predicts the majority "
      f"class scores accuracy {majority:.3f} but catches 0 defaulters.")
print("Accuracy misleads on imbalanced data because it rewards predicting the common class; "
      "precision, recall, F1 and AUC show how well the rare class is found.")

# ---- 2. ROC curves of 3 models ----
print("\n=== 2. ROC curves ===")
plt.figure(figsize=(6, 5))
for name in ["LogisticRegression", "RandomForest", "XGBoost"]:
    p = models[name].predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, p)
    auc = roc_auc_score(y_val, p)
    print(f"{name}: AUC {auc:.3f}")
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="random guess")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend()
plt.title("ROC curves (validation)"); plt.tight_layout()
plt.savefig(C.OUT_DIR / "task4_roc_curves.png"); plt.close()

# ---- 3. Threshold for >= 75% recall ----
print("\n=== 3. Decision threshold ===")
aucs = {n: roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]) for n, m in models.items()}
best_untuned = max(aucs, key=aucs.get)
thr, prec, rec = pick_threshold(y_val, models[best_untuned].predict_proba(X_val)[:, 1])
print(f"{best_untuned}: threshold {thr:.3f} -> recall {rec:.3f}, precision {prec:.3f}")

# ---- 4. Hyperparameter tuning ----
print("\n=== 4. Tuning (3-fold CV on the training set, scoring = ROC-AUC) ===")
rf_search = GridSearchCV(
    build_model(RandomForestClassifier(random_state=C.RANDOM_STATE)),
    {"model__n_estimators": [100, 300], "model__max_depth": [5, 10, None],
     "model__min_samples_leaf": [1, 5]},
    cv=3, scoring="roc_auc", n_jobs=-1).fit(X_train, y_train)
print("Random Forest (GridSearchCV) best params:", clean(rf_search.best_params_))
print(f"  best CV AUC: {rf_search.best_score_:.4f}")

xgb_search = RandomizedSearchCV(
    build_model(XGBClassifier(eval_metric="logloss", random_state=C.RANDOM_STATE)),
    {"model__n_estimators": randint(100, 500), "model__max_depth": randint(2, 8),
     "model__learning_rate": uniform(0.01, 0.2), "model__subsample": uniform(0.6, 0.4),
     "model__colsample_bytree": uniform(0.6, 0.4)},
    n_iter=15, cv=3, scoring="roc_auc", random_state=C.RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
print("XGBoost (RandomizedSearchCV, n_iter=15) best params:", clean(xgb_search.best_params_))
print(f"  best CV AUC: {xgb_search.best_score_:.4f}")

# ---- 5. Regression metrics vs mean baseline ----
print(f"\n=== 5. Regression: '{C.REG_TARGET}' (validation set) ===")
Xr_tr, yr_tr, Xr_va, yr_va = regression_sets(X_train, X_val)
reg = build_model(Ridge(alpha=10.0), drop=(C.REG_TARGET,)).fit(Xr_tr, yr_tr)
base = DummyRegressor(strategy="mean").fit(Xr_tr, yr_tr)
for name, model in [("Ridge", reg), ("Mean baseline", base)]:
    p = model.predict(Xr_va)
    print(f"{name:14}: MAE {mean_absolute_error(yr_va, p):.2f} | "
          f"RMSE {np.sqrt(mean_squared_error(yr_va, p)):.2f} | R2 {r2_score(yr_va, p):.3f}")

# ---- 6. Pick the single best model and evaluate ONCE on the test set ----
print("\n=== 6. Final evaluation on the TEST set (used only once) ===")
candidates = {"RandomForest (tuned)": rf_search.best_estimator_,
              "XGBoost (tuned)": xgb_search.best_estimator_}
val_auc = {n: roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]) for n, m in candidates.items()}
best_name = max(val_auc, key=val_auc.get)
best = candidates[best_name]
print("Validation AUCs:", {n: round(a, 4) for n, a in val_auc.items()}, "->", best_name)

thr, _, _ = pick_threshold(y_val, best.predict_proba(X_val)[:, 1])   # threshold from VALIDATION
p_test = best.predict_proba(X_test)[:, 1]
report(f"{best_name} @ threshold 0.5", y_test, (p_test >= 0.5).astype(int))
report(f"{best_name} @ threshold {thr:.3f} (>=75% recall)", y_test, (p_test >= thr).astype(int))
f1 = f1_score(y_test, (p_test >= thr).astype(int))
print(f"\nBest model: {best_name} | test ROC-AUC {roc_auc_score(y_test, p_test):.3f} | "
      f"F1 {f1:.3f} (at the 75%-recall threshold)")
