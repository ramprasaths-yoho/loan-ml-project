"""The four classifiers (shared by Task 3 and Task 4) + regression data helper."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config as C
from features import build_model, build_preprocessor


def train_classifiers(X_train, y_train, X_val, y_val):
    rs = C.RANDOM_STATE
    models = {}
    models["LogisticRegression"] = build_model(
        LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    models["DecisionTree"] = build_model(
        DecisionTreeClassifier(max_depth=5, random_state=rs)).fit(X_train, y_train)

    rf = build_model(RandomForestClassifier(
        n_estimators=300, oob_score=True, n_jobs=-1, random_state=rs)).fit(X_train, y_train)
    print(f"RandomForest OOB score: {rf.named_steps['model'].oob_score_:.4f}")
    models["RandomForest"] = rf

    # XGBoost early stopping needs the *transformed* validation set, so we fit the
    # preprocessing on train first, then stop boosting when validation AUC stops improving.
    pre = build_preprocessor().fit(X_train)
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4,
                        eval_metric="auc", early_stopping_rounds=30, random_state=rs)
    xgb.fit(pre.transform(X_train), y_train,
            eval_set=[(pre.transform(X_val), y_val)], verbose=False)
    print(f"XGBoost stopped at iteration {xgb.best_iteration}")
    models["XGBoost"] = Pipeline([("pre", pre), ("model", xgb)])
    return models


def regression_sets(X_train, X_val):
    """Rows where the regression target is known."""
    tr, va = X_train[C.REG_TARGET].notna(), X_val[C.REG_TARGET].notna()
    return (X_train[tr], X_train.loc[tr, C.REG_TARGET],
            X_val[va], X_val.loc[va, C.REG_TARGET])
