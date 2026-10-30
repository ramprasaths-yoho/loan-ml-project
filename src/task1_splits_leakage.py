"""Task 1 - Concepts, splits and leakage.   Run: python src/task1_splits_leakage.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config as C
from data import get_splits, load_data

X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
print(f"Train {len(X_train)} | Validation {len(X_val)} | Test {len(X_test)} (test set is put aside)")

# Task 1 uses numeric columns only (text columns are handled properly in Task 2)
Xtr, Xva = X_train[C.NUMERIC], X_val[C.NUMERIC]

# ---- 1. Overfitting: train vs validation accuracy for different tree depths ----
print("\n--- Overfitting demo ---")
depths = [1, 3, 5, 10, None]
train_acc, val_acc = [], []
for d in depths:
    tree = make_pipeline(SimpleImputer(strategy="median"),
                         DecisionTreeClassifier(max_depth=d, random_state=C.RANDOM_STATE))
    tree.fit(Xtr, y_train)
    train_acc.append(tree.score(Xtr, y_train))
    val_acc.append(tree.score(Xva, y_val))
    print(f"Depth {str(d):>4}: train {train_acc[-1]:.2f} val {val_acc[-1]:.2f}")

labels = [str(d) for d in depths]
plt.figure(figsize=(6, 4))
plt.plot(labels, train_acc, marker="o", label="train")
plt.plot(labels, val_acc, marker="o", label="validation")
plt.xlabel("max_depth"); plt.ylabel("accuracy"); plt.legend()
plt.title("Overfitting: train vs validation accuracy")
plt.tight_layout(); plt.savefig(C.OUT_DIR / "task1_overfitting.png"); plt.close()

# ---- 2. 5-fold cross-validation ----
print("\n--- 5-fold cross-validation (Logistic Regression) ---")
lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                   LogisticRegression(max_iter=1000))
scores = cross_val_score(lr, Xtr, y_train, cv=5)
print(f"CV: {scores.mean():.2f} +/- {scores.std():.2f}")

# ---- 3. Data leakage: scaling before vs after the split ----
print("\n--- Data leakage demo ---")
X_all = load_data()[0][C.NUMERIC]           # ALL rows, including validation and test
prep_wrong = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X_all)
clf = LogisticRegression(max_iter=1000).fit(prep_wrong.transform(Xtr), y_train)
print(f"WRONG  (scaler fitted on all data) val accuracy: {clf.score(prep_wrong.transform(Xva), y_val):.4f}")

prep_right = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(Xtr)
clf = LogisticRegression(max_iter=1000).fit(prep_right.transform(Xtr), y_train)
print(f"RIGHT  (scaler fitted on train only) val accuracy: {clf.score(prep_right.transform(Xva), y_val):.4f}")
print(f"Scaler mean of '{C.NUMERIC[0]}': wrong = {prep_wrong[1].mean_[0]:.3f} vs right = {prep_right[1].mean_[0]:.3f} "
      "(the difference is information that leaked from val/test)")
print("""
Explanation:
1. Scaling before the split lets the mean/std of validation and test rows leak into training.
2. The model then 'sees' information about data it should never have seen.
3. The score looks slightly better than reality (here the gap may be tiny, on small or
   skewed data it can be large), so we would overestimate how well the model works.
4. Correct way: split first, fit scaler/imputer on train only, then only transform val/test.""")

# ---- 4. Supervised vs unsupervised + 7-step workflow ----
print("""
--- Supervised or unsupervised? ---
Spam detection           -> supervised (classification, labels spam/not spam)
Customer segmentation    -> unsupervised (clustering, no labels)
House price prediction   -> supervised (regression, label = price)
Anomaly detection        -> usually unsupervised (finds unusual points without labels)

--- ML workflow in 7 steps ---
1. Define the problem and the success metric
2. Collect and understand the data (EDA)
3. Split into train / validation / test
4. Clean and preprocess (missing values, encoding, scaling) - fitted on train only
5. Engineer features and train models
6. Evaluate on validation, tune hyperparameters
7. Final test evaluation, save the model and deploy/monitor""")
