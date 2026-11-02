@'

\# Customer Churn Prediction Using Machine Learning



\## 1. Project Overview



This project predicts whether a telecommunications customer is likely to leave the company using machine learning.



\## 2. Dataset



\- Dataset: IBM Telco Customer Churn

\- File: data/Telco-Customer-Churn.csv

\- Target column: Churn

\- Positive class: Yes



\## 3. Tasks Completed



\### Task 1: Data Splitting and Leakage Prevention

\- Data inspection and splitting.

\- Training, validation and testing.

\- Overfitting and leakage demonstration.



\### Task 2: Data Preprocessing and Feature Engineering

\- Missing-value handling using median and most frequent strategies.

\- Comparison of median and KNN imputation.

\- One-Hot Encoding and Ordinal Encoding.

\- Numerical feature scaling.

\- Created three engineered features:

&#x20; - Monthly charges per tenure month.

&#x20; - Log transformation of monthly charges.

&#x20; - New customer indicator (tenure <= 6 months).

\- Used Pipeline and ColumnTransformer to prevent data leakage.



\### Task 3: Model Training

\- Trained and compared classification models.

\- Evaluated model performance.

\- Performed K-Means customer segmentation.



\### Task 4: Model Evaluation and Hyperparameter Tuning

\- Model evaluation.

\- Hyperparameter tuning.

\- ROC curve comparison.

\- Performance visualizations.



\## 4. Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Seaborn

\- Joblib

\- Git and GitHub



\## 5. Project Structure



\- data/: Dataset

\- src/: Python source files

\- models/: Saved machine learning pipeline

\- outputs/: Evaluation results and visualizations

\- README.md: Project documentation

\- requirements.txt: Python dependencies



\## 6. Installation



```bash

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

