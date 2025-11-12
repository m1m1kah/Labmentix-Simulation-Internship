# ===============================================
# Classification Pipeline for EMI Eligibility
# ===============================================
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("Processed_data.csv")

TARGET = "emi_eligibility"  # 'Not_Eligible', 'Eligible', 'High_Risk'

# Exclude both targets from features
X = df.drop(columns=[TARGET, 'max_monthly_emi'])

# Target
y = df[TARGET]

# Encode target for classifiers like XGBoost

le = LabelEncoder()
y = le.fit_transform(y)
# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Encoding (after split)
# -----------------------------
cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns
num_cols = X_train.select_dtypes(exclude=["object", "string", "category"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# -----------------------------
# 4. Define Evaluation Function
# -----------------------------
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="weighted", zero_division=0)
    recall = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

# -----------------------------
# 5. Define Baseline Models
# -----------------------------
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss", use_label_encoder=False),
    "LightGBM": LGBMClassifier(random_state=42)
}

results = {}
mlflow.set_experiment("EMI_Eligibility_Classification")

# -----------------------------
# 6. Train & Log to MLflow
# -----------------------------
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    results[name] = metrics
    print(f"✅ {name} metrics:", metrics)

    with mlflow.start_run(run_name=f"{name}_Classification"):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.sklearn.log_model(pipeline, f"{name}_model")

# -----------------------------
# 7. Select Best Model
# -----------------------------
best_model_name = max(results, key=lambda x: results[x]["F1"])
best_metrics = results[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print("📊 Best Metrics:", best_metrics)

best_model = models[best_model_name]
best_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", best_model)
])
best_pipeline.fit(X_train, y_train)

# -----------------------------
# 8. Save Best Model
# -----------------------------
joblib.dump(best_pipeline, r"C:\Users\Admin\OneDrive\Career and work\Labmentix internship\Project 5\best_classification_model.pkl")
print("✅ Model saved successfully!")

# -----------------------------
# 9. Optional: Full Report
# -----------------------------
y_pred = best_pipeline.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
