# =========================================
# Project: Predict Max Monthly EMI
# =========================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
import joblib
import mlflow
import mlflow.xgboost
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# -----------------------------
# 1️⃣ Read Data
# -----------------------------
df = pd.read_csv("Processed_data.csv")

# -----------------------------
# 2️⃣ Define features and target
# -----------------------------
X = df.drop(columns=['max_monthly_emi', 'emi_eligibility'])
y = df['max_monthly_emi']  # target

# -----------------------------
# 3️⃣ Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4️⃣ Identify categorical and numeric features
# -----------------------------
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# -----------------------------
# 5️⃣ Preprocessing: Categorical encoding only after split
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # keep numeric columns as is
)

# -----------------------------
# 6️⃣ Evaluation function
# -----------------------------
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)  # returns MSE
    rmse = np.sqrt(mse)                      # manually take sqrt for RMSE
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

# -----------------------------
# 7️⃣ Baseline pipelines
# -----------------------------
# RandomForest
pipeline_rf = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# XGBoost
pipeline_xgb = Pipeline([
    ('preprocess', preprocessor),
    ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# LightGBM
pipeline_lgb = Pipeline([
    ('preprocess', preprocessor),
    ('model', lgb.LGBMRegressor(random_state=42))
])

# Fit baseline models
baseline_models = {
    "RandomForest": pipeline_rf,
    "XGBoost": pipeline_xgb,
    "LightGBM": pipeline_lgb
}

for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print(f"✅ {name} metrics: {metrics}")

# -----------------------------
# 8️⃣ Optional: Hyperparameter tuning with Optuna for RF
# -----------------------------
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 150)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    pipeline = Pipeline([('preprocess', preprocessor), ('model', rf)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

study = optuna.create_study(direction='minimize', sampler=TPESampler(), pruner=HyperbandPruner())
study.optimize(objective, n_trials=20, timeout=600)  # fast tuning

print("✅ Best RF hyperparameters:", study.best_params)

# Replace baseline RF with tuned one
pipeline_rf = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(**study.best_params, random_state=42))
])
pipeline_rf.fit(X_train, y_train)
baseline_models["RandomForest"] = pipeline_rf

# -----------------------------
# 9️⃣ Select best model based on RMSE
# -----------------------------
best_model_name = None
best_model_pipeline = None
best_rmse = float('inf')

for name, model in baseline_models.items():
    metrics = evaluate_model(model, X_test, y_test)
    if metrics['RMSE'] < best_rmse:
        best_rmse = metrics['RMSE']
        best_model_name = name
        best_model_pipeline = model

print(f"✅ Best model: {best_model_name} with RMSE={best_rmse:.2f}")

# -----------------------------
# 🔟 Log best model to MLflow
# -----------------------------
mlflow.set_experiment("MaxMonthlyEMI_Prediction")
with mlflow.start_run(run_name=f"Best_{best_model_name}"):
    metrics = evaluate_model(best_model_pipeline, X_test, y_test)
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
    mlflow.sklearn.log_model(best_model_pipeline, f"best_{best_model_name}_model")
    print(f"✅ Metrics and model logged to MLflow for {best_model_name}")

# -----------------------------
# 1️⃣1️⃣ Save the best model with joblib
# -----------------------------
save_path = r"C:\Users\Admin\OneDrive\Career and work\Labmentix internship\Project 5\best_model.pkl"
joblib.dump(best_model_pipeline, save_path)
print(f"✅ Best model saved to {save_path}")