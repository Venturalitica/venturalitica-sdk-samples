import pandas as pd
import numpy as np
from pathlib import Path
import os
import argparse
import uuid

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
import joblib

# MLOps Imports (MLflow Specialized)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Green AI
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("[Warning] codecarbon not installed. Install with: pip install codecarbon")

# Governance SDK
import venturalitica

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "datasets/loan"
DATA_FILE = DATA_DIR / "german_credit.csv"
RISK_POLICY_PATH = Path(__file__).parent.parent.parent / "policies/loan/risks.oscal.yaml"
GOVERNANCE_POLICY_PATH = Path(__file__).parent.parent.parent / "policies/loan/governance-baseline.oscal.yaml"
POLICIES = [RISK_POLICY_PATH, GOVERNANCE_POLICY_PATH]

TARGET_COL = 'target'
FEATURE_COLS = ['duration', 'credit_amount', 'age', 'residence_since', 'existing_credits', 'num_dependents']
PREDICTION_COL = 'prediction'

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Loan-Governance-MLflow"
MODEL_NAME = "loan-creditscale-model"

# LocalStack S3 for MLflow Artifacts
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:4566"
os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    return train_test_split(df, test_size=0.2, random_state=42)

def train_and_evaluate(version: str, use_mlflow: bool):
    print(f"\n{'='*60}")
    print(f"PIPELINE FOR VERSION {version.upper()} (MLFLOW: {use_mlflow})")
    print("="*60)
    
    train_df, test_df = load_data()
    y_train = train_df[TARGET_COL]
    
    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
                mlflow.create_experiment(EXPERIMENT_NAME, artifact_location="s3://mlflow-artifacts/")
            mlflow.set_experiment(EXPERIMENT_NAME)
        except Exception as e:
            print(f"⚠ MLflow connection warning: {e}. Running without server sync.")
            use_mlflow = False

    run = None
    if use_mlflow:
        run = mlflow.start_run(run_name=f"loan-{version}")
        mlflow.log_params({"version": version, "dataset_size": len(train_df)})

    # 1. PRE-TRAINING GOVERNANCE
    print("[Venturalitica] 🛡️ Checking Training Data for Bias (Gender & Age)...")
    
    # 1.1 Add age grouping for complex policy binding
    train_df['age_group'] = pd.cut(train_df['age'], bins=[0, 25, 45, 100], labels=['young', 'adult', 'senior'])
    
    venturalitica.enforce(
        data=train_df, 
        target=TARGET_COL, 
        gender='gender',
        age='age',
        age_group='age_group',
        policy=POLICIES
    )

    # 2. TRAINING
    print(f"[Training] Version {version}...")
    
    # Start carbon tracking
    tracker = None
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(project_name=f"loan-model-{version}")
        tracker.start()
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])
    
    # Stop carbon tracking
    emissions_kg = 0.0
    if tracker:
        emissions_kg = tracker.stop()
        print(f"[Green AI] Carbon emissions: {emissions_kg:.6f} kgCO₂")

    # 3. POST-TRAINING GOVERNANCE
    y_pred = pipeline.predict(test_df[FEATURE_COLS]) # Predict using only feature columns
    eval_df = test_df.copy()
    eval_df[PREDICTION_COL] = y_pred
    
    print("[Venturalitica] 🛡️ Checking Model Compliance (Gender & Age)...")
    eval_df['age_group'] = pd.cut(eval_df['age'], bins=[0, 25, 45, 100], labels=['young', 'adult', 'senior'])
    
    results = venturalitica.enforce(
        data=eval_df,
        target=TARGET_COL,
        prediction=PREDICTION_COL,
        gender='gender',
        age='age',
        age_group='age_group',
        policy=POLICIES
    )

    # 4. LOGGING
    metrics = {
        "accuracy": accuracy_score(test_df[TARGET_COL], y_pred),
        "f1": f1_score(test_df[TARGET_COL], y_pred)
    }
    
    # Extract specific compliance metrics for MLflow
    for res in results:
        metrics[f"compliance_{res.control_id}"] = float(res.passed)

    if use_mlflow:
        mlflow.log_metrics(metrics)
        # Log carbon emissions
        if emissions_kg > 0:
            mlflow.log_metric("carbon_emissions_kg", emissions_kg)
        signature = infer_signature(test_df[FEATURE_COLS], y_pred)
        mlflow.sklearn.log_model(pipeline, "model", signature=signature)
        mlflow.end_run()

    print(f"  ✓ Process finished. Run ID: {run.info.run_id if run else 'local'}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, choices=['mlflow', 'local'], default='local')
    args = parser.parse_args()

    use_mlflow = (args.framework == 'mlflow')
    results_v1 = train_and_evaluate('v1', use_mlflow)
    
    def get_val(res, key):
        for r in res:
            if r.metric_key == key: return r.actual_value
        return 0.0

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"V1 Accuracy: {get_val(results_v1, 'accuracy_score'):.3f}")
    print(f"V1 Disparity: {get_val(results_v1, 'demographic_parity_diff'):.3f}")

if __name__ == "__main__":
    main()
