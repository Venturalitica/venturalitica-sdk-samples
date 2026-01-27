"""
🎯 LEVEL 3: Production Pipeline Wrapper
================================================================================
⏱️ Time: 45 minutes
🎓 Complexity: ⭐⭐⭐⭐ Expert
🎯 Goal: "Zero-touch" governance for automated pipelines

What you'll learn:
- How to use vl.wrap() to auto-audit sklearn Pipelines
- Green AI: Tracking carbon emissions (Scope 3)
- Production-grade MLOps (WandB + MLflow) with robust error handling

Prerequisites:
- Complete 03_mlops_integration.py
- .env file with WANDB_API_KEY
================================================================================
"""

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

# MLOps Imports
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
import venturalitica as vl
import venturalitica # Explicit import for .wrap() usage if needed, or just use vl

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "datasets/loan"
DATA_FILE = DATA_DIR / "german_credit.csv"
RISK_POLICY_PATH = Path(__file__).parent.parent.parent / "policies/loan/risks.oscal.yaml"
GOVERNANCE_POLICY_PATH = Path(__file__).parent.parent.parent / "policies/loan/governance-baseline.oscal.yaml"
POLICIES = [RISK_POLICY_PATH, GOVERNANCE_POLICY_PATH]

TARGET_COL = 'target'
FEATURE_COLS = ['duration', 'credit_amount', 'age', 'residence_since', 'existing_credits', 'num_dependents']

def load_data():
    if not DATA_FILE.exists():
        # Fallback to SDK loader if local file missing
        print("  ⚠ Local dataset not found, using SDK sample loader...")
        df = vl.load_sample('loan')
    else:
        df = pd.read_csv(DATA_FILE)
    return train_test_split(df, test_size=0.2, random_state=42)

def train_and_evaluate(version: str, use_mlflow: bool, use_wandb: bool):
    print(f"\n{'='*60}")
    print(f"PIPELINE {version.upper()}: MLflow={use_mlflow} | WandB={use_wandb}")
    print("="*60)
    
    train_df, test_df = load_data()
    y_train = train_df[TARGET_COL]
    
    # 1. SETUP MLOPS
    # --------------------------------------------------------------------------
    if use_mlflow:
        mlflow.set_tracking_uri("http://localhost:5000")
        try:
            exp_name = "Loan-Production-Pipeline"
            if not mlflow.get_experiment_by_name(exp_name):
                mlflow.create_experiment(exp_name)
            mlflow.set_experiment(exp_name)
        except Exception as e:
            print(f"⚠ MLflow connection warning: {e}. Running without server sync.")
            use_mlflow = False

    if use_wandb:
        try:
            import wandb
            wandb.init(project="Loan-Production-Pipeline", name=f"pipeline-{version}", mode="online")
        except ImportError:
            print("  ⚠️  wandb not installed. Run: pip install wandb")
            use_wandb = False
        except Exception as e:
            print(f"  ⚠️  WandB Init failed: {e}")
            use_wandb = False

    # Start MLflow run
    run = None
    if use_mlflow:
        run = mlflow.start_run(run_name=f"pipeline-{version}")
        mlflow.log_params({"version": version, "dataset_size": len(train_df)})

    # 2. TRAINING (Wrapped with Venturalitica)
    # --------------------------------------------------------------------------
    print(f"[Training] Version {version} with vl.wrap()...")
    
    # Start carbon tracking
    tracker = None
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(project_name=f"loan-model-{version}", output_dir=".")
        tracker.start()
    
    # Define standard sklearn pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # WRAP THE PIPELINE FOR AUTOMATIC GOVERNANCE
    # This automatically runs pre-training checks on fit()
    # and post-training checks on predict()
    pipeline = vl.wrap(pipeline, policy=POLICIES)
    
    # We pass the full train_df as audit_data/kwargs context for finding sensitive cols
    pipeline.fit(
        train_df[FEATURE_COLS], 
        train_df[TARGET_COL],
        audit_data=train_df, # Full context for metadata & policy
        gender='gender',     # Explicit column mapping if auto-detect fails
        age_group='age_group'
    )
    
    # Stop carbon tracking
    emissions_kg = 0.0
    if tracker:
        emissions_kg = tracker.stop()
        print(f"[Green AI] Carbon emissions: {emissions_kg:.6f} kgCO₂")

    # 3. PREDICTION (Auto-Audited)
    # --------------------------------------------------------------------------
    # The wrapper intercepts predict, performs the audit, and returns prediction.
    y_pred = pipeline.predict(
        test_df[FEATURE_COLS],
        audit_data=test_df, 
        gender='gender',
        age_group='age_group'
    )
    
    # Retrieve results directly from the pipeline wrapper
    results = pipeline.last_audit_results
    
    # 4. UNIFIED LOGGING (Full Metrics)
    # --------------------------------------------------------------------------
    metrics = {
        "accuracy": accuracy_score(test_df[TARGET_COL], y_pred),
        "f1": f1_score(test_df[TARGET_COL], y_pred),
        "carbon_emissions_kg": emissions_kg
    }
    
    # Extract Governance Evidence
    for res in results:
        # Binary Pass/Fail
        metrics[f"governance.{res.control_id}.pass"] = 1.0 if res.passed else 0.0
        # Detailed Technical Value (e.g. 0.012)
        metrics[f"governance.{res.control_id}.value"] = res.actual_value
        # Full Metric Key (e.g. fairness.demographic_parity_diff)
        metrics[f"metrics.{res.metric_key}"] = res.actual_value

    metrics["governance.overall_pass"] = 1.0 if all(r.passed for r in results) else 0.0

    # Log to MLflow
    if use_mlflow:
        mlflow.log_metrics(metrics)
        # Robust Artifact Upload
        try:
            signature = infer_signature(test_df[FEATURE_COLS], y_pred)
            mlflow.sklearn.log_model(pipeline, "model", signature=signature)
        except Exception as e:
            print(f"  ⚠️  MLflow Artifact Upload Failed (S3 Permissions?): {e}")
        mlflow.end_run()

    # Log to WandB
    if use_wandb:
        wandb.log(metrics)
        # Generate Glass Box Report
        report_lines = ["# Automated Pipeline Governance Report", "", "| Control | Result | Value | Metric |", "|---|---|---|---|"]
        for r in results:
            icon = "✅" if r.passed else "❌"
            report_lines.append(f"| {r.control_id} | {icon} | {r.actual_value:.3f} | `{r.metric_key}` |")
        
        with open("pipeline_report.md", "w") as f:
            f.write("\n".join(report_lines))
        
        artifact = wandb.Artifact("pipeline_governance", type="compliance")
        artifact.add_file("pipeline_report.md")
        wandb.log_artifact(artifact)
        wandb.finish()

    print(f"  ✓ Process finished.")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, choices=['mlflow', 'wandb', 'both', 'local'], default='local')
    parser.add_argument("--version", type=str, default="v1.0.0")
    args = parser.parse_args()

    use_mlflow = args.framework in ['mlflow', 'both']
    use_wandb = args.framework in ['wandb', 'both']

    if args.framework == 'local':
        print("Using LOCAL mode (No tracking server)")

    results = train_and_evaluate(args.version, use_mlflow, use_wandb)
    
    # Simple Local Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Compliance: {sum(r.passed for r in results)}/{len(results)} passed")

if __name__ == "__main__":
    main()
