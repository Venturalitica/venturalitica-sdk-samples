"""
🎯 LEVEL 2: SDK + ML + MLOps Tracking
================================================================================
⏱️ Time: 15 minutes
🎓 Complexity: ⭐⭐⭐ Advanced
🎯 Goal: Add experiment tracking for team collaboration

What you'll learn:
- How to log compliance metrics to MLflow
- Team workflows with centralized tracking
- Reproducibility and artifact management

Prerequisites:
- MLflow server running (or use --framework local)
================================================================================
"""

import pandas as pd
import venturalitica as vl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

# MLOps imports
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

print("🚀 Credit Scoring with MLOps Tracking\n")
print("="*60)

# Parse arguments
def train_and_evaluate(use_mlflow: bool, use_wandb: bool):
    print(f"\n{'='*60}")
    print(f"PIPELINE: MLflow={use_mlflow} | WandB={use_wandb}")
    print("="*60)

    # 2. LOAD DATA (moved here to be inside the function scope)
    print("\n📊 Step 1: Load ing data...")
    dataset_path = Path(__file__).parent.parent.parent / "datasets/loan/german_credit.csv"

    if not dataset_path.exists():
        df = vl.load_sample('loan')
    else:
        df = pd.read_csv(dataset_path)

    print(f"  ✓ Loaded {len(df)} loan applications")

    # For Level 2, we keep it simple: use only numeric features
    X = df.select_dtypes(include=['number']).drop(columns=['target'], errors='ignore')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. SETUP MLFLOW
    if use_mlflow:
        print("\n📊 Configuring MLflow...")
        mlflow.set_tracking_uri("http://localhost:5000")
        try:
            experiment_name = "Loan-Governance-Demo"
            if not mlflow.get_experiment_by_name(experiment_name):
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name="loan-fairness-v1")
            mlflow.log_params({"model": "LogisticRegression", "dataset_size": len(df)})
        except Exception as e:
            print(f"  ⚠️  MLflow failed: {e}")
            use_mlflow = False

    # 1b. SETUP WANDB
    if use_wandb:
        print("\n📊 Configuring WandB...")
        try:
            import wandb
            wandb.init(project="Loan-Governance-Demo", name="loan-fairness-v1", mode="online")
            wandb.config.update({"model": "LogisticRegression", "dataset_size": len(df)})
        except ImportError:
            print("  ⚠️  wandb not installed. Run: pip install wandb")
            use_wandb = False

    # 2. PRE-TRAINING AUDIT
    print("\n🛡️  Step 2: Pre-training audit...")
    policy_path = Path(__file__).parent.parent.parent / "policies/loan/risks.oscal.yaml"
    
    pre_results = vl.enforce(
        data=df,
        target='target',
        gender='gender',
        age='age',
        policy=str(policy_path)
    )

    # 3. TRAINING
    print("\n🤖 Step 3: Training model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"  ✓ Model trained: {accuracy:.1%} accuracy")

    if use_mlflow: mlflow.log_metric("accuracy", accuracy)
    if use_wandb: wandb.log({"accuracy": accuracy})

    # 4. POST-TRAINING AUDIT (FULL TRANSPARENCY)
    print("\n🛡️  Step 4: Post-training fairness audit...")
    test_df = df.iloc[X_test.index].copy()
    test_df['prediction'] = model.predict(X_test)
    
    fairness_policy = Path(__file__).parent.parent.parent / "policies/loan/governance-baseline.oscal.yaml"
    
    results = vl.enforce(
        data=test_df,
        target='target',
        prediction='prediction',
        gender='gender',
        age='age',
        policy=str(fairness_policy)
    )

    # 5. INTEGRATED LOGGING (FULL TRANSPARENCY)
    print("\n📦 Step 5: Logging Governance Evidence...")
    
    # Prepare metrics dict for frictionless logging
    gov_metrics = {}
    for r in pre_results: # Log pre-training results
        gov_metrics[f"pre_training.governance.{r.control_id}.pass"] = 1.0 if r.passed else 0.0
        gov_metrics[f"pre_training.governance.{r.control_id}.value"] = r.actual_value
        gov_metrics[f"pre_training.governance.{r.control_id}.threshold"] = r.threshold

    for r in results: # Log post-training results
        # 1. Pass/Fail Boolean
        gov_metrics[f"post_training.governance.{r.control_id}.pass"] = 1.0 if r.passed else 0.0
        # 2. Detailed Value (Control ID based)
        gov_metrics[f"post_training.governance.{r.control_id}.value"] = r.actual_value
        
        # 3. Explicit Metric Key Value (e.g. fairness.demographic_parity_diff)
        # This satisfies "Full Metrics" requirement by exposing the raw technical signal
        gov_metrics[f"post_training.metrics.{r.metric_key}"] = r.actual_value

    gov_metrics["governance.overall_pass"] = 1.0 if all(r.passed for r in results) else 0.0

    # -- MLFLOW LOGGING --
    if use_mlflow:
        mlflow.log_metrics(gov_metrics)
        # Log Tags for easy filtering
        for r in results:
            mlflow.set_tag(f"gov.{r.control_id}", "PASS" if r.passed else "FAIL")
        
        # Log Model (Robust)
        try:
            signature = infer_signature(X_test, model.predict(X_test))
            mlflow.sklearn.log_model(model, "model", signature=signature)
        except Exception as e:
            print(f"  ⚠️  MLflow Artifact Upload Failed (S3 Permissions?): {e}")
            print("  ✓ Metrics were still logged successfully.")
        
        mlflow.end_run()

    # -- WANDB LOGGING --
    if use_wandb:
        wandb.log(gov_metrics)
        # Summary for easy dashboarding
        for r in results:
            wandb.run.summary[f"gov.{r.control_id}"] = "PASS" if r.passed else "FAIL"
        
        # Log Artifact (The Glass Box Report)
        report_lines = ["# Governance Compliance Report", "", "| Control | Result | Value | Threshold |", "|---|---|---|---|"]
        for r in results:
            icon = "✅" if r.passed else "❌"
            report_lines.append(f"| {r.control_id} | {icon} | {r.actual_value:.3f} | {r.threshold} |")
        
        report_text = "\n".join(report_lines)
        
        with open("governance_report.md", "w") as f:
            f.write(report_text)
        
        artifact = wandb.Artifact("governance_report", type="compliance")
        artifact.add_file("governance_report.md")
        wandb.log_artifact(artifact)
        wandb.finish()

    # 8. SUMMARY
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"  Model Accuracy: {accuracy:.1%}")
    print(f"  Compliance: {sum(r.passed for r in results)}/{len(results)} controls passed")

    if use_mlflow:
        # Assuming mlflow.active_run() is still available or we captured the run object
        # For simplicity, we'll just print the generic URI if MLflow was used.
        print(f"\n📊 View in MLflow: http://localhost:5000")

    print("\n📚 Next Steps:")
    print("  • Compare multiple runs in MLflow UI")
    print("  • Try different hyperparameters")
    print("  • Open 03_production_pipeline.py for full production setup (carbon tracking, etc.)")

    return results

def main():
    parser = argparse.ArgumentParser()
    # Updated choices to allow both
    parser.add_argument("--framework", type=str, choices=['mlflow', 'wandb', 'both', 'local'], default='local')
    args = parser.parse_args()

    use_mlflow = args.framework in ['mlflow', 'both']
    use_wandb = args.framework in ['wandb', 'both']

    if args.framework == 'local':
        print("Using LOCAL mode (No tracking server)")

    train_and_evaluate(use_mlflow, use_wandb)

if __name__ == "__main__":
    main()
