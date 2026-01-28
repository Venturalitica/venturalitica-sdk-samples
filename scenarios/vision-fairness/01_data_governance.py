"""
🎯 VISION STEP 1: Data Governance (Article 10)
================================================================================
Audits the dataset quality, bias, and representation BEFORE training starts.
Connects to MLflow (localhost:5000) and WandB.
"""

import os
import pandas as pd
import venturalitica as vl
import mlflow
import wandb
from pathlib import Path
from dotenv import load_dotenv

# Config
load_dotenv()
DATA_PATH = Path("datasets/vision/fairface_cache/metadata.csv")
POLICY_PATH = "policies/vision/data_bias.oscal.yaml"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
SCALE = os.getenv("VL_DATA_SCALE", "POC")

def audit_data():
    print(f"🕵️  Scaling Mode: {SCALE}")
    
    if not DATA_PATH.exists():
        print("⚠️  Data missing. Run 'uv run prepare_data.py'")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    
    if SCALE == "POC":
        print("  ⚡ POC Mode: Using 2,000 samples for audit...")
        df = df.sample(n=min(2000, len(df)), random_state=42)
    else:
        print(f"  🏢 FULL Mode: Auditing all {len(df)} samples...")

    # 2. MLOps Init
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Vision-Data-Audit")
    
    wandb.init(project="vision-fairness", name=f"data-audit-{SCALE.lower()}")

    # 3. Article 10 Audit
    print(f"\n🛡️  [Article 10] Enforcing policy: {POLICY_PATH}")
    
    # Multimodal Monitor for transparency and hardware telemetry
    with vl.monitor("data_audit"):
        with mlflow.start_run(run_name=f"Data Audit ({SCALE})"):
            
            # Map columns to roles
            # 'target' in FairFace is not defined yet, we use a dummy or skip
            df['target_dummy'] = 0 
            
            results = vl.enforce(
                data=df,
                policy=POLICY_PATH,
                target='target_dummy',
                race='race',
                gender='gender'
            )
            
            # Logging to WandB
            if results:
                metrics = {}
                for r in results:
                    metrics[f"data.{r.control_id}"] = r.actual_value
                    metrics[f"data.{r.control_id}_pass"] = 1 if r.passed else 0
                wandb.log(metrics)
                print(f"  ✓ Logged {len(metrics)} metrics to WandB")

    wandb.finish()
    print("\n✅ Data Governance Step Complete. View results in 'venturalitica ui' or MLflow.")

if __name__ == "__main__":
    audit_data()
