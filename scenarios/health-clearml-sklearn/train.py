import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
import venturalitica
import argparse
import joblib

# ClearML Import
try:
    from clearml import Task
except ImportError:
    Task = None

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "datasets/health"
DATA_FILE = DATA_DIR / "breast_cancer.csv"
HEALTH_POLICY_PATH = Path(__file__).parent.parent.parent / "policies/health/clinical-risk.oscal.yaml"

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, choices=['clearml', 'local'], default="local")
    args = parser.parse_args()

    use_clearml = (args.framework == 'clearml' and Task is not None)

    # 1. Load Data
    if not DATA_FILE.exists():
        print(f"❌ Data file not found: {DATA_FILE}")
        return
        
    df = pd.read_csv(DATA_FILE)
    
    if use_clearml:
        task = Task.init(project_name="Health-Governance-ClearML", task_name="breast-cancer-training")
        task.connect({"policy": str(HEALTH_POLICY_PATH)})

    # 2. Preparation
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Training
    print("  Training Clinical Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    y_pred = model.predict(X_test)
    eval_df = X_test.copy()
    eval_df['target'] = y_test
    eval_df['prediction'] = y_pred
    
    # 5. GOVERNANCE CHECK
    print(f"[Venturalitica] 🛡️ Checking Clinical Model Compliance...")
    results = venturalitica.enforce(
        data=eval_df,
        target='target',
        prediction='prediction',
        policy=[HEALTH_POLICY_PATH]
    )

    # 6. Logging
    metrics = {
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }
    
    if use_clearml:
        # Log to ClearML
        logger = Task.current_task().get_logger()
        for k, v in metrics.items():
            logger.report_scalar(title="performance", series=k, value=v, iteration=1)
            
        for res in results:
            logger.report_scalar(title="governance", series=res.control_id, value=int(res.passed), iteration=1)
        
        # Log Model
        joblib.dump(model, "clinical_model.joblib")
        task.upload_artifact(name="model", artifact_object="clinical_model.joblib")
        task.close()

    print(f"  ✓ Process finished. Mode: {args.framework}")

if __name__ == "__main__":
    train()
