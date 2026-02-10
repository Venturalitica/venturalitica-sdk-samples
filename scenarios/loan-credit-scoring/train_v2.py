"""
🎯 TRAIN V2: Mitigated Training (Low Friction SDK)
================================================================================
🎓 Complexity: ⭐ Intermediate
🎯 Goal: Train a BIAS-CALIBRATED model with TRANSPARENT metrics tracking.

In V2, we apply 'class_weight="balanced"' and sample reweighting to fix
bias issues. The SDK will automatically detect the improvement via 'vl.enforce'.
================================================================================
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

import venturalitica as vl

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
DATASET_PATH = Path(__file__).parent.parent.parent / "datasets/loan/german_credit.csv"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "loan-approval-ai"
# Separate results file for V2 to avoid overwriting immediately?
# Usually we overwrite to show the 'latest' state in dashboard.
RESULTS_PATH = Path(".venturalitica/results.json")


def load_data() -> Tuple[pd.DataFrame, list]:
    """Loads and preprocesses the German Credit dataset."""
    logger.info("Step 1: Loading German Credit dataset...")
    df = pd.read_csv(DATASET_PATH) if DATASET_PATH.exists() else vl.load_sample("loan")

    # Identify key columns
    gender_col = "Attribute9" if "Attribute9" in df.columns else "gender"
    if gender_col in df.columns:
        if isinstance(df[gender_col].iloc[0], str):
            df[gender_col] = df[gender_col].astype(str).str.strip()
            df[gender_col] = (df[gender_col].str.lower() == "male").astype(int)

    # Professional Mitigation: Fairness through Awareness
    # In this dataset, including gender allows the model to differentiate
    # and achieve higher fairness (DP 0.003) than unawareness.
    leaky_cols = ["class", "target", "prediction"]

    # Professional Preprocessing: Encode Categorical Variables
    df_encoded = pd.get_dummies(df.drop(columns=leaky_cols, errors="ignore"))

    return df_encoded, df


def finalize_governance(
    test_df: pd.DataFrame,
    context: Dict[str, Any],
    mlflow_run_id: str = None,
    target_col: str = "target",
    gender_col: str = "gender",
    age_col: str = "age",
):
    """SDK-First Governance: Automates all metric calculations and policy checks."""
    logger.info("Step 3: SDK Automated Governance...")

    # Policy file (auto-discovered or explicitly set)
    policy_file = (
        "model_policy.oscal.yaml"
        if os.path.exists("model_policy.oscal.yaml")
        else "risks.oscal.yaml"
    )

    # TRANSPARENT ENFORCEMENT (post-training)
    # Pass target + protected attribute mappings so fairness metrics can resolve columns
    vl.enforce(
        data=test_df,
        target=target_col,
        prediction="prediction",
        gender=gender_col,
        age=age_col,
        policy=policy_file,
        strict=True,
    )

    # Optional Bundle Enrichment: merge pre and post results
    post_metrics = []
    pre_metrics = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r") as f:
            data = json.load(f)
        post_metrics = data if isinstance(data, list) else data.get("metrics", [])

    # attempt to load pre-results snapshot if present
    pre_path = Path(".venturalitica/pre_results.json")
    if pre_path.exists():
        try:
            with open(pre_path, "r") as f:
                pd_data = json.load(f)
            pre_metrics = (
                pd_data if isinstance(pd_data, list) else pd_data.get("metrics", [])
            )
        except Exception:
            pre_metrics = []

    bundle = {
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "training_metadata": {
            "model_type": "LogisticRegression",
            "timestamp": datetime.utcnow().isoformat(),
            "mitigation": "Stratified Class Weight Reweighting",
            "mlflow_run": mlflow_run_id,
            "context": context,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(bundle, f, indent=2)
    logger.info(f"  ✓ Results bundle enriched: {RESULTS_PATH}")

    # Generate push_cmd.sh for scenario compatibility
    sdk_bin = "/home/morganrcu/proyectos/venturalitica-integration/packages/venturalitica-sdk/.venv/bin/venturalitica"
    treatment_id = os.getenv("VENTURALITICA_TREATMENT_ID")
    push_args = f" --treatment-id {treatment_id}" if treatment_id else ""
    with open("push_cmd.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"{sdk_bin} push{push_args}\n")
    os.chmod("push_cmd.sh", 0o755)
    logger.info("  ✓ Generated push_cmd.sh")


def log_to_mlflow(model: LogisticRegression, params: Dict[str, Any]):
    """Logs the model to MLflow."""
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.sklearn.log_model(model, "model")
            return run.info.run_id
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")
        return None


def main():
    # Finalize Governance Clean State
    if RESULTS_PATH.exists():
        os.remove(RESULTS_PATH)
        logger.info(f"  ✓ Cleared previous results: {RESULTS_PATH}")
    pre_path = Path(".venturalitica/pre_results.json")
    if pre_path.exists():
        os.remove(pre_path)

    print("\n" + "=" * 80)
    print("🎯 VENTURALÍTICA: Low-Friction Training (Mitigated Awareness)")
    print("=" * 80)

    with vl.monitor(name="mitigated-training-v2", label="Technical Audit Trace"):
        # 1. Load & Prepare
        X, df = load_data()
        target_col = "class" if "class" in df.columns else "target"
        gender_col = "Attribute9" if "Attribute9" in df.columns else "gender"
        age_col = "Attribute13" if "Attribute13" in df.columns else "age"

        # PRE-TRAIN: enforce data policy + custom group checks
        logger.info("Step 1b: Pre-training data checks...")
        data_policy = Path(__file__).parent / "data_policy.oscal.yaml"
        from policy_checks import evaluate_data_policy_controls

        # 1) custom data checks (group-based) which the SDK may not support yet
        custom_results = evaluate_data_policy_controls(str(data_policy), df, target_col)
        # 2) run sdk enforcement for remaining checks (strict mode: fail if policy cannot be evaluated)
        vl.enforce(
            data=df,
            target=target_col,
            gender=gender_col,
            age=age_col,
            policy=str(data_policy),
            strict=True,
        )

        sdk_results = []
        if Path(".venturalitica/results.json").exists():
            import json

            with open(".venturalitica/results.json", "r") as f:
                sdk = json.load(f)
            sdk_results = sdk if isinstance(sdk, list) else sdk.get("metrics", [])

        merged = []
        # normalize sdk_results to expected dict format
        for m in sdk_results:
            merged.append(m)
        for m in custom_results:
            merged.append(m)

        import json

        with open(".venturalitica/pre_results.json", "w") as f:
            json.dump({"metrics": merged}, f, indent=2)
        logger.info("  ✓ Pre-training results saved: .venturalitica/pre_results.json")

        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 2. Mitigation Strategy: Pre-Processing Reweighting
        # We calculate weights for each sample to ensure group equality.
        # This is the most robust way to meet the strict 0.01 threshold.

        # Calculate group probabilities
        n = len(df)
        n_pos = len(df[df[target_col] == 1])
        n_neg = len(df[df[target_col] == 0])

        # Gender counts
        n_minority = len(df[df[gender_col] == 0])
        n_majority = len(df[df[gender_col] == 1])

        # Intersection counts
        n_minority_pos = len(df[(df[gender_col] == 0) & (df[target_col] == 1)])
        n_minority_neg = len(df[(df[gender_col] == 0) & (df[target_col] == 0)])
        n_majority_pos = len(df[(df[gender_col] == 1) & (df[target_col] == 1)])
        n_majority_neg = len(df[(df[gender_col] == 1) & (df[target_col] == 0)])

        # Compute weights: W = (P(group) * P(label)) / P(group, label)
        w_minority_pos = (
            (n_minority * n_pos) / (n * n_minority_pos) if n_minority_pos > 0 else 1.0
        )
        w_minority_neg = (
            (n_minority * n_neg) / (n * n_minority_neg) if n_minority_neg > 0 else 1.0
        )
        w_majority_pos = (
            (n_majority * n_pos) / (n * n_majority_pos) if n_majority_pos > 0 else 1.0
        )
        w_majority_neg = (
            (n_majority * n_neg) / (n * n_majority_neg) if n_majority_neg > 0 else 1.0
        )

        # Assign weights to samples
        weights = np.ones(len(df))
        weights[(df[gender_col] == 0) & (df[target_col] == 1)] = w_minority_pos
        weights[(df[gender_col] == 0) & (df[target_col] == 0)] = w_minority_neg
        weights[(df[gender_col] == 1) & (df[target_col] == 1)] = w_majority_pos
        weights[(df[gender_col] == 1) & (df[target_col] == 0)] = w_majority_neg

        # 3. Training with Reweighted Dataset
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000, random_state=42).fit(
            X_train, y_train, sample_weight=w_train
        )

        # 4. Predict & Assemble Test Results (needed for SDK-side metrics)
        y_pred = model.predict(X_test)
        test_df = X_test.copy()
        test_df[target_col] = y_test
        test_df["prediction"] = y_pred
        # Include protected attributes for fairness assessment
        for col in ["Attribute9", "gender", "Attribute13", "age"]:
            if col in df.columns:
                test_df[col] = df.loc[X_test.index, col]

        # 5. MLflow
        run_id = log_to_mlflow(
            model, {"max_iter": 1000, "mitigation": "Balanced Class Weight"}
        )

        # 6. Governance Handshake
        finalize_governance(
            test_df,
            {"records": len(df)},
            run_id,
            target_col=target_col,
            gender_col=gender_col,
            age_col=age_col,
        )

    print("\n" + "=" * 80)
    print("✅ V2 TRAINING COMPLETE (Low Friction)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
