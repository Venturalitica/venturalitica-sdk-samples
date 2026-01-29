"""
🎯 TRAIN V1: Baseline Training with BIASED Dataset
================================================================================
⏱️ Time: 5 minutes
🎓 Complexity: ⭐⭐ Intermediate
🎯 Goal: Train a credit scoring model WITHOUT bias mitigation (will FAIL fairness checks)

This script intentionally trains on imbalanced data to demonstrate:
- Real bias in ML models when not properly addressed
- How the platform detects fairness violations
- The need for mitigation strategies (addressed in V2)

Prerequisites:
- SDK installed locally
- SaaS instance running
- Authenticated via `venturalitica login`
================================================================================
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import venturalitica as vl
from datetime import datetime

print("\n" + "=" * 80)
print("🎯 TRAIN V1: Baseline Training (BIASED - Expected to Fail Fairness)")
print("=" * 80)

# 1. LOAD DATA WITH DEMOGRAPHIC IMBALANCE
print("\n📊 Step 1: Loading German Credit dataset...")
dataset_path = Path(__file__).parent.parent.parent / "datasets/loan/german_credit.csv"

if not dataset_path.exists():
    df = vl.load_sample("loan")
else:
    df = pd.read_csv(dataset_path)

print(f"  ✓ Loaded {len(df)} loan applications")

# Identify gender and age columns
gender_col = "Attribute9" if "Attribute9" in df.columns else "gender"
age_col = "Attribute13" if "Attribute13" in df.columns else "age"
target_col = "class" if "class" in df.columns else "target"

print(f"  ✓ Using columns: target={target_col}, gender={gender_col}, age={age_col}")

# Create binary gender if needed (female=0, male=1)
if gender_col in df.columns:
    # Ensure binary gender
    unique_genders = df[gender_col].unique()

    # If string-based gender (male/female), convert to numeric
    if isinstance(unique_genders[0], str):
        df[gender_col] = (df[gender_col] == "male").astype(int)
    elif len(unique_genders) > 2:
        # Use median as threshold
        median_val = df[gender_col].median()
        df[gender_col] = (df[gender_col] >= median_val).astype(int)

# 2. PRE-TRAINING AUDIT (EXPECTED TO SHOW BIAS)
print("\n🛡️  Step 2: Pre-training fairness audit...")
policy_path = Path(__file__).parent / "policies/loan/data_policy.oscal.yaml"

# Run data quality checks
data_audit = vl.enforce(
    data=df,
    target=target_col,
    gender=gender_col,
    age=age_col,
    policy=str(policy_path) if policy_path.exists() else None,
)

print(f"  ✓ Data audit completed")
print(f"    - Records: {len(df)}")
print(f"    - Gender distribution: {df[gender_col].value_counts().to_dict()}")
print(f"    - Target distribution: {df[target_col].value_counts().to_dict()}")

# 3. PREPARE DATA
print("\n🔧 Step 3: Preparing training data...")
X = df.select_dtypes(include=["number"]).drop(columns=[target_col], errors="ignore")
y = df[target_col]

# Create train/test split with random_state=42 (reproducible bias)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ✓ Train set: {len(X_train)} samples")
print(f"  ✓ Test set: {len(X_test)} samples")

# 4. TRAIN MODEL (NO BIAS MITIGATION)
print("\n🤖 Step 4: Training LogisticRegression (NO mitigation)...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 5. EVALUATE MODEL AND COMPUTE FAIRNESS METRICS
print("\n📈 Step 5: Computing metrics...")
accuracy = accuracy_score(y_test, y_pred)
print(f"  ✓ Accuracy: {accuracy:.4f}")

# Compute demographic parity (WILL BE BIASED)
# Demographic parity: |P(y=1 | gender=0) - P(y=1 | gender=1)|
test_data_with_pred = X_test.copy()
test_data_with_pred["gender"] = df.loc[X_test.index, gender_col].values
test_data_with_pred["age"] = df.loc[X_test.index, age_col].values
test_data_with_pred["prediction"] = y_pred

# Split by gender
group_0_pred_rate = (
    test_data_with_pred[test_data_with_pred["gender"] == 0]["prediction"] == 1
).mean()
group_1_pred_rate = (
    test_data_with_pred[test_data_with_pred["gender"] == 1]["prediction"] == 1
).mean()

demographic_parity_diff = abs(group_0_pred_rate - group_1_pred_rate)

# Handle nan values
if np.isnan(demographic_parity_diff):
    demographic_parity_diff = 0.0

print(f"  ✓ Demographic Parity Difference: {demographic_parity_diff:.4f}")
print(f"    - Gender=0 approval rate: {group_0_pred_rate:.4f}")
print(f"    - Gender=1 approval rate: {group_1_pred_rate:.4f}")

# Compute equalized odds (WILL BE BIASED)
# Equal opportunity: |P(y_pred=1 | y=1, gender=0) - P(y_pred=1 | y=1, gender=1)|
# Add actual labels back
test_data_with_pred["actual"] = df.loc[X_test.index, target_col].values

group_0_tpr = (
    test_data_with_pred[
        (test_data_with_pred["gender"] == 0) & (test_data_with_pred["actual"] == 1)
    ]["prediction"]
    == 1
).mean()
group_1_tpr = (
    test_data_with_pred[
        (test_data_with_pred["gender"] == 1) & (test_data_with_pred["actual"] == 1)
    ]["prediction"]
    == 1
).mean()

equalized_odds_diff = abs(group_0_tpr - group_1_tpr)

# Handle nan values
if np.isnan(equalized_odds_diff):
    equalized_odds_diff = 0.0

print(f"  ✓ Equalized Odds Difference: {equalized_odds_diff:.4f}")
print(f"    - Gender=0 TPR: {group_0_tpr:.4f}")
print(f"    - Gender=1 TPR: {group_1_tpr:.4f}")

# 6. POST-TRAINING AUDIT (WILL SHOW VIOLATIONS)
print("\n🛡️  Step 6: Post-training fairness audit...")
post_audit = vl.enforce(
    data=df,
    predictions=y_pred,
    target=target_col,
    gender=gender_col,
    age=age_col,
    policy=str(policy_path) if policy_path.exists() else None,
)

print(f"  ✓ Post-training audit completed")

# 7. PREPARE RESULTS FOR SDK PUSH
print("\n📦 Step 7: Preparing results for SDK...")
results = {
    "metrics": [
        {
            "name": "accuracy_score",
            "value": float(accuracy),
            "threshold": 0.70,
            "passed": bool(accuracy >= 0.70),
        },
        {
            "name": "demographic_parity_diff",
            "value": float(demographic_parity_diff),
            "threshold": 0.10,
            "passed": bool(demographic_parity_diff < 0.10),  # EXPECTED: False (FAIL)
        },
        {
            "name": "equalized_odds_diff",
            "value": float(equalized_odds_diff),
            "threshold": 0.10,
            "passed": bool(equalized_odds_diff < 0.10),  # EXPECTED: False (FAIL)
        },
    ],
    "training_metadata": {
        "model_type": "LogisticRegression",
        "training_timestamp": datetime.utcnow().isoformat(),
        "bias_mitigation": "NONE (Baseline)",
        "dataset_size": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    },
}

# Write results to .venturalitica/results.json
results_dir = Path(".venturalitica")
results_dir.mkdir(exist_ok=True)

results_file = results_dir / "results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"  ✓ Results saved to {results_file}")

# 8. SUMMARY
print("\n" + "=" * 80)
print("✅ V1 TRAINING COMPLETE")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"  • Accuracy: {accuracy:.4f} ({'✓ PASS' if accuracy >= 0.70 else '✗ FAIL'})")
print(
    f"  • Demographic Parity Diff: {demographic_parity_diff:.4f} ({'✓ PASS' if demographic_parity_diff < 0.10 else '✗ FAIL - BIAS DETECTED'})"
)
print(
    f"  • Equalized Odds Diff: {equalized_odds_diff:.4f} ({'✓ PASS' if equalized_odds_diff < 0.10 else '✗ FAIL - BIAS DETECTED'})"
)
print(f"\n⚠️  Expected: FAIRNESS VIOLATIONS DETECTED")
print(f"    This is the baseline - V2 will fix these issues with mitigation")
print("\n→ Next: Run `venturalitica push` to submit results to SaaS")
print("=" * 80 + "\n")
