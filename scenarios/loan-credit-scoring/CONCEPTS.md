# Concepts: Loan Risk Assessment with MLflow & Scikit-learn

## Overview
This scenario demonstrates fairness-aware credit risk modeling using real loan data, MLflow experiment tracking, and Venturalitica's governance framework.

## Key Concepts

### 1. **Credit Risk Modeling**
- **Target**: Loan approval/rejection prediction
- **Data**: Real loan applications with demographics and financial indicators
- **Fairness Concern**: Equal access across demographic groups (EEOC Four-Fifths Rule)

### 2. **Protected Attributes**
- `gender`: M, F, or Other (fairness dimension for disparate impact analysis)
- Cross-tabulation with approval rates to detect systemic bias

### 3. **Metrics**
- **Disparate Impact Ratio**: Ratio of positive rates (approval rate minority / approval rate majority)
  - Compliant if ≥ 0.8 (80% Rule)
- **Demographic Parity Difference**: Absolute difference in approval rates across groups
  - Target: < 0.1 (10 percentage points)
- **Model Accuracy**: Overall correctness
  - Target: ≥ 0.75

### 4. **Governance Controls**
- Pre-training: Class balance check
- Post-training: Disparate impact enforcement
- Workflow: `prepare_data.py` → train model → `01_governance_audit.py` for validation

### 5. **MLflow Integration**
- Tracks model parameters, metrics, and artifacts
- Enables comparison across fairness-aware experiments
- Logs governance check results alongside model performance

## Workflow

```
1. prepare_data.py
   → Downloads UCI credit dataset
   → Encodes demographics
   → Splits into train/test with fairness stratification

2. 00_minimal.ipynb
   → Trains logistic regression baseline
   → Computes fairness metrics
   → Logs to MLflow

3. 01_governance_audit.py
   → Enforces governance policy (loan-credit.oscal.yaml)
   → Reports pass/fail per control
   → Blocks deployment if fairness thresholds violated
```

## Educational Value
- Shows real-world fairness metrics in credit scoring
- Demonstrates why synthetic metrics mislead (use real data)
- Illustrates governance-as-code using OSCAL policies
- Connects model training (MLflow) to compliance (Venturalitica)
