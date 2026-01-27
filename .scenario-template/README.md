# Scenario: {DOMAIN_NAME} ({MLOPS_TOOL} + {FRAMEWORK})

## 📌 Quick Metadata

| Property | Value |
|----------|-------|
| **Level** | ⭐ Beginner / ⭐⭐ Intermediate / ⭐⭐⭐ Advanced |
| **Duration** | ~15 min |
| **Concepts** | Class Imbalance, Demographic Parity, ... |
| **Skills Gained** | SDK enforcement, Policy writing, [MLOps] integration |
| **Real Data** | ✅ UCI CSV / ⚠️ API Required / ❌ Synthetic |
| **Status** | ✅ Production-Ready / ⚠️ In Development |

---

## 🎯 What You'll Build

**Before**: Model with unknown fairness properties  
**After**: Model with verified fairness properties + compliance report

```
Dataset → Pre-training Audit → Training → Post-training Audit → Report
         [Data Quality]                   [Model Fairness]
```

---

## 📋 Learning Outcomes

By completing this scenario, you'll understand:

- [ ] How bias manifests in {DOMAIN_NAME} systems
- [ ] How to detect {METRIC} using Venturalitica SDK
- [ ] How to integrate with {MLOPS_TOOL}
- [ ] How to write {DOMAIN_NAME}-specific OSCAL policies
- [ ] How to interpret fairness metrics and compliance reports

---

## 🚀 5-Minute Quick Start

```bash
# 1. Setup
cd scenarios/{scenario-name}
uv sync

# 2. Run
uv run python train.py

# 3. Check output
cat results/compliance_report.json
```

**Expected output:**
- ✅ All policies PASSED
- 📊 Fairness metrics displayed
- 📋 Audit log generated

---

## 📚 Full Tutorial

### Step 1: Understand the Data

```bash
head -20 ../../datasets/{domain}/{dataset}.csv
wc -l ../../datasets/{domain}/{dataset}.csv
```

**What to notice:**
- Column structure
- Protected attributes (gender, age, etc.)
- Class distribution (is it balanced?)

### Step 2: Examine the Training Script

```bash
cat train.py | head -50
```

**Key sections:**
1. **Configuration** (lines 1-30): Paths, columns, policy references
2. **Pre-training Governance** (lines ~60-80): Data quality checks
3. **Training** (lines ~85-110): Model fitting
4. **Post-training Governance** (lines ~115-135): Fairness checks
5. **Logging** (lines ~140-155): Report generation

### Step 3: Run Pre-training Audit

```bash
uv run python -c "
import pandas as pd
import venturalitica as vl
from pathlib import Path

# Load data
df = pd.read_csv('../../datasets/{domain}/{dataset}.csv')
print(f'Dataset shape: {df.shape}')
print(f'Missing values:\\n{df.isnull().sum()}')

# Pre-training audit
results = vl.enforce(
    data=df,
    target='target',
    gender='gender',  # or relevant protected attribute
    policy=['../../policies/{domain}/*.oscal.yaml']
)

for r in results:
    print(f'{r.control_id}: {\"PASS\" if r.passed else \"FAIL\"}')
"
```

### Step 4: Train Model

```bash
uv run python train.py --framework local
```

**What happens:**
1. Data loaded and audited
2. Model trained on {DATASET_SIZE} samples
3. Fairness metrics calculated
4. Compliance report generated

### Step 5: Interpret Results

```bash
# View compliance report
cat results/compliance_report.json | python -m json.tool

# View metrics
cat results/fairness_metrics.csv
```

**Key metrics in {DOMAIN_NAME}:**
- **Demographic Parity Difference**: Should be < 0.10
- **Equal Opportunity Difference**: Should be < 0.15
- **Class Imbalance**: Training minority class should be > 20%

---

## 🧠 Understanding the Code

### Pre-training Governance

```python
# Why: Detect bias BEFORE training (cheaper than retraining)
venturalitica.enforce(
    data=train_df,
    target='target',           # Target column
    gender='gender',           # Protected attribute
    age='age',                 # Another protected attribute
    policy=POLICIES            # OSCAL rules
)
```

**What gets checked:**
- Is target class balanced?
- Are protected attributes evenly represented?
- Any obvious red flags in the data?

### Post-training Governance

```python
# Why: Verify model learned fair decision boundaries
predictions = model.predict(test_df)

venturalitica.enforce(
    data=test_df,
    target='target',
    prediction=predictions,    # ← NEW: Model predictions
    gender='gender',
    age='age',
    policy=POLICIES
)
```

**What gets checked:**
- Does model treat groups equally?
- Is accuracy consistent across demographics?
- Any disparate impact?

### Policy Structure

```yaml
# Each policy controls one fairness principle

assessment-plan:
  metadata:
    title: "Fair {DOMAIN_NAME} Policy"
  
  reviewed-controls:
    control-selections:
      - include-controls:
          - control-id: no-class-imbalance
            description: "Minority class ≥ 20% of training data"
            props:
              - name: metric_key
                value: class_imbalance
              - name: threshold
                value: "0.20"
              - name: operator
                value: ">="
```

---

## 🔧 Customization: Try This!

### Experiment 1: Test with Different Thresholds

```bash
# Modify policies/fairness-strict.oscal.yaml
# Change: threshold: "0.05"  (more strict)
# Run: uv run python train.py

# Compare results. When does the model fail?
```

### Experiment 2: Use Different Protected Attributes

```python
# In train.py, add:
venturalitica.enforce(
    data=df,
    target='target',
    gender='gender',
    age='age',
    race='race',           # ← NEW
    policy=POLICIES
)
```

### Experiment 3: Compare Models

```bash
# Create train_v1.py (original model)
# Create train_v2.py (with sample weighting)
# Run both and compare fairness metrics

# Which performs better on accuracy vs fairness?
```

### Experiment 4: With MLOps Tracking

```bash
# With MLflow
uv run python train.py --framework mlflow

# With Weights & Biases
uv run python train.py --framework wandb

# View in MLflow UI: http://localhost:5000
# View in W&B: https://wandb.ai/your-project
```

---

## ✅ Validation Checklist

- [ ] Script runs without errors
- [ ] All fairness metrics computed
- [ ] Compliance report generated in `results/`
- [ ] Output metrics make sense for {DOMAIN_NAME}
- [ ] At least one experiment tried from Customization section

---

## 🆘 Troubleshooting

### Dataset Not Found

```bash
# Download datasets (real sources)
cd ../../datasets
uv run download_real_data.py
```

### Policy Enforcement Fails

Check that:
1. Policy file exists: `ls ../../policies/{domain}/`
2. Protected attribute columns exist in data: `df.columns`
3. Policy YAML is valid: `yamllint ../../policies/{domain}/*.yaml`

### Model Training Is Slow

```bash
# Use subset of data:
uv run python train.py --sample-size 1000

# Use CPU only (if GPU available):
export CUDA_VISIBLE_DEVICES=""
uv run python train.py
```

### MLOps Integration Not Working

```bash
# For MLflow:
docker compose -f ../../docker-compose.dev.yml up mlflow -d
uv run python train.py --framework mlflow

# For W&B:
wandb login
uv run python train.py --framework wandb
```

---

## 📊 Expected Output

```
[Venturalitica] 🛡️ Checking Training Data for Bias...
  ✓ PASS: class_imbalance = 0.30 >= 0.20
  ✓ PASS: disparate_impact_gender = 0.92 >= 0.80

[Training] Training {DOMAIN_NAME} Model...
  Model type: {MODEL_TYPE}
  Epochs: {EPOCHS}
  Training samples: {N_TRAIN}

[Venturalitica] 🛡️ Checking Model Compliance...
  ✓ PASS: demographic_parity_diff_gender = 0.08 < 0.10
  ✓ PASS: equal_opportunity_diff_gender = 0.06 < 0.15

[Metrics]
  Accuracy: 0.82
  Precision: 0.80
  Recall: 0.79
  F1-Score: 0.79

╔════════════════════════════════════════╗
║     ✅ ALL CHECKS PASSED              ║
║  Model approved for production use.    ║
╚════════════════════════════════════════╝
```

---

## 🔗 Next Steps

### Option A: Understand More Concepts
- Read: [Demographic Parity Explained](../../../packages/venturalitica-sdk/docs/concepts/demographic-parity.md)
- Read: [Class Imbalance Deep Dive](../../../packages/venturalitica-sdk/docs/concepts/class-imbalance.md)

### Option B: Try Another Scenario
- **Similar Domain**: [Scenario 2](../scenario-2-name)
- **More Advanced**: [Scenario 3](../scenario-3-name)
- **Different Concept**: [Scenario 4](../scenario-4-name)

### Option C: Write Custom Policies
- Guide: [OSCAL Authoring](../../../packages/venturalitica-sdk/docs/oscal-authoring.md)
- Template: Use this scenario's `policies/{domain}/` as starting point

### Option D: Integrate with Your Project
- Guide: [MLOps Integration](../../../packages/venturalitica-sdk/docs/mlops-integration.md)
- Example: `train.py` shows [MLOPS_TOOL] integration pattern

---

## 📚 References

- **SDK Docs**: [Venturalitica Documentation](../../../packages/venturalitica-sdk/docs/README.md)
- **Fairness Concepts**: [Core Concepts](../../../packages/venturalitica-sdk/docs/core-concepts.md)
- **Policy Writing**: [OSCAL Authoring](../../../packages/venturalitica-sdk/docs/oscal-authoring.md)
- **{DOMAIN_NAME} Background**: [Domain Primer](./docs/domain-primer.md)
- **Compliance Mapping**: [EU AI Act Article 9-15](../../../packages/venturalitica-sdk/docs/compliance-mapping.md)

---

## 📝 Notes

- **Data**: Downloaded from {DATA_SOURCE} on first run
- **Reproducibility**: Fixed random seed (42) for all runs
- **Privacy**: All processing local, no data uploaded
- **License**: {DATA_LICENSE}

---

**Last Updated**: January 2026  
**SDK Version**: Compatible with 1.2.0+  
**Python**: 3.10+  
**Estimated Time**: ~15 minutes
