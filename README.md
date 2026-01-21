# Venturalitica SDK Samples

Clean, industry-standard examples demonstrating AI governance with the [Venturalitica SDK](https://github.com/venturalitica/venturalitica-sdk).

## 📂 Repository Structure

```
venturalitica-sdk-samples/
├── datasets/          # Real-world datasets for governance testing
│   ├── loan/         # UCI German Credit dataset
│   ├── hiring/       # UCI Adult Income dataset
│   └── health/       # UCI Breast Cancer dataset
├── policies/          # OSCAL governance policies
│   ├── loan/         # Credit scoring fairness policies
│   ├── hiring/       # Recruitment bias policies
│   └── health/       # Clinical risk policies
└── scenarios/         # Complete training examples
    ├── loan-mlflow-sklearn/
    ├── hiring-wandb-torch/
    └── health-clearml-sklearn/
```

## 🎯 Scenarios

### 1. Loan Approval (Credit Scoring)
**Path**: `scenarios/loan-mlflow-sklearn/`

Demonstrates fairness audits for credit scoring models using the UCI German Credit dataset.

**Key Features**:
- Pre-training data quality checks (Class Imbalance, Disparate Impact)
- Post-training fairness metrics (Demographic Parity, Equal Opportunity)
- Multi-attribute monitoring (Gender, Age)
- MLflow integration for experiment tracking

**Run**:
```bash
cd scenarios/loan-mlflow-sklearn
uv run python train.py
```

**Policies Used**:
- `policies/loan/risks.oscal.yaml` - Domain-specific fairness controls
- `policies/loan/governance-baseline.oscal.yaml` - General AI governance baseline

---

### 2. Hiring Fairness (Recruitment)
**Path**: `scenarios/hiring-wandb-torch/`

Demonstrates bias detection in hiring models using the UCI Adult Income dataset.

**Key Features**:
- Gender and age bias detection
- PyTorch neural network training
- Weights & Biases integration
- Educational audit logs explaining the "80% Rule"

**Run**:
```bash
cd scenarios/hiring-wandb-torch
uv run python train.py
```

**Policies Used**:
- `policies/hiring/hiring-bias.oscal.yaml` - Recruitment-specific fairness controls

---

### 3. Clinical Risk Assessment (Healthcare)
**Path**: `scenarios/health-clearml-sklearn/`

Demonstrates medical diagnosis fairness using the UCI Breast Cancer dataset.

**Key Features**:
- Clinical risk assessment
- ClearML integration for healthcare MLOps
- Sensitivity/Specificity monitoring
- HIPAA-aligned governance controls

**Run**:
```bash
cd scenarios/health-clearml-sklearn
uv run python train.py
```

**Policies Used**:
- `policies/health/clinical-risk.oscal.yaml` - Healthcare-specific risk controls

## 📊 Datasets

All datasets are sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) and are preprocessed for governance testing.

### German Credit Dataset
- **Source**: UCI ML Repository
- **Use Case**: Credit scoring fairness
- **Protected Attributes**: Gender, Age
- **Target**: Loan approval (binary)

### Adult Income Dataset
- **Source**: UCI ML Repository
- **Use Case**: Hiring bias detection
- **Protected Attributes**: Gender, Age, Race
- **Target**: Income >50K (binary)

### Breast Cancer Dataset
- **Source**: UCI ML Repository
- **Use Case**: Clinical diagnosis fairness
- **Protected Attributes**: Age (derived)
- **Target**: Malignant/Benign (binary)

## 🛡️ OSCAL Policies

Each scenario includes specialized OSCAL policies that map to:
- **EU AI Act** Articles 9-15 (High-Risk AI Systems)
- **ISO 42001** AI Management System controls
- **NIST AI Risk Management Framework**

### Policy Structure

```yaml
assessment-plan:
  uuid: unique-policy-id
  metadata:
    title: "Policy Name"
  reviewed-controls:
    control-selections:
      - include-controls:
        - control-id: control-name
          description: "Educational description explaining WHY this matters"
          props:
            - name: metric_key
              value: demographic_parity_diff
            - name: threshold
              value: "0.10"
            - name: operator
              value: "<"
            - name: "input:dimension"  # Functional role
              value: gender             # Semantic variable
```

## 🚀 Getting Started

### Prerequisites

```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
pip install venturalitica-sdk
```

### Run a Sample

1. **Clone this repository**:
   ```bash
   git clone https://github.com/venturalitica/venturalitica-sdk-samples.git
   cd venturalitica-sdk-samples
   ```

2. **Choose a scenario**:
   ```bash
   cd scenarios/loan-mlflow-sklearn
   ```

3. **Run the training script**:
   ```bash
   uv run python train.py
   ```

4. **Review the audit log** in your terminal for compliance results.

## 📚 Learning Path

1. **Start with Loan scenario** - Simplest example with clear fairness metrics
2. **Explore Hiring scenario** - Multi-attribute governance
3. **Advanced: Health scenario** - Domain-specific controls for regulated industries

## 🤝 Contributing

We welcome contributions! To add a new scenario:

1. Create a new directory: `scenarios/{domain}-{mlops}-{framework}/`
2. Add a `train.py` script demonstrating SDK integration
3. Create domain-specific policies in `policies/{domain}/`
4. Add real or synthetic datasets to `datasets/{domain}/`
5. Update this README with your scenario

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## 🔗 Links

- [Venturalitica SDK](https://github.com/venturalitica/venturalitica-sdk)
- [SDK Documentation](https://github.com/venturalitica/venturalitica-sdk/tree/main/docs)
- [OSCAL Standard](https://pages.nist.gov/OSCAL/)
