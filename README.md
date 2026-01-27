# Venturalitica SDK Samples 🛡️

Demo examples for the [Venturalitica SDK](https://github.com/venturalitica/venturalitica-sdk).

**⚡ Zero to Governance Report in 60 seconds.**

---

## 🚀 Quick Start

```bash
# Install SDK
pip install venturalitica

# Clone samples
git clone https://github.com/venturalitica/venturalitica-sdk-samples.git
cd venturalitica-sdk-samples/scenarios/loan-mlflow-sklearn

# Install dependencies
uv sync

# Run your first audit
uv run python 00_minimal.ipynb
```

---

## 📚 The Learning Path

This repo contains a single, well-documented scenario: **Loan Approval Fairness**.

| Level | File | Focus | Time |
|:------|:-----|:------|:-----|
| **00** | `00_minimal.ipynb` | First audit in 60 seconds | 2 min |
| **01** | `01_governance_audit.ipynb` | Full ML training with audits | 10 min |
| **02** | `02_mlops_tracking.py` | MLflow integration | 20 min |
| **03** | `03_production_pipeline.py` | Production-ready pipeline | 45 min |

---

## 🎯 What You'll Achieve

After completing the loan scenario:

- ✅ Detect historical bias in credit approval data
- ✅ Measure Demographic Parity and Equal Opportunity
- ✅ Generate EU AI Act-compliant audit reports
- ✅ Integrate governance into MLflow experiments

---

## 🎓 More Scenarios

Looking for more examples? Check out:

➡️ **[venturalitica-sdk-samples-extra](https://github.com/venturalitica/venturalitica-sdk-samples-extra)**

Advanced scenarios for Healthcare, Computer Vision, and LLM auditing.

---

## 📂 Structure

```
venturalitica-sdk-samples/
├── datasets/                ← UCI German Credit dataset
├── policies/                ← OSCAL governance policies
└── scenarios/
    └── loan-mlflow-sklearn/ ← The canonical demo
```

---

## 🛡️ Powered by OSCAL

Governance policies map to:
- **EU AI Act** (Articles 9-15)
- **NIST AI RMF**
- **ISO/IEC 42001**

---

**Last Updated**: January 2026 | SDK Version: 0.1
