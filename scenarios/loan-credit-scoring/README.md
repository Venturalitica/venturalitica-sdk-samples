# Scenario: Credit Scoring Fairness (Finance) 🏦

This scenario demonstrates how to use the Venturalitica SDK to detect and mitigate bias in a credit scoring system using the **UCI German Credit dataset** (Dataset #144, 1,000 real loan applications).

> 📓 **New!** Levels 00 and 01 are now available as **interactive Jupyter notebooks** (`.ipynb`) for better learning experience!

---

## 📚 Learning Path

This scenario is organized in 4 levels of increasing complexity. **Start with Level 00** for the quickest "Aha!" moment.

| Level | File | Format | Focus | Time |
| :--- | :--- | :--- | :--- | :--- |
| **00** | [`00_minimal.ipynb`](00_minimal.ipynb) 📓 | Notebook | **Aha Moment**: First audit in <60 seconds | 2 min |
| **01** | [`01_governance_audit.ipynb`](01_governance_audit.ipynb) 📓 | Notebook | **Integration**: Full ML training with pre/post audits | 10 min |
| **02** | [`02_metrics_deep_dive.py`](02_metrics_deep_dive.py) | Script | **Metrics**: Advanced fairness & privacy metrics | 15 min |
| **03** | [`03_mlops_integration.py`](03_mlops_integration.py) | Script | **MLOps**: Log compliant experiments to MLflow & WandB | 20 min |
| **04** | [`04_production_wrapper.py`](04_production_wrapper.py) | Script | **Production**: Auto-audit wrapper & Green AI | 45 min |

> 💡 **Tip:** Levels 00-01 use notebooks for interactive learning (Andrew Ng style). Levels 02-03 use scripts for production patterns.

---

## 🎯 What You'll Learn

- How to identify historical bias in credit approval datasets.
- How to measure **Demographic Parity** and **Equal Opportunity**.
- How to using sample re-weighting to mitigate bias.
- How to map technical metrics to **EU AI Act** compliance controls.

---

## 🚀 Quick Start

### Option A: Interactive Notebooks (Recommended for Learning) 📓
```bash
# Open Jupyter and run cell-by-cell
jupyter notebook 00_minimal.ipynb
```

### Option B: Run Scripts Directly
```bash
# Download dataset and run audit in one command
uv run prepare_data.py && uv run 01_governance_audit.py
```

---

## 🛡️ Governance Policy

This scenario uses the [`policies/loan/risks.oscal.yaml`](policies/loan/risks.oscal.yaml) policy, which enforces:
- **Major Class Balance**: Rejection rate must be representative.
- **Demographic Parity**: Approval rates for men and women must be within 10% delta.
- **Accuracy Calibration**: Minimum F1-score across all demographic groups.

---

## 📑 Resources
- [Full SDK Documentation](../../../packages/venturalitica-sdk/docs/README.md)
- [OSCAL Standard Overview](https://pages.nist.gov/OSCAL/)
