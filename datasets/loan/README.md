# Datasets

This directory contains training and evaluation datasets for the CreditScore ML demo.

## Files

| File | Description | Records | Bias Level |
|------|-------------|---------|------------|
| `german_credit_v1.csv` | Original dataset with biased predictions | 1000 | High |
| `german_credit_v2.csv` | Debiased dataset with fair predictions | 1000 | Low |
| `prepare_german_credit.py` | Data preparation script | - | - |

## Dataset: German Credit

**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

**Features**:
- `age`, `age_group` - Applicant age
- `gender` - Male/Female
- `credit_amount` - Loan amount (DM)
- `duration` - Loan duration (months)
- `employment` - Employment status
- `foreign_worker` - A201 (yes) / A202 (no)
- `credit_risk` - Ground truth (1=Good, 0=Bad)
- `prediction` - Model prediction
- `ground_truth` - Actual outcome

## Known Biases

| Protected Attribute | V1 Disparity | V2 Disparity | Threshold |
|---------------------|--------------|--------------|-----------|
| Gender | 10.4% | 7.5% | <10% |
| Age Group | ~20% | ~15% | <15% |

## Usage

```bash
# Regenerate datasets
cd training && uv run python ../datasets/prepare_german_credit.py
```

## Compliance

Full dataset documentation: `../compliance/data-sheets/GERMAN_CREDIT_DATA_SHEET.md`
