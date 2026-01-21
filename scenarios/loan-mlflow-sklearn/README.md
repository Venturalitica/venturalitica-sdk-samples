# Model Training

This directory contains training scripts, model artifacts, and experiment tracking.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Start MLflow (optional, for experiment tracking)
docker compose -f ../../docker-compose.dev.yml up mlflow -d

# 3. Train models
uv run train_models.py

# 4. Upload to S3 (for FairGage agent)
docker compose -f ../../docker-compose.dev.yml up localstack -d
uv run upload_to_s3.py
```

## Files

| File | Description |
|------|-------------|
| `train_models.py` | Train V1 (biased) and V2 (fair) models |
| `upload_to_s3.py` | Upload datasets to LocalStack S3 |
| `pyproject.toml` | Python dependencies |
| `model_v1.joblib` | V1 model artifact (generated) |
| `model_v2.joblib` | V2 model artifact (generated) |
| `scaler_v*.joblib` | Feature scalers (generated) |
| `fairness_report_v*.json` | Fairness audit reports (generated) |

## Models

### V1: CreditScore ML (Standard)
- Algorithm: Logistic Regression
- Fairness: Not considered
- Gender Disparity: ~10.4% (FAILS threshold)
- Status: ❌ Not suitable for production

### V2: CreditScore ML (Fair)
- Algorithm: Logistic Regression with sample weighting
- Fairness: Weighted training for underrepresented groups
- Gender Disparity: ~7.5% (PASSES threshold)
- Status: ✅ Approved for production

## MLflow Integration

Models are logged to MLflow with:
- Performance metrics (accuracy, precision, recall, F1)
- Fairness metrics (demographic parity, equal opportunity)
- Model signature for validation
- Scalers and fairness reports as artifacts

Access MLflow UI: http://localhost:5000

## Compliance

Full model documentation: `../compliance/model-cards/CREDITSCORE_ML_MODEL_CARD.md`
