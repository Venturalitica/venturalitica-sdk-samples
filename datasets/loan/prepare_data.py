"""
Prepare German Credit dataset with proper train/test/validation splits.

Creates:
- train.csv: Training data (60%)
- test.csv: Test data for model evaluation (20%)
- validation.csv: Validation data for fairness monitoring (20%)

For both V1 (biased) and V2 (debiased) versions.

Usage:
    uv run prepare_data.py

This is the main data preparation script for the demo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

DATA_DIR = Path(__file__).parent

# German Credit dataset column names
COLUMN_NAMES = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_status', 'employment', 'installment_rate', 'personal_status', 'other_parties',
    'residence_since', 'property_magnitude', 'age', 'other_payment_plans', 'housing',
    'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker', 'class'
]

def download_german_credit() -> pd.DataFrame:
    """Download German Credit dataset from UCI."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    print(f"Downloading German Credit dataset from UCI...")
    
    df = pd.read_csv(url, sep=' ', header=None, names=COLUMN_NAMES)
    print(f"Downloaded {len(df)} records")
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and engineer features for the demo."""
    result = df.copy()
    
    # Convert class to binary: 1=Good, 0=Bad (original is 1=Good, 2=Bad)
    result['credit_risk'] = (result['class'] == 1).astype(int)
    result = result.drop('class', axis=1)
    
    # Extract gender from personal_status
    result['gender'] = result['personal_status'].apply(
        lambda x: 'Female' if x in ['A92', 'A95'] else 'Male'
    )
    
    # Create age groups
    result['age_group'] = pd.cut(
        result['age'], 
        bins=[0, 25, 35, 45, 100],
        labels=['Young', 'Adult', 'Middle', 'Senior']
    )
    
    # Select relevant columns
    demo_cols = [
        'age', 'age_group', 'gender', 'credit_amount', 'duration',
        'employment', 'foreign_worker', 'credit_risk'
    ]
    
    return result[demo_cols]

def split_data(
    df: pd.DataFrame,
    train_size: float = 0.6,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/test/validation sets with stratification."""
    
    # First split: train vs (test + validation)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=df['credit_risk']
    )
    
    # Second split: test vs validation (50/50 of remaining)
    test_df, val_df = train_test_split(
        temp_df,
        test_size=val_size / (test_size + val_size),
        random_state=random_state,
        stratify=temp_df['credit_risk']
    )
    
    return train_df, test_df, val_df

def add_biased_predictions(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Add biased predictions simulating a flawed model.
    
    Bias characteristics:
    - Gender: Women get ~10% lower approval rate
    - Age: Young (<25) get ~15% lower approval rate
    """
    np.random.seed(seed)
    result = df.copy()
    
    # Base prediction from credit risk (add noise)
    base_prob = result['credit_risk'] * 0.7 + 0.15
    
    # Add gender bias: women penalized
    gender_bias = np.where(result['gender'] == 'Female', -0.12, 0.02)
    
    # Add age bias: young penalized
    age_bias = np.where(result['age'] < 25, -0.15, 0)
    age_bias = np.where(result['age'] > 45, 0.08, age_bias)
    
    # Calculate final prediction
    pred_prob = base_prob + gender_bias + age_bias + np.random.normal(0, 0.1, len(result))
    pred_prob = np.clip(pred_prob, 0, 1)
    
    result['prediction'] = (pred_prob > 0.5).astype(int)
    result['ground_truth'] = result['credit_risk']
    
    return result

def add_fair_predictions(df: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    """
    Add fair predictions simulating a debiased model.
    
    Bias characteristics:
    - Minimal gender disparity (<5%)
    - Reduced age bias (<10%)
    """
    np.random.seed(seed)
    result = df.copy()
    
    # Base prediction from credit risk (add noise)
    base_prob = result['credit_risk'] * 0.75 + 0.12
    
    # Minimal bias (not zero, to be realistic)
    gender_bias = np.where(result['gender'] == 'Female', -0.02, 0.01)
    age_bias = np.where(result['age'] < 25, -0.03, 0)
    
    # Calculate final prediction
    pred_prob = base_prob + gender_bias + age_bias + np.random.normal(0, 0.08, len(result))
    pred_prob = np.clip(pred_prob, 0, 1)
    
    result['prediction'] = (pred_prob > 0.5).astype(int)
    result['ground_truth'] = result['credit_risk']
    
    return result

def print_stats(df: pd.DataFrame, name: str):
    """Print dataset statistics."""
    print(f"\n{name}:")
    print(f"  Records: {len(df)}")
    print(f"  Gender: {df['gender'].value_counts().to_dict()}")
    print(f"  Approval rate: {df['prediction'].mean():.3f}")
    
    gender_rates = df.groupby('gender')['prediction'].mean()
    disparity = abs(gender_rates['Male'] - gender_rates['Female'])
    print(f"  Gender disparity: {disparity:.4f}")

def main():
    print("="*70)
    print(" German Credit Dataset Preparation")
    print(" Train / Test / Validation Splits with Bias Simulation")
    print("="*70)
    
    # Download raw data
    raw = download_german_credit()
    
    # Extract features
    df = extract_features(raw)
    print(f"\nFeatures extracted: {list(df.columns)}")
    
    # Split data
    train_df, test_df, val_df = split_data(df)
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.0f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(df)*100:.0f}%)")
    print(f"  Validation: {len(val_df)} ({len(val_df)/len(df)*100:.0f}%)")
    
    # Create output directories
    v1_dir = DATA_DIR / "v1"
    v2_dir = DATA_DIR / "v2"
    v1_dir.mkdir(exist_ok=True)
    v2_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # V1: Biased Version
    # =========================================================================
    print("\n" + "="*70)
    print(" V1: BIASED DATASET (Simulating flawed production model)")
    print("="*70)
    
    v1_train = add_biased_predictions(train_df, seed=42)
    v1_test = add_biased_predictions(test_df, seed=43)
    v1_val = add_biased_predictions(val_df, seed=44)
    
    print_stats(v1_train, "V1 Train")
    print_stats(v1_test, "V1 Test")
    print_stats(v1_val, "V1 Validation")
    
    v1_train.to_csv(v1_dir / "train.csv", index=False)
    v1_test.to_csv(v1_dir / "test.csv", index=False)
    v1_val.to_csv(v1_dir / "validation.csv", index=False)
    
    # Also save combined for backward compatibility
    v1_all = pd.concat([v1_train, v1_test, v1_val])
    v1_all.to_csv(DATA_DIR / "german_credit_v1.csv", index=False)
    
    # =========================================================================
    # V2: Debiased Version
    # =========================================================================
    print("\n" + "="*70)
    print(" V2: DEBIASED DATASET (Simulating fair retrained model)")
    print("="*70)
    
    v2_train = add_fair_predictions(train_df, seed=123)
    v2_test = add_fair_predictions(test_df, seed=124)
    v2_val = add_fair_predictions(val_df, seed=125)
    
    print_stats(v2_train, "V2 Train")
    print_stats(v2_test, "V2 Test")
    print_stats(v2_val, "V2 Validation")
    
    v2_train.to_csv(v2_dir / "train.csv", index=False)
    v2_test.to_csv(v2_dir / "test.csv", index=False)
    v2_val.to_csv(v2_dir / "validation.csv", index=False)
    
    # Also save combined for backward compatibility
    v2_all = pd.concat([v2_train, v2_test, v2_val])
    v2_all.to_csv(DATA_DIR / "german_credit_v2.csv", index=False)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print(" FILES CREATED")
    print("="*70)
    print(f"""
V1 (Biased):
  {v1_dir}/train.csv      ({len(v1_train)} records)
  {v1_dir}/test.csv       ({len(v1_test)} records)
  {v1_dir}/validation.csv ({len(v1_val)} records)
  {DATA_DIR}/german_credit_v1.csv (combined)

V2 (Debiased):
  {v2_dir}/train.csv      ({len(v2_train)} records)
  {v2_dir}/test.csv       ({len(v2_test)} records)
  {v2_dir}/validation.csv ({len(v2_val)} records)
  {DATA_DIR}/german_credit_v2.csv (combined)

Use for:
  - train.csv: Model training
  - test.csv: Model evaluation (accuracy, precision, recall)
  - validation.csv: Fairness monitoring / agent assessments
""")

if __name__ == "__main__":
    main()
