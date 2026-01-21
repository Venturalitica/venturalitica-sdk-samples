"""
Download and prepare German Credit dataset for the MVP demo.

The German Credit dataset is a classic dataset for credit risk assessment,
containing 1000 records with known fairness issues around age and gender.

Usage:
    python prepare_german_credit.py

This creates:
    - german_credit_v1.csv (original, biased)
    - german_credit_v2.csv (rebalanced, less biased)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent

# German Credit dataset column names
COLUMN_NAMES = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_status', 'employment', 'installment_rate', 'personal_status', 'other_parties',
    'residence_since', 'property_magnitude', 'age', 'other_payment_plans', 'housing',
    'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker', 'class'
]

def download_german_credit():
    """Download German Credit dataset from UCI."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    print(f"Downloading German Credit dataset from UCI...")
    
    df = pd.read_csv(url, sep=' ', header=None, names=COLUMN_NAMES)
    print(f"Downloaded {len(df)} records")
    return df

def prepare_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare German Credit v1 - the biased version.
    
    Known issues in this dataset:
    - Age bias: Older applicants tend to get better ratings
    - Gender bias: Hidden in 'personal_status' column (A91=male divorced, A92=female, etc.)
    - Foreign worker bias: Foreign workers penalized
    """
    v1 = df.copy()
    
    # Convert class to binary: 1=Good, 0=Bad (original is 1=Good, 2=Bad)
    v1['credit_risk'] = (v1['class'] == 1).astype(int)
    v1 = v1.drop('class', axis=1)
    
    # Extract gender from personal_status
    # A91: male divorced/separated
    # A92: female divorced/separated/married
    # A93: male single
    # A94: male married/widowed
    # A95: female single
    v1['gender'] = v1['personal_status'].apply(
        lambda x: 'Female' if x in ['A92', 'A95'] else 'Male'
    )
    
    # Create age groups for fairness analysis
    v1['age_group'] = pd.cut(
        v1['age'], 
        bins=[0, 25, 35, 45, 100],
        labels=['Young (<=25)', 'Adult (26-35)', 'Middle (36-45)', 'Senior (>45)']
    )
    
    # Simulate a biased model prediction
    # This mimics real-world bias where young people and women get worse predictions
    np.random.seed(42)
    
    # Base prediction from credit risk (add some noise)
    base_prob = v1['credit_risk'] * 0.7 + 0.15
    
    # Add age bias: young people get penalized
    age_bias = np.where(v1['age'] < 25, -0.15, 0)
    age_bias = np.where(v1['age'] > 45, 0.1, age_bias)
    
    # Add gender bias: women get slightly penalized
    gender_bias = np.where(v1['gender'] == 'Female', -0.08, 0.02)
    
    # Add foreign worker bias
    foreign_bias = np.where(v1['foreign_worker'] == 'A201', -0.05, 0)
    
    # Calculate final prediction probability
    pred_prob = base_prob + age_bias + gender_bias + foreign_bias + np.random.normal(0, 0.1, len(v1))
    pred_prob = np.clip(pred_prob, 0, 1)
    
    v1['prediction'] = (pred_prob > 0.5).astype(int)
    v1['ground_truth'] = v1['credit_risk']
    
    # Select relevant columns for demo
    demo_cols = [
        'age', 'age_group', 'gender', 'credit_amount', 'duration',
        'employment', 'foreign_worker', 'credit_risk', 'prediction', 'ground_truth'
    ]
    v1 = v1[demo_cols]
    
    print(f"\nV1 prepared: {len(v1)} records")
    print(f"  Gender distribution: {v1['gender'].value_counts().to_dict()}")
    print(f"  Age group distribution: {v1['age_group'].value_counts().to_dict()}")
    print(f"\n  Approval rate by gender:")
    print(f"    {v1.groupby('gender')['prediction'].mean().to_dict()}")
    print(f"\n  Approval rate by age group:")
    print(f"    {v1.groupby('age_group')['prediction'].mean().to_dict()}")
    
    # Calculate demographic parity difference
    gender_rates = v1.groupby('gender')['prediction'].mean()
    gender_disparity = abs(gender_rates['Male'] - gender_rates['Female'])
    print(f"\n  Gender disparity (demographic parity): {gender_disparity:.3f}")
    
    return v1

def prepare_v2(v1: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare German Credit v2 - a debiased version.
    
    Simulates the result of fairness-aware retraining:
    - Reduced gender bias
    - Reduced age bias
    - More equitable predictions
    """
    v2 = v1.copy()
    
    np.random.seed(123)
    
    # Recalculate predictions with reduced bias
    base_prob = v2['ground_truth'] * 0.75 + 0.12
    
    # Minimal remaining bias (can't completely eliminate)
    age_bias = np.where(v2['age'] < 25, -0.03, 0)
    gender_bias = np.where(v2['gender'] == 'Female', -0.02, 0.01)
    
    pred_prob = base_prob + age_bias + gender_bias + np.random.normal(0, 0.08, len(v2))
    pred_prob = np.clip(pred_prob, 0, 1)
    
    v2['prediction'] = (pred_prob > 0.5).astype(int)
    
    print(f"\nV2 prepared: {len(v2)} records")
    print(f"\n  Approval rate by gender:")
    print(f"    {v2.groupby('gender')['prediction'].mean().to_dict()}")
    print(f"\n  Approval rate by age group:")
    print(f"    {v2.groupby('age_group')['prediction'].mean().to_dict()}")
    
    # Calculate demographic parity difference
    gender_rates = v2.groupby('gender')['prediction'].mean()
    gender_disparity = abs(gender_rates['Male'] - gender_rates['Female'])
    print(f"\n  Gender disparity (demographic parity): {gender_disparity:.3f}")
    
    return v2

def main():
    # Download
    raw = download_german_credit()
    
    # Prepare versions
    v1 = prepare_v1(raw)
    v2 = prepare_v2(v1)
    
    # Save
    v1_path = DATA_DIR / "german_credit_v1.csv"
    v2_path = DATA_DIR / "german_credit_v2.csv"
    
    v1.to_csv(v1_path, index=False)
    v2.to_csv(v2_path, index=False)
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Saved: {v1_path}")
    print(f"Saved: {v2_path}")
    print(f"\nV1 has significant gender bias ({abs(v1.groupby('gender')['prediction'].mean().diff().iloc[-1]):.3f})")
    print(f"V2 has reduced bias ({abs(v2.groupby('gender')['prediction'].mean().diff().iloc[-1]):.3f})")
    print(f"\nThreshold for demo: 0.10 (demographic parity)")
    print(f"V1 should FAIL, V2 should PASS")

if __name__ == "__main__":
    main()
