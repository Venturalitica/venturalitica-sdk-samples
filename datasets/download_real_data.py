import pandas as pd
import numpy as np
import requests
from pathlib import Path
from sklearn.datasets import fetch_openml, load_breast_cancer

def setup_loan_data():
    print("Fetching German Credit data...")
    # Using OpenML for German Credit (statlog)
    german = fetch_openml(name='credit-g', version=1, as_frame=True)
    df = german.frame
    # Standardize column naming for governance
    df = df.rename(columns={'personal_status': 'gender'})
    # Convert gender to simpler male/female if possible
    df['gender'] = df['gender'].apply(lambda x: 'male' if 'male' in str(x).lower() else 'female')
    # target is 'class', map to 1/0
    df['target'] = df['class'].apply(lambda x: 1 if x == 'good' else 0)
    
    out_dir = Path('venturalitica-sdk-samples/datasets/loan')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'german_credit.csv', index=False)
    print(f"Saved to {out_dir}/german_credit.csv")

def setup_hiring_data():
    print("Fetching Adult Income data (Hiring Proxy)...")
    # Adult dataset is the gold standard for bias
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    df = adult.frame
    
    out_dir = Path('venturalitica-sdk-samples/datasets/hiring')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'adult_income.csv', index=False)
    print(f"Saved to {out_dir}/adult_income.csv")

def setup_health_data():
    print("Fetching Breast Cancer data...")
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    
    out_dir = Path('venturalitica-sdk-samples/datasets/health')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'breast_cancer.csv', index=False)
    print(f"Saved to {out_dir}/breast_cancer.csv")

if __name__ == '__main__':
    setup_loan_data()
    setup_hiring_data()
    setup_health_data()
