"""
Download real datasets for Venturalitica SDK samples.

This script downloads datasets from public sources (OpenML, UCI, scikit-learn).
Data is NOT versioned with Git LFS - instead it's downloaded on first run and cached locally.

Usage:
    python download_real_data.py                # Download all datasets
    python download_real_data.py --loan         # Download only loan data
    python download_real_data.py --surgery      # Download TCIA medical metadata
    python download_real_data.py --list         # List available datasets
"""

import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml, load_breast_cancer
import argparse
import sys


def setup_loan_data(force=False):
    """Download German Credit dataset from OpenML"""
    out_path = Path(__file__).parent / "loan" / "german_credit.csv"
    
    if out_path.exists() and not force:
        print(f"✓ Loan data already exists: {out_path}")
        return True
    
    try:
        print("⏳ Fetching German Credit data from OpenML...")
        german = fetch_openml(name='credit-g', version=1, as_frame=True)
        df = german.frame
        
        # Standardize columns for governance
        df = df.rename(columns={'personal_status': 'gender'})
        df['gender'] = df['gender'].apply(lambda x: 'male' if 'male' in str(x).lower() else 'female')
        df['target'] = df['class'].apply(lambda x: 1 if x == 'good' else 0)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"✓ Saved: {out_path} ({len(df)} rows)")
        return True
    except Exception as e:
        print(f"✗ Error downloading loan data: {e}")
        return False


def setup_hiring_data(force=False):
    """Download Adult Income dataset from OpenML"""
    out_path = Path(__file__).parent / "hiring" / "adult_income.csv"
    
    if out_path.exists() and not force:
        print(f"✓ Hiring data already exists: {out_path}")
        return True
    
    try:
        print("⏳ Fetching Adult Income data from OpenML...")
        adult = fetch_openml(name='adult', version=2, as_frame=True)
        df = adult.frame
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"✓ Saved: {out_path} ({len(df)} rows)")
        return True
    except Exception as e:
        print(f"✗ Error downloading hiring data: {e}")
        return False


def setup_health_data(force=False):
    """Download Breast Cancer dataset from scikit-learn"""
    out_path = Path(__file__).parent / "health" / "breast_cancer.csv"
    
    if out_path.exists() and not force:
        print(f"✓ Health data already exists: {out_path}")
        return True
    
    try:
        print("⏳ Fetching Breast Cancer data from scikit-learn...")
        cancer = load_breast_cancer(as_frame=True)
        df = cancer.frame
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"✓ Saved: {out_path} ({len(df)} rows)")
        return True
    except Exception as e:
        print(f"✗ Error downloading health data: {e}")
        return False


def setup_surgery_data(force=False):
    """Download TCIA metadata using nbiatoolkit (Guest access)"""
    out_path = Path(__file__).parent / "surgery" / "tcia_metadata.csv"
    
    if out_path.exists() and not force:
        print(f"✓ Surgery data already exists: {out_path}")
        return True
    
    try:
        from nbiatoolkit import NBIAClient
        print("⏳ Fetching TCIA metadata (Spine-Mets-CT-SEG collection)...")
        
        with NBIAClient() as client:
            # Fetch series metadata
            series_data = client.getSeries(Collection='Spine-Mets-CT-SEG')
            if not series_data:
                print("✗ No data returned from TCIA API.")
                return False
            
            df = pd.DataFrame(series_data)
            
            # Fetch patient metadata for demographics
            patients_data = client.getPatients(Collection='Spine-Mets-CT-SEG')
            patients_df = pd.DataFrame(patients_data)
            
            # Merge demographics
            if 'PatientId' in patients_df.columns:
                patients_df = patients_df.rename(columns={'PatientId': 'PatientID'})
            
            if 'PatientID' in df.columns and 'PatientID' in patients_df.columns:
                df = df.merge(patients_df[['PatientID', 'PatientSex']], on='PatientID', how='left')
            
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"✓ Saved: {out_path} ({len(df)} series)")
            return True
            
    except Exception as e:
        print(f"✗ Error downloading surgery data: {e}")
        return False


def list_datasets():
    """List available datasets and their status"""
    datasets = {
        "loan": Path(__file__).parent / "loan" / "german_credit.csv",
        "hiring": Path(__file__).parent / "hiring" / "adult_income.csv",
        "health": Path(__file__).parent / "health" / "breast_cancer.csv",
        "surgery": Path(__file__).parent / "surgery" / "tcia_metadata.csv",
    }
    
    print("\nAvailable Datasets:\n")
    for name, path in datasets.items():
        status = "✓" if path.exists() else "✗"
        size = f"({path.stat().st_size / 1024:.1f} KB)" if path.exists() else "(not downloaded)"
        print(f"  {status} {name:10s} {size}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download datasets for samples")
    parser.add_argument('--loan', action='store_true', help="Download only loan data")
    parser.add_argument('--hiring', action='store_true', help="Download only hiring data")
    parser.add_argument('--health', action='store_true', help="Download only health data")
    parser.add_argument('--surgery', action='store_true', help="Download only surgery (TCIA) data")
    parser.add_argument('--list', action='store_true', help="List all datasets")
    parser.add_argument('--force', action='store_true', help="Re-download even if exists")
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        sys.exit(0)
    
    # If specific datasets chosen, download those; otherwise download all
    download_all = not (args.loan or args.hiring or args.health or args.surgery)
    
    success = True
    if args.loan or download_all:
        success &= setup_loan_data(force=args.force)
    if args.hiring or download_all:
        success &= setup_hiring_data(force=args.force)
    if args.health or download_all:
        success &= setup_health_data(force=args.force)
    if args.surgery or download_all:
        success &= setup_surgery_data(force=args.force)
    
    print()
    if success:
        print("✓ All datasets ready!")
        sys.exit(0)
    else:
        print("✗ Some datasets failed to download. Check your internet connection.")
        sys.exit(1)
