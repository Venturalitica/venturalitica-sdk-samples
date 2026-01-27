#!/usr/bin/env python3
"""
Dataset preparation script for Venturalitica SDK samples.

Downloads and prepares fairness benchmark datasets.
Supports both real-world data (OpenML) and synthetic alternatives.

Usage:
    python download_datasets.py --all              # Download all datasets
    python download_datasets.py --loan             # Download specific dataset
    python download_datasets.py --list             # List available datasets
    python download_datasets.py --loan --synthetic # Use synthetic data instead
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import libraries for real data
try:
    import openml
    HAS_OPENML = True
except ImportError:
    HAS_OPENML = False


# Dataset configurations
DATASETS = {
    'loan': {
        'name': 'German Credit',
        'real': {
            'source': 'openml',
            'dataset_id': 31,  # German Credit (OpenML)
            'target': 'target',
        },
        'synthetic': {
            'generator': 'balanced_classification',
            'n_samples': 1000,
            'n_features': 20,
        },
        'features': ['age', 'gender', 'credit_amount', 'duration', 'purpose'],
    },
    'hiring': {
        'name': 'Adult Income',
        'real': {
            'source': 'openml',
            'dataset_id': 179,  # Adult (OpenML)
            'target': 'class',
        },
        'synthetic': {
            'generator': 'balanced_classification',
            'n_samples': 48842,
            'n_features': 14,
        },
        'features': ['age', 'sex', 'education', 'hours_per_week'],
    },
    'health': {
        'name': 'Breast Cancer',
        'real': {
            'source': 'sklearn',
            'dataset_name': 'breast_cancer',
        },
        'synthetic': {
            'generator': 'balanced_classification',
            'n_samples': 569,
            'n_features': 30,
        },
        'features': ['mean_radius', 'mean_texture', 'age_group'],
    },
}


def download_real_data(dataset_key: str, output_dir: Path) -> pd.DataFrame:
    """Download real-world data from OpenML or sklearn."""
    config = DATASETS[dataset_key]
    real_config = config['real']
    source = real_config.get('source')
    
    print(f"📥 Downloading {config['name']} from {source}...")
    
    if source == 'openml' and HAS_OPENML:
        try:
            dataset_id = real_config['dataset_id']
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=real_config.get('target')
            )
            df = pd.DataFrame(X, columns=attribute_names)
            df[real_config.get('target')] = y
            print(f"✅ Loaded {len(df)} samples from OpenML")
            return df
        except Exception as e:
            print(f"⚠️  OpenML download failed: {e}")
            return None
    
    elif source == 'sklearn':
        try:
            from sklearn import datasets as sklearn_datasets
            dataset_name = real_config.get('dataset_name')
            if dataset_name == 'breast_cancer':
                data = sklearn_datasets.load_breast_cancer(as_frame=True)
                df = data.frame
                df['target'] = data.target
                print(f"✅ Loaded {len(df)} samples from sklearn")
                return df
        except Exception as e:
            print(f"⚠️  Sklearn download failed: {e}")
            return None
    
    return None


def generate_synthetic_data(dataset_key: str) -> pd.DataFrame:
    """Generate synthetic fairness benchmark data."""
    from sklearn.datasets import make_classification
    
    config = DATASETS[dataset_key]
    synth_config = config['synthetic']
    
    print(f"🎲 Generating synthetic {config['name']}...")
    
    if synth_config['generator'] == 'balanced_classification':
        X, y = make_classification(
            n_samples=synth_config['n_samples'],
            n_features=synth_config['n_features'],
            n_informative=int(synth_config['n_features'] * 0.7),
            n_redundant=int(synth_config['n_features'] * 0.2),
            weights=[0.7, 0.3],  # 70/30 class imbalance
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(synth_config['n_features'])])
        
        # Add protected attributes
        if dataset_key == 'loan':
            df['gender'] = np.random.choice(['M', 'F'], size=len(df), p=[0.6, 0.4])
            df['age'] = np.random.randint(20, 75, size=len(df))
            df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
        
        elif dataset_key == 'hiring':
            df['sex'] = np.random.choice(['Male', 'Female'], size=len(df), p=[0.6, 0.4])
            df['age'] = np.random.randint(18, 80, size=len(df))
        
        elif dataset_key == 'health':
            df['age_group'] = np.random.choice(['Young', 'Middle', 'Senior'], size=len(df))
        
        df['target'] = y
        df['prediction'] = np.random.binomial(1, 0.5, size=len(df))  # Random predictions
        
        print(f"✅ Generated {len(df)} synthetic samples")
        return df
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for Venturalitica SDK"
    )
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--loan', action='store_true', help='Download loan dataset')
    parser.add_argument('--hiring', action='store_true', help='Download hiring dataset')
    parser.add_argument('--health', action='store_true', help='Download health dataset')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data (fallback)')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--output', type=Path, default=Path('./datasets'), 
                       help='Output directory for datasets')
    
    args = parser.parse_args()
    
    # Show available datasets
    if args.list:
        print("\n📊 Available Datasets:\n")
        for key, config in DATASETS.items():
            print(f"  {key:10} - {config['name']}")
            if config['real']:
                print(f"             Real: {config['real'].get('source', 'Unknown')}")
            print(f"             Synthetic: {config['synthetic'].get('generator', 'N/A')}")
        print()
        return
    
    # Determine which datasets to download
    to_download = []
    if args.all:
        to_download = list(DATASETS.keys())
    else:
        if args.loan: to_download.append('loan')
        if args.hiring: to_download.append('hiring')
        if args.health: to_download.append('health')
    
    if not to_download:
        print("❌ No datasets specified. Use --all or --loan/--hiring/--health")
        sys.exit(1)
    
    # Create output directories
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each dataset
    for dataset_key in to_download:
        dataset_name = DATASETS[dataset_key]['name']
        
        # Try real data first
        if not args.synthetic:
            df = download_real_data(dataset_key, output_dir)
            if df is not None:
                # Save to CSV
                dataset_subdir = output_dir / dataset_key
                dataset_subdir.mkdir(exist_ok=True)
                output_file = dataset_subdir / f'{dataset_key}.csv'
                df.to_csv(output_file, index=False)
                print(f"💾 Saved to {output_file}\n")
                continue
        
        # Fall back to synthetic
        df = generate_synthetic_data(dataset_key)
        if df is not None:
            dataset_subdir = output_dir / dataset_key
            dataset_subdir.mkdir(exist_ok=True)
            output_file = dataset_subdir / f'{dataset_key}.csv'
            df.to_csv(output_file, index=False)
            print(f"💾 Saved synthetic data to {output_file}\n")
        else:
            print(f"❌ Failed to download or generate {dataset_name}\n")
    
    print("✅ Dataset preparation complete!")
    print(f"📁 Datasets saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
