"""
Download German Credit dataset for loan scenario (no local fallback, no splits).
Adds minimal derived columns (gender, target) to keep scenario scripts compatible.
Saves to datasets/loan/german_credit.csv. If download fails, exits with error.
Usage: uv run prepare_data.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

DATA_PATH = Path(__file__).parent.parent.parent / "datasets/loan/german_credit.csv"

# Column names mapping for German Credit dataset
COLUMN_NAMES = [
    "checking_status",      # Attribute1
    "duration",             # Attribute2
    "credit_history",       # Attribute3
    "purpose",              # Attribute4
    "credit_amount",        # Attribute5
    "savings_status",       # Attribute6
    "employment",           # Attribute7
    "installment_commitment",  # Attribute8
    "personal_status_sex",  # Attribute9
    "other_parties",        # Attribute10
    "residence_since",      # Attribute11
    "property_magnitude",   # Attribute12
    "age",                  # Attribute13 ✅ NOW NUMERIC
    "other_payment_plans",  # Attribute14
    "housing",              # Attribute15
    "existing_credits",     # Attribute16
    "job",                  # Attribute17
    "num_dependents",       # Attribute18
    "own_telephone",        # Attribute19
    "foreign_worker",       # Attribute20
]

def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("📥 Downloading German Credit dataset from UCI ML Repository...")
    print("   Source: archive.ics.uci.edu (Dataset ID: 144)")
    print("   Expected size: ~31 KB (1000 rows)")
    
    try:
        # ✅ Fetch using ucimlrepo API (more reliable than direct file download)
        statlog_german_credit = fetch_ucirepo(id=144)
        
        # Extract features and targets
        X = statlog_german_credit.data.features
        y = statlog_german_credit.data.targets
        
        # Combine into single dataframe
        df = pd.concat([X, y], axis=1)
        
        # ✅ Map generic column names to meaningful names
        attribute_cols = [col for col in df.columns if col.startswith('Attribute')]
        if len(attribute_cols) == len(COLUMN_NAMES):
            col_mapping = {old: new for old, new in zip(attribute_cols, COLUMN_NAMES)}
            df = df.rename(columns=col_mapping)
        
        print(f"  ✓ Loaded dataset with shape: {df.shape}")
    except Exception as exc:
        print(f"\n❌ Download failed: {exc}")
        print("\n🔍 Troubleshooting:")
        print("   1. Check internet connection")
        print("   2. Verify UCI repository is accessible")
        print("   3. Try again: uv run prepare_data.py")
        raise SystemExit(1)

    # ✅ Ensure 'age' column exists and is numeric
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        print(f"  ✓ Age column mapped successfully (numeric)")
    else:
        print(f"  ⚠️  Warning: 'age' column not found. Available columns: {df.columns.tolist()}")
    
    # Minimal derived columns expected by scenarios
    if 'class' in df.columns:  # German Credit uses 'class' as target (1=good, 2=bad)
        df['target'] = (df['class'] == 1).astype(int)  # Convert to binary: 1=good, 0=bad
        print(f"  ✓ Target column created from 'class' (1=good, 0=bad)")
    else:
        raise SystemExit("Error: 'class' column missing in downloaded data. Cannot proceed without real labels.")
    
    # Create gender column from personal_status
    if 'personal_status_sex' in df.columns:
        df["gender"] = df['personal_status_sex'].apply(
            lambda x: "female" if str(x) in ["A92", "A95"] else "male"
        )
        print(f"  ✓ Gender column created from personal_status_sex")
    else:
        raise SystemExit("Error: 'personal_status_sex' column missing in downloaded data. Cannot proceed.")

    df.to_csv(DATA_PATH, index=False)

    print("\n✅ Download complete!")
    print(f"   Location: {DATA_PATH}")
    print(f"   Rows: {len(df)} loan applications")
    print(f"   Key columns: {['age', 'target', 'gender', 'duration', 'credit_amount']}")
    print("\n🚀 Next step: uv run 01_governance_audit.py")


if __name__ == "__main__":
    main()
