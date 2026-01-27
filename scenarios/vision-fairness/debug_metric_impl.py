
import pandas as pd
import numpy as np
import sys
import os

# Ensure we import from the local package source if possible, or installed venv
# sys.path.append(os.path.abspath("../../../packages/venturalitica-sdk/src"))

try:
    from venturalitica.fairness.multiclass import calc_predictive_parity_multiclass
except ImportError:
    print("❌ Could not import calc_predictive_parity_multiclass. checking path...")
    try:
        from venturalitica.fairness import calc_predictive_parity
        print("⚠️ Using wrapper function from fairness/__init__")
    except ImportError:
         print("❌ Could not import ANY metric function. Check install.")
         sys.exit(1)

def test_predictive_parity():
    print("\n🔬 Testing Predictive Parity (Precision Parity) Logic")
    print("===================================================")

    # Scenarion 1: Perfect Fairness
    # Class 0: Precision 0.8 for both Group A and B
    data_fair = pd.DataFrame([
        # Group A
        {'target': 0, 'pred': 0, 'dim': 'A'}, {'target': 0, 'pred': 0, 'dim': 'A'}, 
        {'target': 0, 'pred': 0, 'dim': 'A'}, {'target': 0, 'pred': 0, 'dim': 'A'},
        {'target': 1, 'pred': 0, 'dim': 'A'}, # FP for class 0
        # Group B
        {'target': 0, 'pred': 0, 'dim': 'B'}, {'target': 0, 'pred': 0, 'dim': 'B'}, 
        {'target': 0, 'pred': 0, 'dim': 'B'}, {'target': 0, 'pred': 0, 'dim': 'B'}, 
        {'target': 1, 'pred': 0, 'dim': 'B'}, # FP for class 0
        # Dummy Class 1 rows to satisfy min_classes=2
        {'target': 1, 'pred': 1, 'dim': 'A'}, {'target': 1, 'pred': 1, 'dim': 'B'},
    ] * 5)

    print("\n[Case 1] Controlled Perfect Fairness (Class 0 only populated)")
    res = calc_predictive_parity_multiclass(
        data_fair['target'], data_fair['pred'], data_fair['dim'], strategy='macro'
    )
    print(f"   Expected: 0.0")
    print(f"   Actual:   {res}")
    
    # Scenario 2: High Disparity
    # Class 0: Group A (100% Precision), Group B (0% Precision)
    data_biased = pd.DataFrame([
        # Group A: 2 preds for 0, both Correct (Precision=1.0)
        {'target': 0, 'pred': 0, 'dim': 'A'}, 
        {'target': 0, 'pred': 0, 'dim': 'A'},
        # Group B: 2 preds for 0, both Wrong (Target=1) (Precision=0.0)
        {'target': 1, 'pred': 0, 'dim': 'B'},
        {'target': 1, 'pred': 0, 'dim': 'B'},
        # Dummy Class 1
        {'target': 1, 'pred': 1, 'dim': 'A'}, {'target': 1, 'pred': 1, 'dim': 'B'},
    ] * 10)
    
    print("\n[Case 2] Extreme Disparity (GrA=1.0, GrB=0.0)")
    res = calc_predictive_parity_multiclass(
        data_biased['target'], data_biased['pred'], data_biased['dim'], strategy='macro'
    )
    print(f"   Expected: 1.0")
    print(f"   Actual:   {res}")

    print("\n[Case 3] Multiclass Max Disparity Logic check")
    rows = []
    # Cls 0
    # Group A: 1 TP, 0 FP -> P=1.0
    rows.append({'target': 0, 'pred': 0, 'dim': 'A'})
    # Group B: 4 TP, 1 FP -> P=0.8
    for _ in range(4): rows.append({'target': 0, 'pred': 0, 'dim': 'B'})
    rows.append({'target': 1, 'pred': 0, 'dim': 'B'})
    
    # Cls 1
    # Group A: 1 TP, 1 FP -> P=0.5
    rows.append({'target': 1, 'pred': 1, 'dim': 'A'})
    rows.append({'target': 0, 'pred': 1, 'dim': 'A'})
    # Group B: 9 TP, 1 FP -> P=0.9
    for _ in range(9): rows.append({'target': 1, 'pred': 1, 'dim': 'B'})
    rows.append({'target': 0, 'pred': 1, 'dim': 'B'})
    
    df_multi = pd.DataFrame(rows * 3) # Repeat 3 times (22 rows * 3 = 66)
    
    res = calc_predictive_parity_multiclass(
        df_multi['target'], df_multi['pred'], df_multi['dim'], strategy='macro'
    )
    print(f"   Calculated: {res}")
    print(f"   Note: Check if this is Mean(0.2, 0.4)=0.3 or Max(0.2, 0.4)=0.4?")

if __name__ == "__main__":
    test_predictive_parity()
