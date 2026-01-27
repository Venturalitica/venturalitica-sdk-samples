
import pandas as pd
import numpy as np
import venturalitica as vl
import os

def run_diagnostic():
    print("🔍 [1] Testing Imports...")
    try:
        from venturalitica.metrics import METRIC_REGISTRY
        print("   ✅ venturalitica.metrics imported successfully")
        if "precision_score" in METRIC_REGISTRY:
             print("   ✅ precision_score found in registry")
        if "predictive_equality_diff" not in METRIC_REGISTRY:
             print("   ✅ predictive_equality_diff correctly removed/absent")
        else:
             print("   ⚠️ predictive_equality_diff still in registry!")
    except ImportError as e:
        print(f"   ❌ ImportError: {e}")
        return

    print("\n🔍 [2] Generating Synthetic Multiclass Data...")
    # Generate random multiclass data (Age: 0-8)
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'target': np.random.randint(0, 9, n),     # 9 Age classes
        'prediction': np.random.randint(0, 9, n), # Random preds
        'age': np.random.choice(['20-29', '30-39', '40-49'], n), # Groups
        'race': np.random.choice(['White', 'Black', 'Asian'], n),
        'gender': np.random.choice(['Male', 'Female'], n)
    })
    print(f"   ✅ Data shape: {df.shape}")

    print("\n🔍 [3] Testing Age Policy (Multiclass Metrics)...")
    policy_path = "policies/vision/fairness_age.oscal.yaml"
    
    if not os.path.exists(policy_path):
        print(f"   ❌ Policy file not found: {policy_path}")
        return

    try:
        results = vl.enforce(
            data=df, 
            policy=policy_path,
            target='target', 
            prediction='prediction',
            age='age' # Bind age for fairness checks
        )
        
        print("\n   📊 Results Summary:")
        for r in results:
            status = "✅" if r.passed else "❌"
            val = f"{r.actual_value:.3f}" if isinstance(r.actual_value, (float, int)) else str(r.actual_value)
            print(f"   {status} {r.control_id:<25} Key={r.metric_key:<20} Val={val:<8} Threshold={r.operator} {r.threshold}")
            if hasattr(r, 'metadata') and r.metadata:
                 print(f"       📝 Meta: {r.metadata}")
            
    except Exception as e:
        print(f"   ❌ Enforcement Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnostic()
