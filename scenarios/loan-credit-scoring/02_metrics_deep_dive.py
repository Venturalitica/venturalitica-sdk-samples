"""
🎯 ENHANCED EXAMPLE: Alternative Fairness & Privacy Metrics
================================================================================
⏱️  Time: 12 minutes
🎓 Complexity: ⭐⭐⭐ Advanced
🎯 Goal: Demonstrate Equalized Odds, Predictive Parity, and Privacy Metrics

What you'll learn:
- Beyond Demographic Parity: why Equalized Odds matters
- Predictive Parity: critical for high-stakes decisions (loans, hiring, health)
- Privacy metrics: k-anonymity, l-diversity, t-closeness, data minimization
- How to interpret semantic indicators (🟢🟡🔴)
- Why alternative metrics catch fairness problems DP misses
================================================================================
"""

import pandas as pd
import numpy as np
import venturalitica as vl
from venturalitica.metrics import (
    calc_demographic_parity, calc_equal_opportunity, calc_equalized_odds_ratio,
    calc_predictive_parity, calc_k_anonymity, calc_l_diversity, calc_t_closeness,
    calc_data_minimization_score, METRIC_REGISTRY, METRIC_METADATA
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path

print("🚀 Alternative Fairness & Privacy Metrics - Deep Dive\n")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n📊 Step 1: Loading German Credit Data...")
dataset_path = Path(__file__).parent.parent.parent / "datasets/loan/german_credit.csv"

if not dataset_path.exists():
    print(f"  ⚠️  Dataset not found. Run: python datasets/download_datasets.py --loan")
    exit(1)

df = pd.read_csv(dataset_path)
print(f"  ✓ Loaded {len(df)} loan applications")
print(f"  Columns: {list(df.columns)[:10]}...")

# Create derived columns expected by policy
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 100], 
                         labels=['Young', 'Adult', 'Senior'])
print(f"  ✓ Created age_group binned column")

# ============================================================================
# 2. EXPLORE METRIC REGISTRY
# ============================================================================
print("\n📋 Step 2: Exploring Metric Registry...")
print(f"  Available metrics: {len(METRIC_REGISTRY)}")

# Show FAIRNESS metrics
print("\n  🔍 Fairness Metrics:")
fairness_metrics = [m for m in METRIC_REGISTRY.keys() if 'parity' in m or 'odds' in m]
for metric in fairness_metrics[:5]:
    meta = METRIC_METADATA.get(metric, {})
    print(f"    • {metric}")
    print(f"      Description: {meta.get('description', 'N/A')}")
    print(f"      Category: {meta.get('category', 'N/A')}")

# Show PRIVACY metrics
print("\n  🔍 Privacy Metrics:")
privacy_metrics = [m for m in METRIC_REGISTRY.keys() if 'anonymity' in m or 'diversity' in m or 'closeness' in m or 'minimization' in m]
for metric in privacy_metrics[:4]:
    meta = METRIC_METADATA.get(metric, {})
    print(f"    • {metric}")
    print(f"      Description: {meta.get('description', 'N/A')}")

# ============================================================================
# 3. DEMONSTRATE FAIRNESS METRICS: Why Alternative Metrics Matter
# ============================================================================
print("\n" + "="*70)
print("🎯 Section 1: FAIRNESS METRICS - Why We Need Alternatives")
print("="*70)

# Prepare data for training
X = df.drop(columns=['target'])
print(f"  Encoding categorical columns: {[c for c in X.columns if X[c].dtype == 'object']}")
X = pd.get_dummies(X, drop_first=True)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train simple model
print("\n  Training model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create test dataframe with predictions (pulling original columns for readability)
test_df = df.loc[X_test.index].copy()
test_df['prediction'] = y_pred
# Note: German Credit dataset often encodes gender implicitly or as integers. 
# Ensuring mapping if needed, otherwise trusting original content.
if test_df['gender'].dtype != 'O':
     # Fallback mapping if it's 1/2 etc. Adjust based on actual data if known.
     # For german credit usually: status_sex column exists, but this script assumed 'gender'.
     # We keep as is if it's already usable.
     pass

print("\n📊 Fairness Metric Comparison:")
print("-" * 70)

metrics_to_compare = [
    ("demographic_parity_diff", "Demographic Parity Difference"),
    ("equal_opportunity_diff", "Equal Opportunity (TPR Parity)"),
    ("equalized_odds_ratio", "Equalized Odds (TPR + FPR)"),
    ("predictive_parity", "Predictive Parity (Precision Parity)"),
]

results = {}
for metric_key, metric_name in metrics_to_compare:
    try:
        func = METRIC_REGISTRY[metric_key]
        value = func(
            test_df,
            target='target',
            prediction='prediction',
            dimension='gender'
        )
        results[metric_key] = value
        
        # Get metadata
        meta = METRIC_METADATA.get(metric_key, {})
        ideal_value = meta.get('ideal_value', 'N/A')
        
        # Interpretation
        if value <= 0.10:
            emoji = "🟢"
            interpretation = "LOW RISK - Good fairness"
        elif value <= 0.20:
            emoji = "🟡"
            interpretation = "MEDIUM RISK - Acceptable but consider improvement"
        else:
            emoji = "🔴"
            interpretation = "HIGH RISK - Fairness violation detected"
        
        print(f"\n✓ {metric_name}")
        print(f"  Value: {value:.4f} (Ideal: {ideal_value})")
        print(f"  Risk: {emoji} {interpretation}")
        if meta.get('reference'):
            print(f"  Reference: {meta.get('reference')}")
    
    except Exception as e:
        print(f"\n✗ {metric_name}")
        print(f"  Error: {str(e)[:100]}")

# ============================================================================
# 4. DEMONSTRATE PRIVACY METRICS
# ============================================================================
print("\n" + "="*70)
print("🎯 Section 2: PRIVACY METRICS - Protecting Sensitive Data")
print("="*70)

print("\n📊 Privacy Metric Results:")
print("-" * 70)

# Prepare data for privacy metrics
privacy_cols = df[['age', 'gender', 'job']].copy()
privacy_cols['age_bin'] = pd.cut(privacy_cols['age'], bins=5)
privacy_cols_binned = privacy_cols[['age_bin', 'gender', 'job']]

privacy_metrics_to_test = [
    ("k_anonymity", "k-Anonymity", {'quasi_identifiers': ['age_bin', 'gender']}),
    ("l_diversity", "l-Diversity", {
        'quasi_identifiers': ['age_bin', 'gender'],
        'sensitive_attribute': 'job'
    }),
    ("data_minimization", "Data Minimization (GDPR)", {
        'sensitive_columns': ['age', 'job']
    }),
]

for metric_key, metric_name, kwargs in privacy_metrics_to_test:
    try:
        func = METRIC_REGISTRY.get(metric_key)
        if not func:
            print(f"\n⚠️  {metric_name} - Not yet in registry (will be added)")
            continue
        
        # For k-anonymity, use full dataframe
        if metric_key == 'k_anonymity':
            value = calc_k_anonymity(privacy_cols_binned, **kwargs)
        elif metric_key == 'l_diversity':
            value = calc_l_diversity(privacy_cols_binned, **kwargs)
        elif metric_key == 'data_minimization':
            value = calc_data_minimization_score(df, **kwargs)
        
        meta = METRIC_METADATA.get(metric_key, {})
        
        # Interpretation for privacy metrics (higher is usually better for k, lower for t-closeness)
        if metric_key == 'k_anonymity':
            if value >= 5:
                emoji, interp = "🟢", "GOOD - k≥5 is GDPR recommendation"
            elif value >= 2:
                emoji, interp = "🟡", "MEDIUM - Some privacy protection"
            else:
                emoji, interp = "🔴", "HIGH RISK - k<2 means individuals identifiable"
        elif metric_key == 'data_minimization':
            if value >= 0.70:
                emoji, interp = "🟢", "GOOD - Most columns are non-sensitive"
            elif value >= 0.50:
                emoji, interp = "🟡", "MEDIUM - Mixed sensitive/non-sensitive"
            else:
                emoji, interp = "🔴", "HIGH RISK - Too many sensitive columns collected"
        else:
            emoji, interp = "ℹ️ ", "Calculated"
        
        print(f"\n✓ {metric_name}")
        print(f"  Value: {value:.4f}")
        print(f"  Risk: {emoji} {interp}")
        if meta.get('reference'):
            print(f"  Reference: {meta.get('reference')}")
        if meta.get('paper_reference'):
            print(f"  Paper: {meta.get('paper_reference')}")
    
    except Exception as e:
        print(f"\n✗ {metric_name}")
        print(f"  Error: {str(e)[:150]}")

# ============================================================================
# 5. USE ENHANCED POLICY
# ============================================================================
print("\n" + "="*70)
print("🎯 Section 3: Running Full Governance Audit with Enhanced Policy")
print("="*70)

enhanced_policy = Path(__file__).parent.parent.parent / "policies/loan/governance-enhanced.oscal.yaml"

if enhanced_policy.exists():
    print(f"\n🔍 Running audit with enhanced policy: {enhanced_policy.name}")
    try:
        vl.enforce(data=df, policy=str(enhanced_policy))
        print("✓ Audit completed successfully!")
    except Exception as e:
        print(f"⚠️  Audit encountered issues: {str(e)[:100]}")
else:
    print(f"\n⚠️  Enhanced policy not found at {enhanced_policy}")

# ============================================================================
# 6. SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("📋 SUMMARY & KEY TAKEAWAYS")
print("="*70)

print("""
🔑 Key Insights:

1. DEMOGRAPHIC PARITY vs EQUALIZED ODDS:
   • Demographic Parity only checks if acceptance rates are equal
   • BUT doesn't check if errors are distributed fairly
   • Equalized Odds checks BOTH TPR and FPR parity → more rigorous

2. PREDICTIVE PARITY - Most Important for High-Stakes:
   • In lending: If we predict "approve", probability of profit should be
     equal across genders (precision parity)
   • In hiring: If we predict "qualified", probability of job success should
     be equal across genders
   • In healthcare: If we predict "diagnosis", probability of accuracy should
     be equal across racial groups

3. PRIVACY METRICS - Complementary to Fairness:
   • k-Anonymity: Can we re-identify individuals? (GDPR requires k≥5)
   • l-Diversity: Can we infer sensitive attributes? 
   • t-Closeness: Can we perform inference attacks?
   • Data Minimization: Are we collecting only necessary data?

4. 🎨 SEMANTIC INDICATORS:
   • 🟢 LOW RISK: Metric within acceptable threshold
   • 🟡 MEDIUM RISK: Metric marginally above threshold  
   • 🔴 HIGH RISK: Serious fairness/privacy violation

💡 Did you mean?
   If metrics fail, check:
   • Is 'gender' column correctly mapped? (Column existence)
   • Are categorical values consistent? (No 'M/F' vs '1/2' mismatch)
   • Is target truly binary (0/1)? (Some models return continuous predictions)
   • Are groups balanced? (Very small group sizes fail statistical tests)

🔗 Resources:
   • Fairlearn: https://fairlearn.org/
   • Paper on Equalized Odds: https://arxiv.org/abs/1610.02413
   • GDPR Data Protection: https://gdpr-info.eu/
   • Privacy Papers: https://en.wikipedia.org/wiki/Differential_privacy
""")

print("\n" + "="*70)
print("✅ Example completed!")
print("="*70)
