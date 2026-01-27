"""
🎯 VISION STEP 3: Advanced Metrics & Drill-down
================================================================================
Demonstrates manual auditing for complex metrics like Predictive Parity.
Analyzes how precision/recall varies across intersections of protected groups.
"""

import os
import pandas as pd
import venturalitica as vl
from pathlib import Path

# Config
DATA_PATH = Path("datasets/vision/fairface_cache/metadata.csv")
POLICY_PATH = "policies/vision/fairness.oscal.yaml"
CHECKPOINT_PATH = "checkpoints/epoch=9-step=10850.ckpt"

from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
from datasets import Dataset

from model_params import MultiTaskResNet

def load_real_model():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    
    # Load with explicit class, mapping directly to CPU to avoid device mismatch
    # (since training was likely on GPU but this script runs on CPU usually)
    model = MultiTaskResNet.load_from_checkpoint(CHECKPOINT_PATH, map_location=torch.device('cpu'))
    model.to('cpu')
    model.eval()
    return model

def analyze():
    print("🔍 Deep-Dive: Analyzing Predictive Parity on REAL Trained Model...")
    
    if not DATA_PATH.exists():
        print("⚠️  Data missing. Run 'uv run prepare_data.py'")
        return

    # 1. Load Real Validation Data
    df = pd.read_csv(DATA_PATH)
    
    # Same stratified split logic to get the VALIDATION set
    from sklearn.model_selection import train_test_split
    df['stratify_col'] = df['race'] + "_" + df['gender']
    _, val_df = train_test_split(
        df, test_size=0.2, random_state=42, 
        stratify=df['stratify_col']
    )
    
    # Use a smaller sample for quick analysis if needed, but 20% of 86k is ~17k.
    # Let's verify on 2000 samples for the script speed.
    val_sample = val_df.sample(2000, random_state=42)
    
    print(f"  📂 Loaded {len(val_sample)} validation samples.")

    # 2. Run Inference
    print("  🤖 Running inference (ResNet-34)...")
    model = load_real_model()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, df, transform):
            self.df = df
            self.transform = transform
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row['image_path']).convert('RGB')
            return self.transform(img)

    ds = InferenceDataset(val_sample, transform)
    dl = DataLoader(ds, batch_size=32, num_workers=4)
    
    all_preds_gender = []
    
    with torch.no_grad():
        from tqdm import tqdm
        for batch in tqdm(dl, desc="Inferencing"):
            # Forward pass
            outputs = model(batch)
            out_gender = outputs['gender']
            
            # Get gender predictions (1=Male, 0=Female)
            preds = torch.argmax(out_gender, dim=1).cpu().numpy()
            all_preds_gender.extend(preds)
            
    # Add predictions to DataFrame
    val_sample['prediction'] = all_preds_gender
    
    # Map target: Female=0, Male=1 (Check 02_model_training.py mapping!)
    # In 02_model_training.py: {'Female': 0, 'Male': 1}
    val_sample['target'] = val_sample['gender'].map({'Female': 0, 'Male': 1})
    val_sample['prediction_label'] = val_sample['prediction'].map({0: 'Female', 1: 'Male'})
    
    # 3. Manual Audit
    results = vl.enforce(
        data=val_sample,
        policy=POLICY_PATH,
        target='target',
        prediction='prediction',
        race='race',
        gender='gender'
    )
    
    print("\n📊 Advanced Audit Results (Real Data):")
    print("-" * 80)
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        val = f"{float(r.actual_value):.3f}" if r.actual_value is not None else "N/A"
        print(f"{r.control_id:<35} | {val:<10} | {status}")
    print("-" * 80)

    # 4. Advanced Metrics Calculation
    print("\n⚖️  Fairlearn Style Metrics (Gender)")
    
    def get_group_metrics(group_df):
        # Calculate confusion matrix components
        y_true = group_df['target']
        y_pred = group_df['prediction']
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        # Avoid division by zero
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall / Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity
        
        return pd.Series({'TPR': tpr, 'FPR': fpr, 'TNR': tnr, 'Count': len(group_df)})

    gender_metrics = val_sample.groupby('gender').apply(get_group_metrics)
    print("\nPer-Group Metrics:")
    print(gender_metrics)
    
    # Calculate Differences
    eq_opp = gender_metrics['TPR'].max() - gender_metrics['TPR'].min()
    pred_eq = gender_metrics['FPR'].max() - gender_metrics['FPR'].min()
    
    print(f"\n📊 Equal Opportunity (TPR Diff): {eq_opp:.3f}")
    print(f"📊 Predictive Equality (FPR Diff): {pred_eq:.3f}")
    print(f"📊 Equalized Odds (Max Diff):    {max(eq_opp, pred_eq):.3f}")

    if max(eq_opp, pred_eq) > 0.10:
        print("\n❌  EQUALIZED ODDS VIOLATION DETECTED")
        print("    The model behaves significantly differently for Male vs Female.")
    else:
        print("\n✅  EQUALIZED ODDS COMPLIANT")

if __name__ == "__main__":
    analyze()
