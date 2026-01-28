"""
🎯 VISION STEP 2: Multi-Task Model Training (Article 15)
================================================================================
Trains a ResNet34 with 3 heads (Gender, Race, Age) for robust feature learning.
Uses PyTorch Lightning + Venturalítica Callback.
Connects to MLflow (localhost:5000) and WandB.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback
import venturalitica as vl
import mlflow
import wandb
import numpy as np
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

# Config
load_dotenv()
torch.set_float32_matmul_precision('medium')

DATA_PATH = Path("datasets/vision/fairface_cache/metadata.csv")
POLICY_PATH = "policies/vision/fairness.oscal.yaml"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
SCALE = os.getenv("VL_DATA_SCALE", "POC")
BATCH_SIZE = 32 # Reduced from 128 to avoid OOM with Dual-Branch ResNet34
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "10"))
NUM_WORKERS = 15

# Label Mappings
GENDER_MAP = {'Male': 0, 'Female': 1}
RACE_MAP = {
    'White': 0, 'Black': 1, 'East Asian': 2, 'Indian': 3, 
    'Latino': 4, 'Middle Eastern': 5, 'Southeast Asian': 6
}

# ==============================================================================
# Model & Data
# ==============================================================================
class FairFaceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        face_img = Image.open(row['image_path']).convert('RGB')
        
        # MiVOLO Context: For now, use face as body proxy if no separate crops.
        # Ideally, we'd load person crops here.
        body_img = face_img 

        if self.transform:
            face_img = self.transform(face_img)
            body_img = self.transform(body_img)
        
        labels = {
            'gender': GENDER_MAP[row['gender']],
            'race': RACE_MAP[row['race']],
            'age': int(row['age']) 
        }
        
        meta = {
            'race_str': row['race'], 
            'gender_str': row['gender'], 
            'age_str': row['age'],
            'body_img': body_img
        }
        
        return face_img, labels, meta

from model_params import MultiTaskResNet

# ==============================================================================
# Auditing Callback
# ==============================================================================
class VenturaliticaGovernanceCallback(Callback):
    def __init__(self, policy_path: str):
        self.policy_path = policy_path
        self.outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Outputs is dict from val_step. Need to flatten race/gender/age strings if they are lists
        # But val_step returns meta lists.
        self.outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.outputs: return
        print(f"\n🛡️  [Governance Audit] Epoch {trainer.current_epoch}...")
        
        # Prepare Dataframes for each task
        # Helper to flatten lists or tensors
        def flat(key):
            # If tensor (preds/targets), use torch.cat
            first = self.outputs[0][key]
            if isinstance(first, torch.Tensor):
                return torch.cat([x[key] for x in self.outputs]).cpu().numpy()
            # If list (strings), extend
            else:
                out = []
                for x in self.outputs: out.extend(x[key])
                return out

        audit_gender = pd.DataFrame({
            'target': flat('targets_gender'),
            'prediction': flat('preds_gender'),
            'race': flat('race'), 
            'gender': flat('gender'),
            'age': flat('age_str')
        })

        audit_race = pd.DataFrame({
            'target': flat('targets_race'),
            'prediction': flat('preds_race'),
            'race': flat('race'),
            'gender': flat('gender'),
            'age': flat('age_str')
        })

        audit_age = pd.DataFrame({
            'target': flat('targets_age'),
            'prediction': flat('preds_age'),
            'race': flat('race'),
            'gender': flat('gender'),
            'age': flat('age_str')
        })

        # Debug Logging
        print(f"  > Auditing Gender: {audit_gender.shape} samples")
        # print(audit_gender.head(3))
        print(f"  > Auditing Race: {audit_race.shape} samples")
        # print(audit_race.head(3))
        print(f"  > Auditing Age: {audit_age.shape} samples")
        # print(audit_age.head(3))

        # Enforce Policies
        # Prepare Intersectional Columns (Race x Gender)
        # Note: We do this for one dataframe (e.g. audit_race or audit_gender) or all.
        # Since the policy is "Vision", we can check it on the "Race" dataframe (which contains race/gender info).
        
        # Helper to add column
        audit_race['intersectional_group'] = audit_race['race'].astype(str) + "_" + audit_race['gender'].astype(str)
        # Also clean strings to be safe (though they shouldn't be null)

        # Enforce Policies
        print(f"  > Enforcing Gender Policy ({len(audit_gender)})")
        results_g = vl.enforce(data=audit_gender, policy="policies/vision/fairness_gender.oscal.yaml", target='target', prediction='prediction', race='race', gender='gender', age='age')
        
        print(f"  > Enforcing Race Policy ({len(audit_race)})")
        results_r = vl.enforce(data=audit_race, policy="policies/vision/fairness_race.oscal.yaml", target='target', prediction='prediction', race='race', gender='gender', age='age')
        
        print(f"  > Enforcing Age Policy ({len(audit_age)})")
        results_a = vl.enforce(data=audit_age, policy="policies/vision/fairness_age.oscal.yaml", target='target', prediction='prediction', race='race', gender='gender', age='age')

        print(f"  > Enforcing Intersectional Policy ({len(audit_race)})")
        results_i = vl.enforce(
            data=audit_race, 
            policy="policies/vision/fairness_intersectional.oscal.yaml", 
            target='target', 
            prediction='prediction', 
            intersectional_group='intersectional_group',
            # New intersectional context for SDK v3
            intersectional_attrs={'Race': audit_race['race'], 'Gender': audit_race['gender']}
        )
        
        all_results = (results_g or []) + (results_r or []) + (results_a or []) + (results_i or [])

        if wandb.run and all_results:
            metrics = {getattr(r, 'metric_key', r.control_id): float(getattr(r, 'actual_value', 0)) for r in all_results}
            for r in all_results:
                metrics[f"gov.{r.control_id}_pass"] = 1 if r.passed else 0
            wandb.log(metrics)
        
        # 📊 Advanced Diagnostics: Confusion Matrices
        print("\n📊 Confusion Matrices (Ground Truth vs Prediction)")
        
        def print_cm(name, y_true, y_pred, labels=None):
            try:
                cm = confusion_matrix(y_true, y_pred)
                print(f"\n  > {name} Confusion Matrix:")
                
                # Simple ASCII Table
                if labels:
                    # Print Header
                    header = "      " + " ".join([f"{l[:4]:>5}" for l in labels])
                    print(header)
                
                for i, row in enumerate(cm):
                    label = labels[i][:4] if labels and i < len(labels) else f"{i}"
                    row_str = " ".join([f"{x:>5}" for x in row])
                    print(f"  {label:>4} [{row_str}]")
                    
                # Calculate per-class accuracy (diagonal / sum)
                diag = cm.diagonal()
                sums = cm.sum(axis=1)
                # Avoid div by zero
                accuracies = np.divide(diag, sums, out=np.zeros_like(diag, dtype=float), where=sums!=0)
                
                print(f"  > Per-class Reliability: {['{:.2f}'.format(x) for x in accuracies]}")
            except Exception as e:
                print(f"  ⚠️ Could not print CM for {name}: {e}")

        # Gender CM
        print_cm("GENDER", audit_gender['target'], audit_gender['prediction'], labels=['Male', 'Female'])
        
        # Race CM
        r_labels = list(RACE_MAP.keys()) # Ensure order is 0-6 matching map
        # Sort headers by index just to be safe
        r_sorted = sorted(RACE_MAP.items(), key=lambda x: x[1])
        r_names = [x[0] for x in r_sorted]
        print_cm("RACE", audit_race['target'], audit_race['prediction'], labels=r_names)
        
        # Age CM (0-8)
        # Age buckets: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
        age_labels = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
        print_cm("AGE", audit_age['target'], audit_age['prediction'], labels=age_labels)

        self.outputs.clear()

# ==============================================================================
# Main
# ==============================================================================
def train():
    print(f"🚀 Scaling Mode: {SCALE}")
    df = pd.read_csv(DATA_PATH)
    if SCALE == "POC":
        df = df.sample(n=min(2000, len(df)), random_state=42)
    elif SCALE == "MEDIUM":
        df = df.sample(n=min(10000, len(df)), random_state=42)
    
    # MiVOLO Augmentations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)
    ])
    
    # Stratified Split (Andrew's Tip #8 & #9)
    from sklearn.model_selection import train_test_split
    
    # Stratify by Race + Gender to ensure representation in val set
    df['stratify_col'] = df['race'] + "_" + df['gender']
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, 
        stratify=df['stratify_col']
    )
    
    print(f"✅ Data Split: Train={len(train_df)}, Val={len(val_df)} (Stratified)")
    
    # 📉 LDS Weights Calculation
    print("📊 Calculating LDS weights for Age Estimation...")
    age_counts = train_df['age'].value_counts().sort_index()
    # Handle missing ages in small subsets
    counts = np.zeros(9)
    for idx, count in age_counts.items():
        counts[int(idx)] = count
    
    import scipy.ndimage as ndimage
    lds_weights_raw = ndimage.gaussian_filter1d(counts.astype(float), sigma=1)
    lds_weights = 1.0 / (lds_weights_raw + 1e-6)
    lds_weights = torch.from_numpy(lds_weights / lds_weights.min()).float()
    print(f"  ✓ LDS Weights: {lds_weights.tolist()}")

    train_loader = DataLoader(
        FairFaceDataset(train_df, transform), batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        FairFaceDataset(val_df, transform), batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, pin_memory=True
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Vision-MultiTask-Fairness-V2")
    wandb.init(project="vision-fairness", name=f"multitask-lds-{SCALE.lower()}")

    from pytorch_lightning.callbacks import RichProgressBar
    
    # Data Validation Logic (Check if data is scaled properly)
    first_batch = next(iter(train_loader))
    images, labels, _ = first_batch
    print(f"\n🔬 [DATA DIAGNOSTIC] First Batch Stats:")
    print(f"   - Input Shape: {images.shape}")
    print(f"   - Pixel Mean: {images.mean().item():.3f} (Expected ~0.0 if normalized)")
    print(f"   - Pixel Std: {images.std().item():.3f} (Expected ~1.0 if normalized)")
    print(f"   - Target Samples: G={labels['gender'][:5]}, R={labels['race'][:5]}")

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS, 
        callbacks=[VenturaliticaGovernanceCallback(POLICY_PATH), RichProgressBar()], 
        logger=False
    )
    
    with vl.monitor("vision_training"):
        with mlflow.start_run(run_name=f"MultiTask (LDS + DualInput)"):
            trainer.fit(MultiTaskResNet(lds_weights=lds_weights), train_loader, val_loader)
            
    wandb.finish()

if __name__ == "__main__":
    train()
