import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import wandb
import venturalitica
import argparse
import uuid

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "datasets/hiring"
DATA_FILE = DATA_DIR / "adult_income.csv"
HIRING_POLICY_PATH = Path(__file__).parent.parent.parent / "policies/hiring/hiring-bias.oscal.yaml"

class HiringModel(nn.Module):
    def __init__(self, input_dim):
        super(HiringModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, choices=['wandb', 'local'], default="local")
    args = parser.parse_args()

    use_wandb = (args.framework == 'wandb')

    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    df['target'] = (df['class'] == '>50K').astype(int)
    
    if use_wandb:
        wandb.init(
            project="Hiring-Governance-WandB",
            config={
                "dataset": "Adult Income",
                "architecture": "MLP",
                "framework": "PyTorch"
            }
        )

    # 2. PRE-TRAINING GOVERNANCE
    print("[Venturalitica] 🛡️ Checking Adult Income Data for Bias...")
    venturalitica.enforce(
        data=df,
        target='target',
        sex='sex',
        policy=[HIRING_POLICY_PATH]
    )

    # 3. Preparation
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    X = numeric_df.drop('target', axis=1).values.astype(np.float32)
    y = numeric_df['target'].values.astype(np.float32).reshape(-1, 1)
    
    model = HiringModel(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if use_wandb:
        wandb.watch(model)

    # 4. Training Loop
    print("  Training model...")
    inputs = torch.from_numpy(X[:2000])
    labels = torch.from_numpy(y[:2000])
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if use_wandb and epoch % 10 == 0:
            wandb.log({"loss": loss.item(), "epoch": epoch})

    # 5. POST-TRAINING GOVERNANCE
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).numpy()
    
    eval_df = df.iloc[:2000].copy()
    eval_df['prediction'] = (predictions > 0.5).astype(int)

    print("[Venturalitica] 🛡️ Checking Model Compliance...")
    results = venturalitica.enforce(
        data=eval_df,
        target='target',
        prediction='prediction',
        sex='sex',
        policy=[HIRING_POLICY_PATH]
    )

    if use_wandb:
        # Log compliance results to WandB
        compliance_metrics = {f"governance/{res.control_id}": int(res.passed) for res in results}
        wandb.log(compliance_metrics)
        wandb.finish()

    print(f"  ✓ Process finished. Mode: {args.framework}")

if __name__ == "__main__":
    train()
