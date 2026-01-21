import pandas as pd
import numpy as np
from pathlib import Path

def generate_hiring_data(n_samples=1000):
    np.random.seed(42)
    
    # Features
    experience = np.random.randint(0, 20, n_samples)
    education = np.random.randint(12, 22, n_samples) # years of education
    gender = np.random.choice(['male', 'female'], n_samples)
    
    # Biased ground truth: males with less experience get hired more often in this synthetic "bad" data
    hiring_score = 0.4 * experience + 0.3 * (education - 12)
    hiring_score[gender == 'male'] += 2.0 # Bias
    
    # Normalize and threshold
    hiring_score = (hiring_score - hiring_score.min()) / (hiring_score.max() - hiring_score.min())
    hired = (hiring_score > 0.6).astype(int)
    
    df = pd.DataFrame({
        'experience': experience,
        'education': education,
        'gender': gender,
        'hired': hired
    })
    
    return df

if __name__ == "__main__":
    df = generate_hiring_data()
    output_dir = Path(__file__).parent
    
    # Split
    train = df.sample(frac=0.8, random_state=42)
    test = df.drop(train.index)
    
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    print(f"Hiring data generated in {output_dir}")
