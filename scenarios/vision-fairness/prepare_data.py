"""
🎯 PREPARE DATA: FairFace Cache
================================================================================
Downloads and caches a subset of the FairFace dataset for reproducible training.
"""

import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from PIL import Image
import shutil

# Configuration
CACHE_DIR = Path("datasets/vision/fairface_cache")
NUM_SAMPLES = 2000  # Subset for demo speed

def prepare_data():
    print(f"🚀 Preparing FairFace dataset ({NUM_SAMPLES} samples)...")
    
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "images").mkdir(exist_ok=True)
    
    # Load Streaming (1.25 padding)
    print("  ⬇️  Streaming from Hugging Face...")
    dataset = load_dataset("HuggingFaceM4/FairFace", "1.25", split="train", streaming=True)
    
    # FairFace mappings (Fixed Taxonomy matching Paper)
    race_map = {0: 'White', 1: 'Black', 2: 'East Asian', 3: 'Indian', 4: 'Latino', 5: 'Middle Eastern', 6: 'Southeast Asian'}
    gender_map = {0: 'Male', 1: 'Female'}
    
    samples = []
    
    print(f"  🎯 Mode: Full Dataset Download (No limit)")

    valid_count = 0
    # Calculate stats on the fly
    buckets = {race: 0 for race in race_map.values()}

    for i, item in enumerate(dataset):
        try:
            r_id = item['race']
            race_name = race_map.get(r_id, 'Unknown')
            
            # Save Image
            img_filename = f"{valid_count:06d}.jpg" # Increased zero padding for 100k+
            img_path = CACHE_DIR / "images" / img_filename
            item['image'].convert('RGB').save(img_path)
            
            # Save Metadata
            samples.append({
                'image_path': str(img_path),
                'race': race_name,
                'gender': gender_map.get(item['gender'], 'Unknown'),
                'age': item['age']
            })
            
            buckets[race_name] += 1
            valid_count += 1
            
            if valid_count % 100 == 0:
                stats = " | ".join([f"{k[:2]}:{v}" for k,v in buckets.items()])
                print(f"  ✓ {valid_count} imgs [{stats}]", end='\r')
                
        except Exception as e:
            print(f"  ⚠️  Skipping error: {e}")
            continue

    print(f"\n  💾 Saving metadata to {CACHE_DIR}/metadata.csv")
    df = pd.DataFrame(samples)
    df.to_csv(CACHE_DIR / "metadata.csv", index=False)
    print("✅ Full Data preparation complete.")

if __name__ == "__main__":
    prepare_data()
