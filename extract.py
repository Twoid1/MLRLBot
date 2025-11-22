#!/usr/bin/env python3
"""
Extract the correct 50 features from your ML model dict
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("EXTRACTING CORRECT FEATURES")
print("="*80)

# Load ML dict
ml_path = Path('models/ml/ml_predictor.pkl')
ml_dict = joblib.load(ml_path)

# Extract the 50 selected features
selected_features = ml_dict['selected_features']

print(f"\n Found {len(selected_features)} selected features")
print(f"\nFirst 10:")
for i, feat in enumerate(selected_features[:10], 1):
    print(f"  {i:2d}. {feat}")

print(f"\nLast 10:")
for feat in selected_features[-10:]:
    print(f"  - {feat}")

# Test with model
print("\n" + "="*60)
print("TESTING WITH MODEL")
print("="*60)

model = ml_dict['model']
scaler = ml_dict['scaler']

# Create test data
test_df = pd.DataFrame(
    np.random.randn(3, len(selected_features)),
    columns=selected_features
)

print(f"\nTest data shape: {test_df.shape}")

try:
    # Scale features (model was trained with scaled data!)
    test_scaled = scaler.transform(test_df)
    
    # Predict
    predictions = model.predict_proba(test_scaled)
    
    print(f" SUCCESS! Model predictions work")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Sample predictions:\n{predictions}")
    
except Exception as e:
    print(f" FAILED: {e}")
    exit(1)

# Save selected features
output_dir = Path('models/features')
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / 'selected_features.pkl'
joblib.dump(selected_features, output_file)

print(f"\n Saved to: {output_file}")

# Verify
loaded = joblib.load(output_file)
print(f"\nVerification:")
print(f"  Type: {type(loaded)}")
print(f"  Length: {len(loaded)}")
print(f"  First 5: {loaded[:5]}")

print("\n" + "="*80)
print(" SUCCESS!")
print("="*80)
print(f"\nSaved: {len(selected_features)} features")
print("\nNext: python test_live.py")