#!/usr/bin/env python3
import os
import sys
import django
import pandas as pd
import numpy as np

# Initialize Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Setting.settings')
try:
    django.setup()
except Exception as e:
    print(f"Error setting up Django environment: {e}")
    sys.exit(1)

from customer_signal.ml_pipeline import preprocess, _load_model
from customer_signal.signal_logic import risk_band

def test_ml_pipeline(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' does not exist.")
        return

    print("=" * 70)
    print(f"Reading CSV file: {csv_path}")
    print("=" * 70)

    try:
        df = pd.read_csv(csv_path)
        # print(df.head(3))
        # print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Strip whitespaces from headers to prevent column mismatch
    # df.columns = df.columns.str.strip()

    if 'customer_id' not in df.columns:
        print("Error: CSV must contain a 'customer_id' column.")
        return

    print(f"Found {len(df)} customer records in CSV.")

    # Load Model Bundle
    print("Loading Decision Tree Regressor bundle...")
    try:
        bundle = _load_model()
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]
        print(f"Model loaded successfully. Expecting {len(feature_columns)} features:")
        print("-" * 70)
        # Print features in groups of 4 for readability
        for i in range(0, len(feature_columns), 4):
            chunk = feature_columns[i:i+4]
            print("  ".join(f"{col:<22}" for col in chunk))
        print("-" * 70)
    except Exception as e:
        print(f"Failed to load model bundle: {e}")
        return

    # Preprocess
    print("Running preprocessing pipeline...")
    try:
        processed_df = preprocess(df, is_training=False)
        processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)
        print(processed_df.head(3).to_string(index=False))


    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return

    # Predict
    print("Running model inference...")
    try:
        preds = model.predict(processed_df)
        preds = np.clip(preds, 0, 100)
    except Exception as e:
        print(f"Model prediction failed: {e}")
        return

    # Print results sequentially
    print("\n" + "=" * 70)
    print(f"{'Customer ID':<20} | {'Churn Score':<12} | {'Risk Band':<12}")
    print("-" * 70)

    high_count, attention_count, low_count = 0, 0, 0
    for idx, row in df.iterrows():
        cid = row['customer_id']
        score = preds[idx]
        band = risk_band(float(score))

        if band == "High":
            high_count += 1
        elif band == "Attention":
            attention_count += 1
        else:
            low_count += 1

        print(f"{cid:<20} | {score:<12.2f} | {band:<12}")

    print("=" * 70)
    print("Summary of predictions:")
    print(f"  -> High Risk Check : {high_count}")
    print(f"  -> Attention Risk  : {attention_count}")
    print(f"  -> Low Risk Check  : {low_count}")
    print(f"Total processed      : {len(df)}")
    print("=" * 70)

if __name__ == '__main__':
    # Default to the random split dataset if no path provided
    default_csv = os.path.join("datasets", "Telco_customer_dashboard1.csv")
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv
    test_ml_pipeline(csv_path)
