#!/usr/bin/env python3
"""
Setup helper script to prepare your realtor dataset for the Streamlit dashboard.
This script will:
1. Check your CSV file size
2. Convert to Parquet if needed
3. Validate the data structure

Usage:
    python setup_dataset.py <your_dataset.csv>
"""

import pandas as pd
import sys
import os
from pathlib import Path


def check_required_columns(df, required_cols=None):
    """Check if dataset has the expected columns."""
    if required_cols is None:
        required_cols = ['price', 'bed', 'bath', 'state', 'city']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"⚠️  Missing columns: {missing}")
        print(f"   Found columns: {list(df.columns)}")
        return False
    return True


def setup_dataset(filepath):
    """Prepare dataset for the dashboard."""
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found.")
        sys.exit(1)
    
    filename = os.path.basename(filepath)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    print("=" * 60)
    print("Real-Estate-Prediction Setup")
    print("=" * 60)
    print(f"\n📁 File: {filename}")
    print(f"📊 Size: {file_size_mb:.2f} MB")
    
    # Read and validate
    print("\n⏳ Reading dataset...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)
    
    # Check columns
    print("\n✓ Checking column structure...")
    if check_required_columns(df):
        print("  ✓ All required columns found")
    else:
        print("  ⚠️  Some expected columns missing (may still work)")
    
    # Suggest Parquet conversion if large
    print("\n📦 Format recommendation:")
    if file_size_mb > 100:
        print(f"  ⚠️  Your CSV is {file_size_mb:.0f} MB - conversion to Parquet recommended!")
        parquet_path = filepath.rsplit('.', 1)[0] + '.parquet'
        response = input(f"\n  Convert to Parquet? (y/n): ").strip().lower()
        
        if response == 'y':
            print(f"  Converting...")
            try:
                df.to_parquet(parquet_path, compression='snappy', index=False)
                parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)
                compression = (1 - parquet_size / file_size_mb) * 100
                
                print(f"  ✓ Converted to: {parquet_path}")
                print(f"  📉 Size reduced from {file_size_mb:.2f} MB → {parquet_size:.2f} MB ({compression:.0f}% smaller)")
                print(f"\n  💡 Tip: Upload '{parquet_path}' instead of '{filename}' to the Streamlit app")
                dataset_to_use = parquet_path
            except Exception as e:
                print(f"  ❌ Conversion failed: {e}")
                dataset_to_use = filename
        else:
            dataset_to_use = filename
    else:
        print(f"  ✓ CSV size is good ({file_size_mb:.2f} MB)")
        dataset_to_use = filename
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ Setup Complete!")
    print("=" * 60)
    print(f"\n📌 Next steps:")
    print(f"   1. Run: streamlit run real_estate_dashboard.py")
    print(f"   2. Upload '{dataset_to_use}' using the sidebar file uploader")
    print(f"   3. Explore your real estate data!\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python setup_dataset.py <your_dataset.csv>")
        print("\nExample:")
        print("    python setup_dataset.py realtor_data.csv")
        sys.exit(1)
    
    setup_dataset(sys.argv[1])
