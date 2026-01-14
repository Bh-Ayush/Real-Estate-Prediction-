#!/usr/bin/env python3
"""
Utility script to convert CSV files to Parquet format for efficient storage and loading.
Parquet files are ~10x smaller than CSV and load much faster.

Usage:
    python convert_to_parquet.py <input.csv> [output.parquet]
    
Example:
    python convert_to_parquet.py realtor_data.csv realtor_data.parquet
"""

import pandas as pd
import sys
import os


def convert_csv_to_parquet(csv_path, output_path=None):
    """
    Convert CSV file to Parquet format.
    
    Args:
        csv_path (str): Path to input CSV file
        output_path (str, optional): Path to output Parquet file. 
                                   If None, uses same name with .parquet extension.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)
    
    if output_path is None:
        output_path = csv_path.rsplit('.', 1)[0] + '.parquet'
    
    print(f"Reading CSV: {csv_path}")
    try:
        # Read CSV with progress indication
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # Get file sizes
    csv_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
    
    print(f"Converting to Parquet: {output_path}")
    try:
        df.to_parquet(output_path, compression='snappy', index=False)
        parquet_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = (1 - parquet_size / csv_size) * 100
        
        print(f"✓ Conversion complete!")
        print(f"  CSV size:      {csv_size:.2f} MB")
        print(f"  Parquet size:  {parquet_size:.2f} MB")
        print(f"  Compression:   {compression_ratio:.1f}% smaller")
        print(f"\nYou can now upload '{output_path}' to the Streamlit app!")
    except Exception as e:
        print(f"Error converting to Parquet: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_parquet.py <input.csv> [output.parquet]")
        print("\nExample:")
        print("    python convert_to_parquet.py realtor_data.csv realtor_data.parquet")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_csv_to_parquet(csv_file, output_file)
