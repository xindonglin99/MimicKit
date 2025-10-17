"""
This could quickly check what the content is like in .pkl, .csv or .npz files. It could be useful if you don't know what format is your data file.

Usage:
    python tools/data_format/check_data_content.py --file path/to/file.pkl --type pkl
    python tools/data_format/check_data_content.py --file path/to/file.csv --type csv
    python tools/data_format/check_data_content.py --file path/to/file.npz --type npz
"""

import argparse
import pickle
import pandas as pd
import numpy as np
import os
import sys

def inspect_csv(file_path):
    """Inspect a CSV file using pandas."""
    try:
        df = pd.read_csv(file_path)
        print(f"\n✅ Successfully loaded CSV file: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nSample rows:")
        print(df.head(5))
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")

def inspect_pkl(file_path, max_items=10):
    """Inspect a pickle file that may contain dicts or DataFrames (csv)."""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        print(f"\n✅ Successfully loaded pickle file: {file_path}")
        print(f"Top-level object type: {type(data)}")

        if isinstance(data, pd.DataFrame):
            print(f"\n📘 DataFrame Summary:")
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            print(data.head(5))
        
        elif isinstance(data, dict):
            print(f"\n📂 Dictionary with {len(data)} keys. Showing up to {max_items} entries:")
            for i, (key, value) in enumerate(data.items()):
                if i >= max_items:
                    print(f"... ({len(data) - max_items} more keys)")
                    break
                print(f"\n🔑 Key: '{key}'")

                if isinstance(value, np.ndarray):
                    print(f"Type: numpy.ndarray, Shape: {value.shape}, Dtype: {value.dtype}")
                    
                print(f"Value: {value}")
        
        else:
            print(f"\n⚠️ Unrecognized object type. Please inspect manually.")

    except Exception as e:
        print(f"❌ Error reading pickle file: {e}")

def inspect_npz(file_path):
    """Inspect a .npz file using numpy."""
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"\n✅ Successfully loaded NPZ file: {file_path}")
        print(f"Contained arrays: {data.files}")
        for key in data.files:
            print(f"\n🔑 Key: '{key}'")
            print(f"First sample: {data[key][0]}")
    except Exception as e:
        print(f"❌ Error reading NPZ file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inspect contents of a .pkl, .csv or .npz file.")
    parser.add_argument("--file", required=True, help="Path to the file.")
    parser.add_argument("--type", required=True, choices=["pkl", "csv", "npz"], help="File type (pkl, csv, or npz).")
    args = parser.parse_args()

    file_path = args.file
    file_type = args.type.lower()

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    if file_type == "csv":
        inspect_csv(file_path)
    elif file_type == "pkl":
        inspect_pkl(file_path)
    elif file_type == "npz":
        inspect_npz(file_path)

if __name__ == "__main__":
    main()
