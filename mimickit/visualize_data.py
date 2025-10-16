#!/usr/bin/env python3
"""
inspect_file.py

Inspect .pkl or .csv files interactively.

Usage:
    python inspect_file.py --file path/to/file.pkl --type pkl
    python inspect_file.py --file path/to/file.csv --type csv
"""

import argparse
import pickle
import pandas as pd
import numpy as np
import os
import sys
import joblib
import anim.motion as motion

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

def summarize_value(value, max_chars=300):
    """Summarize a single value: type, shape, and preview."""
    summary = f"Type: {type(value)}"

    # Add shape or length info
    if hasattr(value, "shape"):
        summary += f", Shape: {value.shape}"
    elif isinstance(value, (list, tuple, dict, set)):
        summary += f", Length: {len(value)}"

    # Add content preview
    preview = None
    try:
        if isinstance(value, pd.DataFrame):
            preview = value.head(2).to_string()
        elif isinstance(value, (np.ndarray, list, tuple)):
            preview = np.array(value[:5]) if len(value) > 5 else np.array(value)
        elif isinstance(value, dict):
            preview = f"Keys: {list(value.keys())[:5]}"
        else:
            preview = str(value)
    except Exception:
        preview = "<unprintable value>"

    preview = str(preview)
    if len(preview) > max_chars:
        preview = preview[:max_chars] + " ..."

    summary += f"\n  Sample: {preview}"
    return summary

def inspect_pkl(file_path, max_items=10):
    """Inspect a pickle file that may contain dicts, DataFrames, or arrays."""
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
                print(summarize_value(value))
        
        elif isinstance(data, list):
            print(f"\n📦 List with {len(data)} elements.")
            if len(data) > 0:
                print("First element summary:")
                print(summarize_value(data[0]))
        
        else:
            print(f"\n⚠️ Unrecognized object type. Preview:")
            print(summarize_value(data))

    except Exception as e:
        print(f"❌ Error reading pickle file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inspect contents of a .pkl or .csv file.")
    parser.add_argument("--file", required=True, help="Path to the file.")
    parser.add_argument("--type", required=True, choices=["pkl", "csv"], help="File type (pkl or csv).")
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

if __name__ == "__main__":
    main()
