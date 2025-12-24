import os
import numpy as np
import torch

data_path = "/root/autodl-tmp/GaussianGraph/dataset/mydata/playroom/language_features"

def check_files():
    files = os.listdir(data_path)
    s_files = [f for f in files if f.endswith('_s.npy')]
    f_files = [f for f in files if f.endswith('_f.npy')]
    
    print(f"Found {len(s_files)} _s.npy files and {len(f_files)} _f.npy files.")
    
    # Check a few _s.npy files
    for i, f in enumerate(s_files[:5]):
        path = os.path.join(data_path, f)
        try:
            data = np.load(path)
            print(f"\nChecking {f}:")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Min: {np.min(data)}, Max: {np.max(data)}")
            print(f"  Has NaN: {np.isnan(data).any()}")
            print(f"  Has Inf: {np.isinf(data).any()}")
            
            # Check levels
            if data.ndim == 3 and data.shape[0] == 4:
                for l in range(4):
                    level_data = data[l]
                    print(f"    Level {l}: Min {np.min(level_data)}, Max {np.max(level_data)}")
                    # Check if background is 0 or -1
                    # If float, it might be 0.0 or -1.0
                    unique_vals = np.unique(level_data)
                    print(f"    Level {l} unique values count: {len(unique_vals)}")
                    if len(unique_vals) < 10:
                        print(f"    Level {l} unique values: {unique_vals}")
        except Exception as e:
            print(f"Error reading {f}: {e}")

if __name__ == "__main__":
    check_files()
