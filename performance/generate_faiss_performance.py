#!/usr/bin/env python3
"""
FAISS vs FastGraphCompute (FGC) Performance Benchmarking Tool
Continuously benchmarks and averages performance data, saving results to CSV.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from fastgraphcompute import binned_select_knn

# Configuration - Set to 'size' or 'dimension' to determine what to vary
TEST_MODE = 'size'  # 'size' or 'dimension'

# Fixed parameters
SEED = 42
K = 40

# Test ranges
SIZE_RANGE = list(range(1000, 10000, 1000)) + list(range(10000, 100000, 10000)) + list(range(100000, 500000, 50000)) + list(range(500000, 1000000, 100000)) + list(range(1000000, 3000001, 200000))
DIMENSION_RANGE = list(range(1, 101))  # 1 to 100
FIXED_SIZE = 100000  # Used when testing dimensions
FIXED_DIMENSION = 5  # Used when testing sizes

# CSV file paths
SIZE_CSV = 'faiss_data_sizes.csv'
DIM_CSV = 'faiss_data_dims.csv'

# Set seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)


def generate_data(n_points, dimensions, seed_offset=0):
    """Generate random data with fixed seed."""
    np.random.seed(SEED + seed_offset)
    return np.random.randn(n_points, dimensions).astype(np.float32)


def time_faiss(data, k):
    """Time FAISS KNN search."""
    try:
        import faiss

        # Build index
        index = faiss.IndexFlatL2(data.shape[1])
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        # Add data and search
        start_time = time.time()
        index.add(data)
        distances, indices = index.search(data, k)
        end_time = time.time()

        return (end_time - start_time) * 1000  # Convert to ms
    except ImportError:
        print("FAISS not installed. Install with: pip install faiss-gpu")
        return None
    except Exception as e:
        print(f"FAISS error: {e}")
        return None


def time_fgc(data, k):
    """Time FGC KNN search."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Convert numpy array to PyTorch tensor
        coordinates = torch.tensor(data, dtype=torch.float32, device=device).contiguous()
        row_splits = torch.tensor([0, len(data)], dtype=torch.int64, device=device)

        # Warm up GPU if using CUDA
        if device == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        indices, distances = binned_select_knn(
            k, coordinates, row_splits, direction=None, n_bins=None)

        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        return (end_time - start_time) * 1000  # Convert to ms
    except Exception as e:
        print(f"FGC error: {e}")
        return None


def load_or_create_dataframe(csv_path, test_mode):
    """Load existing CSV or create new DataFrame."""
    if os.path.exists(csv_path):
        print(f"Loading existing data from {csv_path}")
        df = pd.read_csv(csv_path)
        # Ensure all required columns exist
        required_cols = ['k', 'faiss_time', 'fgc_time', 'count']
        if test_mode == 'size':
            required_cols.insert(0, 'size')
        else:
            required_cols.insert(0, 'dimension')

        for col in required_cols:
            if col not in df.columns:
                if col == 'k':
                    df[col] = K  # Default to current K value
                elif col == 'count':
                    df[col] = 1
                else:
                    df[col] = 0.0
    else:
        print(f"Creating new DataFrame for {csv_path}")
        if test_mode == 'size':
            df = pd.DataFrame(columns=['size', 'k', 'faiss_time', 'fgc_time', 'count'])
        else:
            df = pd.DataFrame(columns=['dimension', 'k', 'faiss_time', 'fgc_time', 'count'])

    return df


def update_average(prev_avg, count, new_value):
    """Update average using count-based incremental averaging."""
    if new_value is None:
        return prev_avg, count
    return (prev_avg * count + new_value) / (count + 1), count + 1


def save_results(df, csv_path):
    """Save results to CSV."""
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


def main():
    print(f"FAISS vs FGC Performance Benchmarking (Mode: {TEST_MODE})")
    print("=" * 80)

    # Determine CSV file and test parameters
    if TEST_MODE == 'size':
        csv_path = SIZE_CSV
        test_values = SIZE_RANGE
        fixed_param = FIXED_DIMENSION
        param_name = 'size'
    else:
        csv_path = DIM_CSV
        test_values = DIMENSION_RANGE
        fixed_param = FIXED_SIZE
        param_name = 'dimension'

    # Load or create DataFrame
    df = load_or_create_dataframe(csv_path, TEST_MODE)

    # Check FAISS availability
    try:
        import faiss
        faiss_available = True
        print("FAISS is available")
    except ImportError:
        faiss_available = False
        print("FAISS not available - only FGC will be benchmarked")

    print(f"Testing {len(test_values)} different {param_name} values")
    print(f"K = {K}")
    print()

    for i, test_value in enumerate(test_values):
        print(f"Progress: {i+1}/{len(test_values)} - Testing {param_name}: {test_value}")

        # Generate data based on test mode
        if TEST_MODE == 'size':
            data = generate_data(test_value, fixed_param)
            lookup_value = test_value
        else:
            data = generate_data(fixed_param, test_value)
            lookup_value = test_value

        # Find existing row or create new one
        if TEST_MODE == 'size':
            existing_row = df[(df['size'] == lookup_value) & (df['k'] == K)]
        else:
            existing_row = df[(df['dimension'] == lookup_value) & (df['k'] == K)]

        if len(existing_row) == 0:
            # Create new row
            if TEST_MODE == 'size':
                new_row = {'size': lookup_value, 'k': K, 'faiss_time': 0.0, 'fgc_time': 0.0, 'count': 0}
            else:
                new_row = {'dimension': lookup_value, 'k': K, 'faiss_time': 0.0, 'fgc_time': 0.0, 'count': 0}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            row_idx = len(df) - 1
        else:
            row_idx = existing_row.index[0]

        # Get current values
        current_faiss_avg = df.loc[row_idx, 'faiss_time']
        current_fgc_avg = df.loc[row_idx, 'fgc_time']
        current_count = int(df.loc[row_idx, 'count'])

        # Time FAISS if available
        if faiss_available:
            faiss_time = time_faiss(data, K)
            if faiss_time is not None:
                new_faiss_avg, new_count_f = update_average(current_faiss_avg, current_count, faiss_time)
                df.loc[row_idx, 'faiss_time'] = new_faiss_avg
                print(f"  FAISS: {faiss_time:.2f}ms (avg: {new_faiss_avg:.2f}ms)")
            else:
                print(f"  FAISS: ERROR")

        # Time FGC
        fgc_time = time_fgc(data, K)
        if fgc_time is not None:
            new_fgc_avg, new_count_g = update_average(current_fgc_avg, current_count, fgc_time)
            df.loc[row_idx, 'fgc_time'] = new_fgc_avg
            print(f"  FGC: {fgc_time:.2f}ms (avg: {new_fgc_avg:.2f}ms)")
        else:
            print(f"  FGC: ERROR")

        # Update count (use whichever was successful)
        if faiss_time is not None or fgc_time is not None:
            df.loc[row_idx, 'count'] = current_count + 1

        # Calculate and display speedup
        if faiss_available and faiss_time is not None and fgc_time is not None:
            speedup = faiss_time / fgc_time
            avg_speedup = new_faiss_avg / new_fgc_avg if new_fgc_avg > 0 else 0
            print(f"  Speedup: {speedup:.2f}x (avg: {avg_speedup:.2f}x)")

        # Save results after each round
        save_results(df, csv_path)
        print()

    print("Benchmarking complete!")
    print(f"Final results saved to {csv_path}")


if __name__ == "__main__":
    main()
