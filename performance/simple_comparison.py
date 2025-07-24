#!/usr/bin/env python3
"""
Simple comparison between FAISS and FastGraphCompute (FGC) for KNN search.
Minimal implementation with fixed parameters for quick testing.
"""

import time
import numpy as np
import torch

from fastgraphcompute import binned_select_knn

# Set fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Fixed parameters
K = [1, 5]
DIMENSIONS = [5]
TEST_SIZES = [50000, 500000, 1000000]


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
        return "NOT_INSTALLED"
    except Exception as e:
        print(f"FAISS error: {e}")
        return None


def time_fgc(data, k):
    """Time FGC KNN search."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Convert numpy array to PyTorch tensor (following test pattern)
        coordinates = torch.tensor(data, dtype=torch.float32, device=device).contiguous()
        row_splits = torch.tensor(
            [0, len(data)], dtype=torch.int64, device=device)

        # Warm up GPU if using CUDA
        if device == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        # Use the same function signature as tests - pass k as individual value
        indices, distances = binned_select_knn(
                k, coordinates, row_splits, direction=None, n_bins=None)

        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        return (end_time - start_time) * 1000  # Convert to ms
    except Exception as e:
        print(f"FGC error: {e}")
        return None


def main():
    print(f"Simple KNN Comparison (K={K}, D={DIMENSIONS})")
    print("=" * 80)

    # Check what's available
    try:
        import faiss
        faiss_available = True
    except ImportError:
        faiss_available = False
        print("Note: FAISS not installed - showing FGC performance only")
        print("To install FAISS: pip install faiss-gpu")
        print()

    for size in TEST_SIZES:
        print(f"\nSize: {size}")
        print("=" * 80)
        print(f"{'K':<4} {'D':<4} {'FAISS (ms)':<12} {'FGC (ms)':<12} {'Speedup':<10}")
        print("-" * 80)

        for k in K:
            for d in DIMENSIONS:
                # Generate test data
                data = generate_data(size, d)

                # Time both methods
                faiss_time = time_faiss(data, k)
                fgc_time = time_fgc(data, k)

                # Calculate speedup
                if isinstance(faiss_time, (int, float)) and isinstance(fgc_time, (int, float)):
                    speedup = f"{faiss_time / fgc_time:.2f}x"
                else:
                    speedup = "N/A"

                # Format results
                if faiss_time == "NOT_INSTALLED":
                    faiss_str = "NOT_INST"
                elif isinstance(faiss_time, (int, float)):
                    faiss_str = f"{faiss_time:.2f}"
                else:
                    faiss_str = "ERROR"

                fgc_str = f"{fgc_time:.2f}" if isinstance(
                    fgc_time, (int, float)) else "ERROR"

                print(f"{k:<4} {d:<4} {faiss_str:<12} {fgc_str:<12} {speedup:<10}")


if __name__ == "__main__":
    main()
