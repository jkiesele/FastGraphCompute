import time
import torch

from fastgraphcompute import oc_helper_matrices, select_with_default
from fastgraphcompute.extensions.oc_helper import _helper_inputs_filter

def benchmark_oc_helper_performance(device='cpu', num_nodes=1000000, num_row_splits=5, num_unique_per_split=1000, num_runs=5):
    """
    Benchmarks the performance of the oc_helper_matrices and select_with_default functions for large-scale inputs,
    averaging the time over a specified number of runs.

    Args:
        device (str): Device to run the benchmark on ('cpu' or 'cuda').
        num_nodes (int): The total number of nodes (length of asso_indices).
        num_row_splits (int): The number of row splits (number of batches).
        num_unique_per_split (int): The number of unique asso_indices per row split.
        num_runs (int): Number of times to run each operation to average the execution time.

    Returns:
        A dictionary containing the average execution times for each function over the specified number of runs.
    """
    #print all options passed
    print(f"device: {device}, num_nodes: {num_nodes}, num_row_splits: {num_row_splits}, num_unique_per_split: {num_unique_per_split}, num_runs: {num_runs}")

    # Generate random asso_indices with a mix of -1s and valid indices
    asso_indices = torch.randint(0, num_unique_per_split, (num_nodes,), dtype=torch.int32, device=device) - 1
    
    #make sure the size is still ok
    assert len(asso_indices) == num_nodes
    
    # Create row_splits, equally dividing the nodes
    split_size = num_nodes // num_row_splits
    row_splits = torch.arange(0, num_nodes + 1, split_size, dtype=torch.int32, device=device)

    # Ensure the row_splits ends at num_nodes
    row_splits[-1] = num_nodes

    # Benchmark oc_helper_matrices
    oc_helper_times = []
    
    # run once to warm up
    oc_helper_matrices(asso_indices, row_splits)
    torch.cuda.synchronize() if device == 'cuda' else None  # Ensure CUDA is synchronized
    torch.cuda.empty_cache() if device == 'cuda' else None  # Clear CUDA cache
    for _ in range(num_runs):
        start_time = time.time()
        M, M_not, _ = oc_helper_matrices(asso_indices, row_splits)
        torch.cuda.synchronize() if device == 'cuda' else None  # Ensure CUDA is synchronized
        oc_helper_times.append(time.time() - start_time)
        torch.cuda.empty_cache() if device == 'cuda' else None  # Clear CUDA cache
    avg_oc_helper_time = sum(oc_helper_times) / num_runs



    # Create a dummy expanded tensor for select_with_default test
    asso_indices_expanded = asso_indices.unsqueeze(-1).repeat(1, 10)

    # Benchmark select_with_default
    select_with_default_times = []
    # run once to warm up
    select_with_default(M, asso_indices_expanded, -100)
    torch.cuda.synchronize() if device == 'cuda' else None
    for _ in range(num_runs):
        start_time = time.time()
        sel = select_with_default(M, asso_indices_expanded, -100)
        torch.cuda.synchronize() if device == 'cuda' else None  # Ensure CUDA is synchronized
        select_with_default_times.append(time.time() - start_time)
    avg_select_with_default_time = sum(select_with_default_times) / num_runs

    # Return benchmark results
    return {
        "oc_helper_matrices_avg_time": avg_oc_helper_time,
        "select_with_default_avg_time": avg_select_with_default_time
    }


# Example of how to run the benchmark:
if __name__ == '__main__':

    if True:
        # Run the benchmark on CPU
        print("Benchmarking on CPU:")
        cpu_benchmark = benchmark_oc_helper_performance(device='cpu', num_nodes=1000000, num_row_splits=5, num_unique_per_split=100, num_runs=5)
        print(cpu_benchmark)
        cpu_benchmark = benchmark_oc_helper_performance(device='cpu', num_nodes=200, num_row_splits=2, num_unique_per_split=300, num_runs=5)
        print(cpu_benchmark)

    # Run the benchmark on CUDA (if available)
    if torch.cuda.is_available():
        print("Benchmarking on CUDA")
        cuda_benchmark = benchmark_oc_helper_performance(device='cuda', num_nodes=1000000, num_row_splits=5, num_unique_per_split=1000, num_runs=5)
        print(cuda_benchmark)
        cuda_benchmark = benchmark_oc_helper_performance(device='cuda', num_nodes=1000000, num_row_splits=5, num_unique_per_split=100, num_runs=5)
        print(cuda_benchmark)
        cuda_benchmark = benchmark_oc_helper_performance(device='cuda', num_nodes=200000, num_row_splits=1, num_unique_per_split=1000, num_runs=5)
        print(cuda_benchmark)
        cuda_benchmark = benchmark_oc_helper_performance(device='cuda', num_nodes=200000, num_row_splits=1, num_unique_per_split=100, num_runs=5)
        print(cuda_benchmark)
        cuda_benchmark = benchmark_oc_helper_performance(device='cuda', num_nodes=1000000, num_row_splits=20, num_unique_per_split=100, num_runs=5)
        print(cuda_benchmark)
