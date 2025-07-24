import time
import torch
import os.path as osp
import fastgraphcompute
# Your compiled C++ extension module
from fastgraphcompute.extensions.oc_helper import max_same_valued_entries_per_row_split

torch.ops.load_library(osp.join(osp.dirname(osp.realpath(
    fastgraphcompute.extensions.__file__)), 'oc_helper_helper.so'))
unique_with_counts = torch.ops.oc_helper_helper.unique_with_counts


def ref_max_same_valued_entries_per_row_split(asso_idx, row_splits, filter_negative: bool = True):
    """
    This function calculates the maximum number of the same values in each row split.

    Args:
        asso_idx (torch.Tensor): A tensor containing indices (e.g., labels or values).
        row_splits (torch.Tensor): A tensor defining the start and end points of each row split.

    Returns:
        max_per_split (torch.Tensor): A tensor containing the maximum count of the same values for each row split.
        global_max (int): The global maximum count across all row splits.
        objects_per_split (torch.Tensor): A tensor containing the number of objects in each row split.
                                          Note: affected by filter_negative

    Notes:
        FIXME: Eventually this should be replaced by a C++/CUDA implementation to avoid the Python loop and enable jit.
    """

    n_row_splits = row_splits.size(0) - 1  # number of row splits
    max_per_split = torch.zeros(
        n_row_splits, dtype=torch.int64, device=asso_idx.device)
    objects_per_split = torch.zeros(
        n_row_splits, dtype=torch.int64, device=asso_idx.device)

    for rs_idx in range(n_row_splits):
        start_vertex = row_splits[rs_idx]
        end_vertex = row_splits[rs_idx + 1]

        # Extract the slice of asso_idx for the current row split
        asso_idx_slice = asso_idx[start_vertex:end_vertex]

        # Filter out negative values (asso_idx >= 0)
        if filter_negative:
            asso_idx_filtered = asso_idx_slice[asso_idx_slice >= 0]
        else:
            asso_idx_filtered = asso_idx_slice

        if asso_idx_filtered.numel() == 0:
            continue  # Skip if no valid indices in this split

        # Perform unique operation on the filtered slice and get counts
        unique_vals, counts = torch.unique(
            asso_idx_filtered, return_counts=True)

        # Find the maximum count and store it for this row split
        max_per_split[rs_idx] = counts.max()
        objects_per_split[rs_idx] = unique_vals.size(0)

    # Return the max_per_split and the global maximum value across all splits
    global_max = max_per_split.max()
    return max_per_split, global_max, objects_per_split


def performance_test_unique_with_counts(device='cpu'):
    print(f"\nPerformance Test on device: {device}")

    num_elements = 10_000_000  # 10 million elements
    num_unique = 100_000       # Number of unique elements

    # Generate a large tensor with duplicates
    x = torch.randint(0, num_unique, (num_elements,),
                      dtype=torch.int64, device=device)

    # Warm-up (to avoid initial overhead)
    for _ in range(3):
        _ = unique_with_counts(x)

    # Time custom unique_with_counts function
    start_time = time.time()
    unique_vals_custom, counts_custom = unique_with_counts(x)
    # force synchronise
    if device != 'cpu':
        torch.cuda.synchronize()
    custom_time = time.time() - start_time
    print(f"Custom unique_with_counts time: {custom_time:.4f} seconds")

    # Time torch.unique function
    start_time = time.time()
    unique_vals_torch, counts_torch = torch.unique(x, return_counts=True)
    if device != 'cpu':
        torch.cuda.synchronize()
    torch_time = time.time() - start_time
    print(f"torch.unique time: {torch_time:.4f} seconds")

    # Verify correctness
    if not torch.equal(unique_vals_custom, unique_vals_torch):
        print("Unique values do not match!")
    else:
        print("Unique values match.")

    if not torch.equal(counts_custom, counts_torch):
        print("Counts do not match!")
    else:
        print("Counts match.")

    # Print number of unique elements
    print(f"Number of unique elements: {unique_vals_custom.size(0)}")

    # Performance comparison
    if custom_time < torch_time:
        print(
            f"Custom function is faster by {torch_time - custom_time:.4f} seconds.")
    else:
        print(
            f"torch.unique is faster by {custom_time - torch_time:.4f} seconds.")


def performance_test_max_same_valued_entries_per_row_split(device='cpu'):
    asso_idx, row_splits = torch.randint(-1, 2000, (300000, 1), dtype=torch.int64), torch.tensor([
        0, 4000, 200000, 300000], dtype=torch.int64)

    asso_idx = asso_idx.to(device)
    row_splits = row_splits.to(device)

    def test(func, asso_idx, row_splits, cuda):
        start_time = time.time()
        for _ in range(10):
            _ = func(asso_idx, row_splits, True)
            if cuda:
                torch.cuda.synchronize()
        return (time.time() - start_time) / 10.

    t_op = test(ref_max_same_valued_entries_per_row_split,
                asso_idx, row_splits, cuda=device != 'cpu')
    c_op = test(max_same_valued_entries_per_row_split,
                asso_idx, row_splits, cuda=device != 'cpu')

    print(
        f"time comparison: just torch took {t_op}s, custom took {c_op}s per call on device {device}")


if __name__ == '__main__':
    performance_test_unique_with_counts(device='cpu')
    performance_test_max_same_valued_entries_per_row_split('cpu')

    if torch.cuda.is_available():
        performance_test_unique_with_counts(device='cuda')
        performance_test_max_same_valued_entries_per_row_split('cuda')
    else:
        print("\nCUDA is not available. Skipping GPU tests.")
