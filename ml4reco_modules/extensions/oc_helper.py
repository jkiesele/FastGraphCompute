import torch
import ml4reco_modules.extensions
import os.path as osp

#load the lib
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'oc_helper_cpu.so'))
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'oc_helper_cuda.so'))

torch.ops.load_library(osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'select_with_default_cpu.so'))
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'select_with_default_cuda.so'))

def max_same_valued_entries_per_row_split(asso_idx, row_splits, filter_negative: bool = True):
    """
    This function calculates the maximum number of the same values in each row split.

    Args:
        asso_idx (torch.Tensor): A tensor containing indices (e.g., labels or values).
        row_splits (torch.Tensor): A tensor defining the start and end points of each row split.
    
    Returns:
        max_per_split (torch.Tensor): A tensor containing the maximum count of the same values for each row split.
        global_max (int): The global maximum count across all row splits.

    Notes:
        FIXME: Eventually this should be replaced by a C++/CUDA implementation to avoid the Python loop and enable jit.
    """
    
    n_row_splits = row_splits.size(0) - 1  # number of row splits
    max_per_split = torch.zeros(n_row_splits, dtype=torch.int64, device=asso_idx.device)

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
        unique_vals, counts = torch.unique(asso_idx_filtered, return_counts=True)

        # Find the maximum count and store it for this row split
        max_per_split[rs_idx] = counts.max()

    # Return the max_per_split and the global maximum value across all splits
    global_max = max_per_split.max()
    return max_per_split, global_max


def _helper_inputs(truth_indices, row_splits, filter_negative: bool = True):
    """
    Processes the `truth_indices` tensor by splitting it according to `row_splits`,
    computes the unique values for each split, and optionally filters out negative values.

    Args:
        truth_indices (torch.Tensor): 
            A 1D tensor of shape `(V,)` containing values that represent indices or labels. 
            It may contain duplicate values, and potentially negative values.
            
        row_splits (torch.Tensor): 
            A 1D tensor of shape `(R,)` where `R` is the number of row split positions. 
            It defines how the `truth_indices` tensor is split into independent segments. 
            The first element must be 0, and the last element must be the length of `truth_indices`.
            
        filter_negative (bool, optional): 
            If `True`, any negative values in `truth_indices` will be filtered out. 
            Defaults to `True`.
    
    Returns:
        tuple:
            - **unique_vals (torch.Tensor)**: 
              A 1D tensor containing the unique values from `truth_indices` for all row splits, 
              optionally filtered to exclude values less than 0.
              
            - **unique_row_splits (torch.Tensor)**: 
              A 1D tensor that maps each unique value from `unique_vals` to its corresponding 
              row split, based on `row_splits`.
              
            - **max_same_unique_per_split (torch.Tensor)**: 
              A scalar tensor representing the maximum number of same unique values found in any row split.
              
            - **lengths (torch.Tensor)**: 
              A 1D tensor that stores the lengths of each segment created by `row_splits`, 
              i.e., the number of elements between consecutive values in `row_splits`.
              
            - **max_length (torch.Tensor)**: 
              A scalar tensor representing the maximum length of any row split segment, 
              i.e., the maximum value in `lengths`.
              
    Description:
        The `_helper_inputs` function is designed to process an input tensor `truth_indices` that
        is split into multiple segments according to the positions specified in `row_splits`. For each
        segment, it calculates the unique values and determines which segment each unique value belongs to.
        
        The function supports an optional filtering of negative values from `truth_indices`, controlled by the 
        `filter_negative` argument. When `filter_negative=True`, any values in `truth_indices` that are less 
        than zero will be excluded from the results.
        
        The function also counts how many unique values appear in each row split and returns the maximum number 
        of unique values found in any row split. Additionally, it computes the lengths of the segments created 
        by `row_splits` and returns the maximum segment length.
        
    Example Usage:
        ```python
        truth_indices = torch.tensor([0, 1, 5, 60, 3, 3, -10, 90, 4, 3, -3, 3])
        row_splits = torch.tensor([0, 6, 12])
        
        unique_vals, unique_row_splits, max_unique_per_split, lengths, max_length = _helper_inputs(truth_indices, row_splits)
        
        print('Unique Values:', unique_vals)
        print('Unique Row Splits:', unique_row_splits)
        print('Max Unique per Split:', max_unique_per_split)
        print('Lengths of each split:', lengths)
        print('Max Length of any split:', max_length)
        ```

    Notes:
        - Beware: the CPU and CUDA versions do not give the exact identical output in terms of index ordering, but give the same output
          in terms of functionality.
        - Ensure that `row_splits` starts with 0 and ends with the length of `truth_indices`, as the function assumes 
          the first and last values in `row_splits` to be boundaries for splitting.
        - If `truth_indices` contains negative values and `filter_negative=True`, those values will not be included 
          in the final result.
    """

    
    # Calculate lengths of each segment
    lengths = row_splits[1:] - row_splits[:-1]
    max_length = torch.max(lengths)
    
    # Generate row IDs for each element in truth_indices
    row_ids = torch.repeat_interleave(torch.arange(len(lengths)).to(truth_indices.device), lengths)
    
    # Get unique truth indices and their inverse mapping
    unique_vals, inverse_indices = torch.unique(truth_indices, return_inverse=True)
    
    # Combine both into a single matrix of shape (V, 2) to keep track of value and row split
    value_and_row_ids = torch.stack((unique_vals[inverse_indices], row_ids), dim=1)
    
    # Now we want to extract the unique combinations of value and row id, excluding -1s
    valid_value_and_row_ids = value_and_row_ids[value_and_row_ids[:, 0] != -1]  # filter out -1s
    unique_value_and_row_ids = torch.unique(valid_value_and_row_ids, dim=0)  # shape (U', 2)
    
    # Split the result into unique values and corresponding row splits
    unique_vals = unique_value_and_row_ids[:, 0]
    unique_row_splits = unique_value_and_row_ids[:, 1]

    if filter_negative:
        valid_mask = unique_vals >= 0
        unique_vals = unique_vals[valid_mask]
        unique_row_splits = unique_row_splits[valid_mask]

    
    # Count the number of unique values per row split
    #ucop = torch.ops.row_split_unique_count.row_split_unique_count
    _, max_same_unique_per_split = max_same_valued_entries_per_row_split(truth_indices, row_splits, filter_negative)
    
    #cast unique_vals, unique_row_splits, max_unique_per_split to same dtype as truth_indices
    unique_vals = unique_vals.type(truth_indices.dtype)
    unique_row_splits = unique_row_splits.type(truth_indices.dtype)
    max_same_unique_per_split = max_same_unique_per_split.type(truth_indices.dtype)
    
    return unique_vals, unique_row_splits, max_same_unique_per_split, lengths, max_length

#for jit
def _helper_inputs_filter(truth_indices, row_splits):
    return _helper_inputs(truth_indices, row_splits, filter_negative=True)



def oc_helper_matrices(
        truth_idxs: torch.Tensor,
        row_splits: torch.Tensor,
        calc_m_not: bool = True):
    
    """
    This function calculates the matrices M and M_not for the oc_helper module.

    Arguments:
        truth_idxs (torch.Tensor): The indices of the true particles.
        row_splits (torch.Tensor): The row_splits tensor that defines how the
                                   truth_idxs are split into segments.
        calc_m_not (bool): If True, the M_not matrix is also calculated, otherwise
                           it will contain -1s.

    Returns:
        torch.Tensor: The M matrix, as indices such that the point properties can be selected with select_with_default.
                      Row split boundaries are not crossed.
                      The dimensionality is (N_objects, N_max_points_per_object).
                      The matrix is not sorted by row splits anymore.
                      Can be used in conjunction with select_with_default to select the points.
        torch.Tensor: The M_not matrix (either calculated or filled with -1s).
                      It contains indices to select all points that do *not* belong to the object.
                      Row split boundaries are not crossed.
                      The dimensionality is (N_objects, N_max_points_per_row_split).
                      The matrix is not sorted by row splits anymore.
                      Can be used in conjunction with select_with_default to select the points.
    """

    # Sanity check: ensure both tensors are on the same device
    assert truth_idxs.device == row_splits.device, "Both truth_idxs and row_splits must be on the same device"
    
    if truth_idxs.device.type == 'cuda':
        op = torch.ops.oc_helper_cuda.oc_helper_cuda
    else:
        op = torch.ops.oc_helper_cpu.oc_helper_cpu

    # get the helper inputs filtered
    unique_idxs, unique_rs_asso, max_n_unique_over_splits, _, max_n_in_splits = _helper_inputs_filter(truth_idxs, row_splits)

    # add at least one dimension since the op interface requires it
    max_n_unique_over_splits = max_n_unique_over_splits.unsqueeze(0)
    max_n_in_splits = max_n_in_splits.unsqueeze(0)

    # sort by unique_rs_asso to maintain row split order - not necessary but easier to debug and better for coalesced memory access
    sorted_indices = unique_rs_asso.argsort()
    unique_idxs = unique_idxs[sorted_indices]
    unique_rs_asso = unique_rs_asso[sorted_indices]

    '''
     torch::Tensor asso_idx,
    torch::Tensor unique_idx,
    torch::Tensor unique_rs_asso,
    torch::Tensor rs,
    torch::Tensor max_n_unique_over_splits,
    torch::Tensor max_n_in_splits,
    bool calc_m_not
    '''
    # Call the C++ or CUDA operation
    M, M_not = op(
        truth_idxs,
        unique_idxs, 
        unique_rs_asso,
        row_splits,
        max_n_unique_over_splits, 
        max_n_in_splits,
        calc_m_not)
    
    return M, M_not

def select_with_default(indices: torch.Tensor,
                        tensor: torch.Tensor,
                        default):
    """
    This function selects the values from a tensor based on the given indices,
    e.g. the M or M_not matrix from oc_helper_matrices.
    For -1 entries in M or M_not, it uses the default value.
    """
    if tensor.device.type == 'cuda':
        op = torch.ops.select_with_default_cuda.select_with_default_cuda
    else:
        op = torch.ops.select_with_default_cpu.select_with_default_cpu

    return op(indices, tensor, default)