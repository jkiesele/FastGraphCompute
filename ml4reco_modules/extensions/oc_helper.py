import torch
import ml4reco_modules.extensions
import os.path as osp

#load the lib
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'oc_helper_cpu.so'))
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'oc_helper_cuda.so'))


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
              
            - **max_unique_per_split (torch.Tensor)**: 
              A scalar tensor representing the maximum number of unique values found in any row split.
              
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
    num_unique_per_split = torch.bincount(unique_row_splits.to(torch.int64), minlength=len(lengths))  # shape (R-1,)
    
    # Find the maximum number of unique values per row split
    max_unique_per_split = torch.max(num_unique_per_split)

    #cast unique_vals, unique_row_splits, max_unique_per_split to same dtype as truth_indices
    unique_vals = unique_vals.type(truth_indices.dtype)
    unique_row_splits = unique_row_splits.type(truth_indices.dtype)
    max_unique_per_split = max_unique_per_split.type(truth_indices.dtype)
    
    return unique_vals, unique_row_splits, max_unique_per_split, lengths, max_length

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
        torch.Tensor: The M_not matrix (either calculated or filled with -1s).
                      It contains indices to select all points that do *not* belong to the object.
                      Row split boundaries are not crossed.
                      The dimensionality is (N_objects, N_max_points_per_row_split).
                      The matrix is not sorted by row splits anymore.
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
