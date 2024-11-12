import torch
def row_splits_from_strict_batch(batch):
    """
    Converts a *strictly increasing* PyTorch Geometric `batch` tensor into a `row_splits` tensor.

    Args:
        batch (torch.Tensor): A tensor of size `(num_nodes,)` where each element indicates
                              the batch index (graph index) that the corresponding node belongs to.
                              The `batch` tensor is assumed to be strictly increasing, meaning that
                              all nodes belonging to a given batch are grouped together in order.

    Returns:
        row_splits (torch.Tensor): A tensor of size `(num_batches + 1,)` that defines the
                                   start and end of each batch. The first element is 0, and 
                                   the last element is the total number of nodes. Each intermediate 
                                   element indicates where a new batch starts.
    
    Example:
        batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int32)
        row_splits = row_splits_from_batch(batch)
        # row_splits: tensor([0, 3, 5, 8], dtype=torch.int32)
    
    Description:
        This function assumes that the `batch` tensor is strictly increasing, meaning that all nodes 
        belonging to the same batch are contiguous. It calculates the `row_splits` tensor, which defines 
        the boundaries between consecutive batches in a concatenated representation of multiple batches 
        (or graphs). The first element of the `row_splits` tensor is always 0, and the last element is 
        the total number of nodes in the batch.
        
        The function finds the positions where the `batch` index changes using `torch.nonzero` and uses 
        these positions to define the row splits. If the `batch` tensor is not strictly increasing, the 
        function may produce incorrect results.

    """
    # Detect where the batch changes
    change_indices = torch.nonzero(batch[1:] != batch[:-1], as_tuple=False).squeeze()
    
    # Append the start (0) and end (number of nodes) to row splits
    return torch.cat([torch.tensor([0]), change_indices + 1, torch.tensor([len(batch)])]).to(torch.int32)


def strict_batch_from_row_splits(row_splits):
    """
    Converts a `row_splits` tensor into a PyTorch Geometric `batch` tensor.

    Args:
        row_splits (torch.Tensor): A tensor of size `(num_batches + 1,)` that defines the start 
                                   and end of each batch. The first element is 0, and the last
                                   element is the total number of nodes.

    Returns:
        batch (torch.Tensor): A tensor of size `(V,)`, where each element indicates which batch 
                              (or graph) the corresponding node belongs to. Each element of `batch`
                              contains the index of the batch that node belongs to.

    Example:
        row_splits = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
        batch = batch_from_row_splits(row_splits)
        # batch: tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int64)
    """
    # Ensure row_splits is int64 for tensor indexing
    row_splits = row_splits.to(torch.int64)

    # Compute lengths of each batch (difference between consecutive row splits)
    lengths = row_splits[1:] - row_splits[:-1]
    
    # Use repeat_interleave to assign batch indices
    batch = torch.repeat_interleave(torch.arange(len(lengths), dtype=torch.long), lengths)
    
    return batch
