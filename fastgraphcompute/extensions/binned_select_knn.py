import torch
# import fastgraphcompute.extensions
import os.path as osp
from .bin_by_coordinates import bin_by_coordinates
from .index_replacer import index_replacer
from typing import Optional, Tuple

# load the custom extension library
torch.ops.load_library(osp.join(osp.dirname(
    osp.realpath(__file__)), 'binned_knn_ops.so'))

@torch.jit.script
def binned_select_knn(K: int,
                      coords: torch.Tensor,
                      row_splits: torch.Tensor,
                      direction: Optional[torch.Tensor] = None,
                      n_bins: Optional[torch.Tensor] = None,
                      max_bin_dims: int = 3,
                      torch_compatible_indices: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform K-Nearest Neighbors selection using binning with C++ autograd support.

    Args:
        K (int): Number of nearest neighbors.
        coords (torch.Tensor): Input coordinates for points.
        row_splits (torch.Tensor): Row splits following ragged tensor convention.
        direction (torch.Tensor, optional): Direction constraint for neighbors. 0: can only be neighbour, 1: can only have neighbour, 2: neither
        n_bins (torch.Tensor, optional): Number of bins per dimension.
        max_bin_dims (int, optional): Maximum number of bin dimensions.
        torch_compatible_indices (bool, optional): Compatibility flag for PyTorch behavior.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Indices and distances of the nearest neighbors.
    """
    # Validate input coordinates
    if coords.shape[1] == 0:
        raise ValueError(
            "Input coordinates must have at least one dimension. Got 0 dimensions.")
    if max_bin_dims == 0:
        raise ValueError("max_bin_dims must be greater than 0. Got 0.")

    # Type checking for JIT compatibility
    if not isinstance(K, int):
        K = int(K)
    if not isinstance(max_bin_dims, int):
        max_bin_dims = int(max_bin_dims)

    # Automatically adjust max_bin_dims based on coordinate dimensions
    # FGC supports max_bin_dims of 2, 3, 4, or 5 only
    # Limit to min of coordinate dimensions and 5
    coord_dims = coords.shape[1]
    max_bin_dims = min(max_bin_dims, coord_dims, 5)
    max_bin_dims = max(max_bin_dims, 2)  # Ensure at least 2

    # Ensure row_splits is a tensor
    if not isinstance(row_splits, torch.Tensor):
        row_splits = torch.tensor(
            row_splits, dtype=torch.int64, device=coords.device)

    # Ensure coordinates are float32 for CUDA kernel compatibility
    if coords.dtype != torch.float32:
        coords = coords.to(dtype=torch.float32)

    # Convert n_bins to tensor if it's an integer
    if n_bins is not None and not isinstance(n_bins, torch.Tensor):
        n_bins = torch.tensor(n_bins, dtype=torch.int64, device=coords.device)

    # For autograd tensors, use clone().contiguous() to ensure clean contiguous tensors
    if coords.requires_grad:
        coords = coords.clone().contiguous()
    else:
        coords = coords.contiguous()

    row_splits = row_splits.contiguous()

    if direction is not None:
        if direction.requires_grad:
            direction = direction.clone().contiguous()
        else:
            direction = direction.contiguous()

    if n_bins is not None:
        n_bins = n_bins.contiguous()

    row_splits = row_splits.to(dtype=torch.int64, copy=False)

    if n_bins is not None and isinstance(n_bins, torch.Tensor):
        n_bins = n_bins.to(dtype=torch.int64, copy=False)

    # Use the C++ autograd kernel
    idx, dist = torch.ops.fastgraphcompute_custom_ops.binned_select_knn_autograd(
        coords, row_splits, K, direction, n_bins, max_bin_dims, torch_compatible_indices)

    return idx, dist



@torch.jit.script
def torch_binned_select_knn(K: int,
                      coords: torch.Tensor,
                      row_splits: torch.Tensor,
                      direction: Optional[torch.Tensor] = None,
                      n_bins: Optional[torch.Tensor] = None,
                      max_bin_dims: int = 3,
                      torch_compatible_indices: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    Pure PyTorch reference implementation of ``binned_select_knn`` for testing and debugging.
    This version is not optimized and should not be used in production.

    For each batch defined by ``row_splits``, it computes a full pairwise distance matrix
    with ``torch.cdist`` and then uses ``torch.topk`` to select the nearest neighbors.
    No spatial binning is performed in this implementation.

    Arguments kept for API compatibility but ignored by this implementation:
      - ``n_bins``
      - ``max_bin_dims``
      - ``torch_compatible_indices``

    ``direction`` is not supported here and will raise ``NotImplementedError`` if provided.
    The returned neighbors include self references, which can be filtered out later if needed.
    """
    #does not work with direction, raise if set, the others are just ignored
    if direction is not None:
        raise NotImplementedError("Direction constraint is not implemented in the torch version of binned_select_knn.")

    num_verts = coords.shape[0]
    # Pre-allocate outputs: invalid neighbor index = -1, distance = 0
    idx_out = torch.full((num_verts, K), -1, dtype=torch.int64, device=coords.device)
    dist_out = torch.zeros((num_verts, K), dtype=torch.float32, device=coords.device)

    for i in range(len(row_splits) - 1):
        start = int(row_splits[i].item())
        end = int(row_splits[i + 1].item())
        # skip empty segments to avoid topk(k=0) error
        if start >= end:
            continue

        batch_coords = coords[start:end]

        # create a full distance matrix for the batch
        dist_matrix = torch.cdist(batch_coords, batch_coords)
        # get K nearest neighbours (distances and local indices)
        k_eff = min(K, end - start)
        knn_dist, knn_idx = torch.topk(dist_matrix, k_eff, largest=False)

        # make indices global and store into pre-allocated output
        idx_out[start:end, :k_eff] = knn_idx + start
        dist_out[start:end, :k_eff] = knn_dist

    return idx_out, dist_out ** 2


