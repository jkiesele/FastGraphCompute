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
def _binned_select_knn(
        K: int,
        coordinates: torch.Tensor,
        bin_idx: torch.Tensor,
        dim_bin_idx: torch.Tensor,
        bin_boundaries: torch.Tensor,
        n_bins: torch.Tensor,
        bin_width: torch.Tensor,
        torch_compatible_indices: bool = False,
        direction: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

    # check if direction is None, if so create an empty tensor
    if direction is None:
        direction_input = torch.empty(
            0, device=coordinates.device, dtype=dim_bin_idx.dtype)
    else:
        direction_input = direction

    idx, dist = torch.ops.binned_select_knn.binned_select_knn(
        coordinates, bin_idx, dim_bin_idx, bin_boundaries, n_bins, bin_width,
        direction_input, torch_compatible_indices, direction is not None, K)

    return idx, dist


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
        direction (torch.Tensor, optional): Direction constraint for neighbors.
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
    coord_dims = coords.shape[1]
    if max_bin_dims < coord_dims:
        max_bin_dims = coord_dims

    # Ensure row_splits is a tensor
    if not isinstance(row_splits, torch.Tensor):
        row_splits = torch.tensor(
            row_splits, dtype=torch.int32, device=coords.device)

    # Ensure coordinates are float32 for CUDA kernel compatibility
    if coords.dtype != torch.float32:
        coords = coords.to(dtype=torch.float32)

    # Convert n_bins to tensor if it's an integer
    if n_bins is not None and not isinstance(n_bins, torch.Tensor):
        n_bins = torch.tensor(n_bins, dtype=torch.int32, device=coords.device)

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
    idx, dist = torch.ops.fastgraphcompute_custom_ops.binned_select_knn(
        coords, row_splits, K, direction, n_bins, max_bin_dims, torch_compatible_indices)

    return idx, dist
