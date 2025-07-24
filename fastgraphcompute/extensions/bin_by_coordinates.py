import torch
# import fastgraphcompute.extensions
import os.path as osp
from typing import Tuple

# load the custom extension library
torch.ops.load_library(osp.join(osp.dirname(
    osp.realpath(__file__)), 'binned_knn_ops.so'))


def bin_by_coordinates(coordinates: torch.Tensor, row_splits: torch.Tensor, bin_width: torch.Tensor = None, n_bins: torch.Tensor = None, calc_n_per_bin: bool = True, pre_normalized: bool = False, name: str = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assign bins to coordinates.

    Args:
        coordinates (torch.Tensor): Coordinates per input point (shape: [n_vert, n_coords]).
        row_splits (torch.Tensor): Row splits following the ragged tensor convention.
        bin_width (torch.Tensor, optional): Bin width, used if `n_bins` is not specified.
        n_bins (torch.Tensor, optional): Maximum number of bins per dimension, used if `bin_width` is not specified.
        calc_n_per_bin (bool, optional): If True, calculates the number of points per bin and returns it.
        pre_normalized (bool, optional): If True, assumes coordinates are already normalized.
        name (str, optional): Name for debugging purposes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        Bin indices, flat bin indices, number of bins, bin width, and optionally number of points per bin.
    """

    original_device = coordinates.device

    # Ensure the coordinates are finite - use torch.jit.is_scripting to handle differently in script mode
    if torch.jit.is_scripting():
        pass  # Skip this check in script mode
    else:
        if not torch.isfinite(coordinates).all():
            raise ValueError(
                f"BinByCoordinates: input coordinates {name} contain non-finite values")

    # Create a copy of coordinates for modification
    coords = coordinates.clone()

    # Normalize coordinates if not pre-normalized
    if not pre_normalized:
        min_coords = torch.min(coords, dim=0, keepdim=True).values
        coords = coords - min_coords

    # Calculate max coordinates and ensure there's a small range to avoid zero-range bins
    dmax_coords = torch.max(coords, dim=0).values
    min_coords_per_dim = torch.min(coords, dim=0).values

    # Handle zero-range dimensions - add 1.0 where min == max
    dmax_coords = torch.where(min_coords_per_dim ==
                              dmax_coords, dmax_coords + 1.0, dmax_coords)

    # Add a small epsilon to avoid boundary issues
    dmax_coords = dmax_coords + 1e-3

    # Replace non-finite values with 1.0
    ones = torch.ones_like(dmax_coords)
    dmax_coords = torch.where(torch.isfinite(dmax_coords), dmax_coords, ones)

    # Ensure that the maximum coordinates are greater than 0
    if torch.jit.is_scripting():
        pass  # Skip this check in script mode
    else:
        if not (dmax_coords > 0).all():
            raise ValueError(
                "BinByCoordinates: dmax_coords must be greater than zero.")

    # Calculate bin_width or n_bins
    if bin_width is None:
        # n_bins must be provided if bin_width is None
        if n_bins is None:
            raise ValueError("Either bin_width or n_bins must be provided.")

        # Convert n_bins to tensor if it's not already
        if not torch.is_tensor(n_bins):
            n_bins = torch.tensor(
                n_bins, dtype=torch.int64, device=coords.device)

        # Make sure it has the coordinate dimension
        if n_bins.dim() == 0:
            n_bins = n_bins.repeat(coords.shape[1])

        # Calculate bin_width from n_bins
        bin_width = dmax_coords / n_bins.to(dtype=torch.float32)
        # Ensure uniform bin width across dimensions
        bin_width = torch.max(bin_width).unsqueeze(-1)
    else:
        # Calculate n_bins from bin_width
        if n_bins is None:
            n_bins = (dmax_coords / bin_width).to(dtype=torch.int64) + 1

    # Ensure that the bin dimensions are valid
    if torch.jit.is_scripting():
        pass  # Skip these checks in script mode
    else:
        if not (n_bins > 0).all():
            raise ValueError(
                "BinByCoordinates: n_bins must be greater than zero.")
        if not (bin_width > 0).all():
            raise ValueError(
                "BinByCoordinates: bin_width must be greater than zero.")

    # unified library call to bin_by_coordinates
    bin_indices, flat_bin_indices, n_bins_out, bin_width_out, n_per_bin = torch.ops.bin_by_coordinates.bin_by_coordinates(
        coords, row_splits.to(
            dtype=torch.int64), bin_width, n_bins, calc_n_per_bin, pre_normalized
    )

    return bin_indices.to(original_device), flat_bin_indices.to(original_device), n_bins_out.to(original_device), bin_width_out.to(original_device), n_per_bin.to(original_device)
