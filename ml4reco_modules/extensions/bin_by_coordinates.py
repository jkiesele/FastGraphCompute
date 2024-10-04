import torch
import ml4reco_modules.extensions
import os.path as osp

#load the custom extension library
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'bin_by_coordinates_cpu.so'))
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'bin_by_coordinates_cuda.so'))

def bin_by_coordinates(coordinates, row_splits, bin_width=None, n_bins=None, calc_n_per_bin=True, pre_normalized=False, name=""):
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

    # Ensure the coordinates are finite
    if not torch.isfinite(coordinates).all():
        raise ValueError(f"BinByCoordinates: input coordinates {name} contain non-finite values")
    
    #select cpu or gpu version based on device of coordinates
    if coordinates.device.type == 'cuda':
        op = torch.ops.bin_by_coordinates_cuda.bin_by_coordinates
    else:
        op = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu

    # Normalize coordinates if not pre-normalized
    if not pre_normalized:
        min_coords = torch.min(coordinates, dim=0, keepdim=True).values
        coordinates = coordinates - min_coords

    # Calculate max coordinates and ensure there's a small range to avoid zero-range bins
    dmax_coords = torch.max(coordinates, dim=0).values
    dmax_coords = torch.where(torch.min(coordinates, dim=0).values == dmax_coords, dmax_coords + 1.0, dmax_coords) + 1e-3
    dmax_coords = torch.where(torch.isfinite(dmax_coords), dmax_coords, torch.tensor(1.0))

    # Ensure that the maximum coordinates are greater than 0
    assert (dmax_coords > 0).all(), "BinByCoordinates: dmax_coords must be greater than zero."

    # Calculate bin_width or n_bins
    if bin_width is None:
        assert n_bins is not None, "Either bin_width or n_bins must be provided."
        # check if n_bins is a tensor if not make it one
        if not isinstance(n_bins, torch.Tensor):
            n_bins = torch.tensor(n_bins, dtype=torch.int32)
        #make sure it has the coordinate dimension
        if n_bins.dim() == 0:
            n_bins = n_bins.repeat(coordinates.shape[1])
        bin_width = dmax_coords / n_bins.to(dtype=torch.float32)
        bin_width = torch.max(bin_width).unsqueeze(-1)  # Ensure uniform bin width across dimensions
    elif n_bins is None:
        assert bin_width is not None, "Either bin_width or n_bins must be provided."
        n_bins = (dmax_coords / bin_width).to(dtype=torch.int32) + 1

    # Ensure that the bin dimensions are valid
    assert (n_bins > 0).all(), "BinByCoordinates: n_bins must be greater than zero."
    assert (bin_width > 0).all(), "BinByCoordinates: bin_width must be greater than zero."

    # Call the custom kernel `bin_by_coordinates_cpu` to assign bins to coordinates
    bin_indices, flat_bin_indices, n_per_bin = op(
        coordinates, row_splits, bin_width, n_bins, calc_n_per_bin
    )

    return bin_indices, flat_bin_indices, n_bins, bin_width, n_per_bin

