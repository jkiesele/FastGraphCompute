import torch

# Load the shared library (make sure this path is correct for your environment)
_bin_by_coordinates = torch.ops.load_library('bin_by_coordinates_cpu.so')

def BinByCoordinates(coordinates, row_splits, bin_width=None, n_bins=None, calc_n_per_bin=True, pre_normalized=False):
    """
    Assigns bins to coordinates.

    Parameters:
    - coordinates: torch.Tensor(float32), the coordinates per input point
    - row_splits: torch.Tensor(int32), row splits following the PyTorch convention
    - bin_width: torch.Tensor(float32) / None, will be the same for all dimensions (either bin_width or n_bins must be specified)
    - n_bins: torch.Tensor(int32) / None, the maximum number of bins in any dimension (either bin_width or n_bins must be specified)
    - calc_n_per_bin: bool, calculates the number of points per bin and returns it
    - pre_normalized: bool, whether the coordinates are already normalized

    Returns:
    - bin indices (dim = [rs] + dim(coordinates)). The first index constitutes the row split index
    - bin indices (the above) flattened
    - number of bins used per dimension (dim = dim(coordinates))
    - bin width used (dim = 1)
    - (opt) number of points per bin (dim = 1)
    """

    # Ensure that either bin_width or n_bins is provided
    if bin_width is None and n_bins is None:
        raise ValueError("Either bin_width or n_bins must be specified")

    if not pre_normalized:
        min_coords = coordinates.min(dim=0, keepdim=True)[0]
        coordinates = coordinates - min_coords

    dmax_coords = coordinates.max(dim=0)[0]
    dmax_coords = torch.where((coordinates.min(dim=0)[0] == dmax_coords), dmax_coords + 1.0, dmax_coords) + 1e-3

    dmax_coords = torch.where(torch.isfinite(dmax_coords), dmax_coords, torch.tensor(1.0, dtype=dmax_coords.dtype))

    if bin_width is None:
        assert n_bins is not None
        bin_width = dmax_coords / n_bins.float()
        n_bins = None  # Recalculate in dimensions
        bin_width = bin_width.max().unsqueeze(-1)  # Ensure bin width is the same for all dimensions

    if n_bins is None:
        assert bin_width is not None
        n_bins = (dmax_coords / bin_width).ceil().int()
        n_bins += 1

    assert torch.all(n_bins > 0), "Number of bins must be positive"
    assert torch.all(bin_width > 0), "Bin width must be positive"

    # Call the C++ extension
    if calc_n_per_bin:
        binass, flatbinass, nperbin = _bin_by_coordinates.bin_by_coordinates_cpu(
            coordinates, row_splits, bin_width, n_bins, calc_n_per_bin
        )
        return binass, flatbinass, n_bins, bin_width, nperbin
    else:
        binass, flatbinass, _, _ = _bin_by_coordinates.bin_by_coordinates_cpu(
            coordinates, row_splits, bin_width, n_bins, calc_n_per_bin
        )
        return binass, flatbinass, n_bins, bin_width
