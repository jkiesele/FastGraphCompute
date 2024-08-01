import torch

def bin_by_coordinates(coordinates, row_splits, bin_width=None, n_bins=None, calc_n_per_bin=True, pre_normalized=False):
    """
    Assigns bins to coordinates using either specified bin width or number of bins.
    """
    if bin_width is None and n_bins is None:
        raise ValueError("Either bin_width or n_bins must be specified")
    
    if not pre_normalized:
        min_coords = coordinates.min(dim=0, keepdim=True)[0]
        coordinates = coordinates - min_coords
    
    max_coords = coordinates.max(dim=0, keepdim=True)[0]
    max_coords = torch.where((coordinates.min(dim=0, keepdim=True)[0] == max_coords), max_coords + 1.0, max_coords) + 1e-3

    if bin_width is None:
        bin_width = max_coords / n_bins.float()
        bin_width = bin_width.max().unsqueeze(-1)  # Ensure bin width is the same for all dimensions
    
    if n_bins is None:
        n_bins = (max_coords / bin_width).ceil().int()

    # Ensuring conditions for bin calculations
    assert torch.all(n_bins > 0), "Number of bins must be positive"
    assert torch.all(bin_width > 0), "Bin width must be positive"

    # Compute bin indices for each coordinate
    bin_indices = (coordinates / bin_width).int()
    flat_bin_indices = bin_indices.matmul(n_bins.cumprod(dim=0)[:-1])
    
    # Optionally compute the number of points per bin
    if calc_n_per_bin:
        n_per_bin = torch.zeros(n_bins.prod(), dtype=torch.int32)
        indices = flat_bin_indices + row_splits[:-1].unsqueeze(1) * n_bins.prod()
        n_per_bin.put_(indices, torch.ones_like(indices), accumulate=True)
        return bin_indices, flat_bin_indices, n_bins, bin_width, n_per_bin
    
    return bin_indices, flat_bin_indices, n_bins, bin_width
