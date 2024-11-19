import torch
import fastgraphcompute.extensions
import os.path as osp
from .bin_by_coordinates import bin_by_coordinates
from .index_replacer import index_replacer

#load the custom extension library
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(fastgraphcompute.extensions.__file__)), 'binned_select_knn_cpu.so'))
if torch.cuda.is_available():
    torch.ops.load_library(osp.join(osp.dirname(osp.realpath(fastgraphcompute.extensions.__file__)), 'binned_select_knn_cuda.so'))

#load the gradient library
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(fastgraphcompute.extensions.__file__)), 'binned_select_knn_grad_cpu.so'))
if torch.cuda.is_available():
    torch.ops.load_library(osp.join(osp.dirname(osp.realpath(fastgraphcompute.extensions.__file__)), 'binned_select_knn_grad_cuda.so'))

#just a wrapper function to call the custom extension
def _binned_select_knn(
    K: int,
    coordinates: torch.Tensor,
    bin_idx: torch.Tensor,
    dim_bin_idx: torch.Tensor,
    bin_boundaries: torch.Tensor,
    n_bins, 
    bin_width , 
    torch_compatible_indices=False,
    direction = None):

    if coordinates.device.type == 'cuda':
        op = torch.ops.binned_select_knn_cuda.binned_select_knn_cuda
    else:
        op = torch.ops.binned_select_knn_cpu.binned_select_knn_cpu

    #check if direction is None, if so create an empty tensor
    if direction is None:
        direction_input = torch.empty(0, device=coordinates.device, dtype=dim_bin_idx.dtype)
    else:
        direction_input = direction
        
    #this can possibly be removed for deployment    
    def assert_same_dtype(*tensors):
        dtypes = [tensor.dtype for tensor in tensors]
        assert all(dtype == dtypes[0] for dtype in dtypes), f"Mismatch in dtypes: {dtypes}"
    assert_same_dtype(bin_idx, dim_bin_idx, bin_boundaries, n_bins, direction_input)
    assert_same_dtype(coordinates, bin_width)

    idx, dist = op(coordinates, bin_idx, dim_bin_idx, bin_boundaries, n_bins, bin_width,
              direction_input, torch_compatible_indices, direction is not None, K)

    return idx, dist


class _BinnedKNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                coords: torch.Tensor,
                row_splits: torch.Tensor,
                K: int, 
                direction=None, 
                n_bins=None, 
                max_bin_dims: int = 3, 
                torch_compatible_indices=False):
        
        # Estimate a good number of bins for homogeneous distributions
        elems_per_rs = torch.max(row_splits) / row_splits.shape[0]
        elems_per_rs = elems_per_rs.to(dtype=torch.int32) + 1
    
        # Limit max_bin_dims to the number of coordinate dimensions
        max_bin_dims = min(max_bin_dims, coords.shape[1])
    
        # Calculate n_bins if not provided
        if n_bins is None:
            n_bins = torch.pow(elems_per_rs.float() / (K / 32), 1. / float(max_bin_dims))
            n_bins = n_bins.to(dtype=torch.int32)
            n_bins = torch.where(n_bins < 5, torch.tensor(5, dtype=torch.int32), n_bins)
            n_bins = torch.where(n_bins > 30, torch.tensor(30, dtype=torch.int32), n_bins)
    
        # Handle binning for the coordinates
        bin_coords = coords
        if bin_coords.shape[-1] > max_bin_dims:
            bin_coords = bin_coords[:, :max_bin_dims]  # Truncate the extra dimensions
    
        # Call BinByCoordinates to assign bins
        dbinning, binning, nb, bin_width, nper = bin_by_coordinates(bin_coords, row_splits, n_bins=n_bins)
    
        # Sort the points by bin assignment
        sorting = torch.argsort(binning, dim=0)
        #cast sorting to int32
        sorting = sorting.to(dtype=torch.int32)
    
        # Gather sorted coordinates and bin information
        scoords = coords[sorting]
        sbinning = binning[sorting]
        sdbinning = dbinning[sorting]
    
        if direction is not None:
            direction = direction[sorting]
    
        # Create bin boundaries (cumulative sum of number of points per bin)
        bin_boundaries = torch.cat([torch.zeros(1, dtype=torch.int32).to(coords.device), nper], dim=0)
        bin_boundaries = torch.cumsum(bin_boundaries, dim=0, dtype=torch.int32)
    
        # Ensure the bin boundaries are valid
        assert torch.max(bin_boundaries) == torch.max(row_splits), "Bin boundaries do not match row splits."
    
        # Call the _BinnedSelectKnn kernel
        idx, dist = _binned_select_knn(K, scoords, sbinning, sdbinning, bin_boundaries=bin_boundaries, 
                                     n_bins=nb, bin_width=bin_width, 
                                     torch_compatible_indices=torch_compatible_indices, direction=direction)
        
        idx = index_replacer(idx, sorting)  # Placeholder: handle index sorting replacement
        #cast sorting back to int64 for scatter
        sorting = sorting.to(dtype=torch.int64) 
        
        dist = torch.scatter(dist, 0, sorting.unsqueeze(-1).expand_as(dist), dist)
        idx = torch.scatter(idx, 0, sorting.unsqueeze(-1).expand_as(idx), idx)

        ctx.save_for_backward(idx, dist, coords)
        
        return idx, dist
        
    @staticmethod
    def backward(ctx, grad_idx, grad_dist):
        # Retrieve saved tensors from forward pass
        idx, dist, coords = ctx.saved_tensors

        if grad_dist.device.type == 'cuda':
            op = torch.ops.binned_select_knn_grad_cuda.binned_select_knn_grad_cuda
        else:
            op = torch.ops.binned_select_knn_grad_cpu.binned_select_knn_grad_cpu

        # Call your custom operation for computing coordinate gradients
        grad_coordinates = op(grad_dist, idx, dist, coords)
        #print all names and shapes
        
        # Return gradients for each input; return None for inputs that don't require gradients
        return grad_coordinates, None, None, None, None, None, None  # None for other options if not differentiable



def binned_select_knn(K: int, 
                      coords: torch.Tensor,
                      row_splits: torch.Tensor,
                      direction=None, 
                      n_bins=None, 
                      max_bin_dims: int = 3, 
                      torch_compatible_indices=False):
    """
    Perform K-Nearest Neighbors selection using binning.

    Args:
        K (int): Number of nearest neighbors.
        coords (torch.Tensor): Input coordinates for points.
        row_splits (torch.Tensor): Row splits following ragged tensor convention.
        direction (torch.Tensor, optional): Direction constraint for neighbors.
        n_bins (torch.Tensor, optional): Number of bins per dimension.
        max_bin_dims (int, optional): Maximum number of bin dimensions.
        tf_compatible (bool, optional): Compatibility flag for TensorFlow behavior.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Indices and distances of the nearest neighbors.
    """
    return _BinnedKNNFunction.apply(coords, row_splits, K, direction, n_bins, max_bin_dims, torch_compatible_indices)
    

