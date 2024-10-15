import torch

#import the custom operations
from . import binned_select_knn
from . import select_with_default

# define a gravnet layer with a single layer of message passing following
# the gravnet algorithm: https://arxiv.org/abs/1902.07987

class GravNetOp(torch.nn.Module):
    def __init__(self, input_dim, 
                 output_dim, 
                 space_dim, 
                 propagate_dimensions, 
                 k=16,
                 optimization_arguments : dict = {}):
        
        super(GravNetOp, self).__init__()

        assert 2 * propagate_dimensions == output_dim, "GravNetOp always returns mean and max over the neighbors w.r.t their propagate_dimensions."

        self.input_dim = input_dim
        self.space_dim = space_dim
        self.propagate_dimensions = propagate_dimensions
        self.k = k

        # define the two linear transformations
        self.space_transformations = torch.nn.Linear(input_dim, space_dim)
        self.propagate_transformations = torch.nn.Linear(input_dim, propagate_dimensions)

        self.optimization_arguments = optimization_arguments

    def forward(self, x, row_splits):
        
        # apply the transformations
        space = self.space_transformations(x) # B x S
        propagate = self.propagate_transformations(x) # B x FLR
        
        # compute the neighbors and distances
        neighbor_idx, distsq = binned_select_knn(self.k, space, row_splits,
                                                 **self.optimization_arguments) # to be optimised

        # compute the weights
        weights = torch.exp(-10. * distsq) # B x K

        # compute the weighted neighbour features, needs expansion to B x K x FLR
        propagate = select_with_default(neighbor_idx, propagate , 0.0) # B x K x FLR
        weights = weights.unsqueeze(-1)
        propagate = propagate * weights

        fmean = torch.mean(propagate, dim=1) # B x K x FLR -> B x FLR
        fmax = torch.max(propagate, dim=1).values # B x K x FLR -> B x FLR

        # compute the output
        output = torch.cat([fmean, fmax], dim=1)
        return output