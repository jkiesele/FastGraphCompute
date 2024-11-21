import torch

#import the custom operations
from . import binned_select_knn
from . import select_with_default

class GravNetOp(torch.nn.Module):
    """
    GravNetOp implements a single layer of the GravNet algorithm [arxiv:1902.07987], which 
    is designed to learn local graph structures based on learned spatial coordinates. 
    It combines both distance-based aggregation and feature propagation using message passing.

    Args:
        in_channels (int): The dimensionality of the input features.
        out_channels (int): The dimensionality of the output features after message passing.
        space_dimensions (int): The dimensionality of the learned spatial coordinates used for neighborhood construction.
        propagate_dimensions (int): The dimensionality of the features propagated between neighbors.
        k (int): The number of nearest neighbors to consider for each point in the learned space.
        output_activation (torch.nn.Module, optional): Activation function applied to the output layer.
                                                        Defaults to ReLU.
        return_more (bool, optional): Whether to return additional information for debugging/monitoring Defaults 
                                       to False. If False, only the output features are returned.
                                       If True, also the indices of the k-nearest neighbors, the distances^2 to them
                                       and the learned coordinates are returned.
        optimization_arguments (dict, optional): Additional arguments for optimizing the k-NN selection.

    Attributes:
        space_transformations (torch.nn.Linear): Linear layer to project input features into a lower-dimensional
                                                 space for neighbor selection.
        propagate_transformations (torch.nn.Linear): Linear layer to transform input features into a space
                                                     for message passing.
        output_transformations (torch.nn.Sequential): Linear and activation layers to produce the final output
                                                      features after message aggregation.
        optimization_arguments (dict): Additional arguments for k-NN optimization.

    Methods:
        forward(x, row_splits): Computes the output of the GravNet layer for a batch of input features.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 space_dimensions,
                 propagate_dimensions,
                 k,
                 output_activation=torch.nn.ReLU(),
                 return_more = False,
                 optimization_arguments: dict = {}):
        
        super(GravNetOp, self).__init__()

        self.k = k

        # Linear layer to project input features into the learned space (B x input_dim -> B x space_dim)
        self.space_transformations = torch.nn.Linear(in_channels, space_dimensions)
        # Linear layer to transform input features into propagation features (B x input_dim -> B x propagate_dimensions)
        self.propagate_transformations = torch.nn.Linear(in_channels, propagate_dimensions)
        # Linear and activation layers for producing the final output
        self.output_transformations = torch.nn.Sequential(
            torch.nn.Linear(in_channels + 2 * propagate_dimensions, out_channels),
            output_activation
        )

        # Store optimization arguments for neighbor selection
        self.optimization_arguments = optimization_arguments

        self.create_output = self._output_pass if return_more else self._output_restrict

    def _output_restrict(self, x, nidx, distsq, coords):
        return x
    
    def _output_pass(self, x, nidx, distsq, coords):
        return x, nidx, distsq, coords


    def forward(self, x, row_splits):
        """
        Forward pass of the GravNet layer.

        Args:
            x (torch.Tensor): Input feature tensor of shape (B, input_dim), where B is the batch size.
            row_splits (torch.Tensor): Tensor indicating the row splits for separate batches.

        Returns:
            torch.Tensor: Output feature tensor of shape (B, output_dim).
        """
        
        # Step 1: Transform input features into learned spatial coordinates
        space = self.space_transformations(x)  # B x space_dim
        
        # Step 2: Transform input features into propagation features
        propagate = self.propagate_transformations(x)  # B x propagate_dimensions
        
        # Step 3: Determine the k-nearest neighbors based on the learned space
        # neighbor_idx: Indices of k-nearest neighbors
        # distsq: Squared distances to k-nearest neighbors
        neighbor_idx, distsq = binned_select_knn(self.k, space, row_splits,
                                                 **self.optimization_arguments)

        # Step 4: Compute weights based on distances (using a Gaussian kernel)
        weights = torch.exp(-10. * distsq)  # B x K

        # Step 5: Aggregate features from neighboring nodes
        # Gather features of the k-nearest neighbors
        propagate = select_with_default(neighbor_idx, propagate, 0.0)  # B x K x propagate_dimensions
        # Expand weights to match the feature dimensions (B x K -> B x K x propagate_dimensions)
        weights = weights.unsqueeze(-1)
        # Apply the weights to the features
        propagate = propagate * weights

        # Step 6: Aggregate features using mean and max pooling
        fmean = torch.mean(propagate, dim=1)  # B x propagate_dimensions
        fmax = torch.max(propagate, dim=1).values  # B x propagate_dimensions

        # Step 7: Concatenate the mean and max pooled features
        output = torch.cat([x, fmean, fmax], dim=1)  # B x (2 * propagate_dimensions)

        # Step 8: Apply final transformation to get output features
        out = self.output_transformations(output)
        return self.create_output(out, neighbor_idx, distsq, space)
    