import time

import argh
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn  # Assuming GravNetOp is available in torch_geometric
from ml4reco_modules.gnn_ops import GravNetOp

class GravNetModel(nn.Module):
    def __init__(self, in_dim, prop_dim, space_dim, k, hidden_dim, output_dim, device, the_op=None):
        super(GravNetModel, self).__init__()

        # Define two dense layers at the beginning
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if the_op is None:
            the_op = pyg_nn.GravNetConv
            self.use_rs = False
        else:
            self.use_rs = True

        # Define four GravNet layers in the middle
        self.gravnet1 = the_op(
            in_channels=hidden_dim,
            out_channels=2 * prop_dim,
            space_dimensions=space_dim,
            propagate_dimensions=prop_dim,
            k=k
        )
        self.gravnet2 = the_op(
            in_channels=2 * prop_dim,
            out_channels=2 * prop_dim,
            space_dimensions=space_dim,
            propagate_dimensions=prop_dim,
            k=k
        )
        self.gravnet3 = the_op(
            in_channels=2 * prop_dim,
            out_channels=2 * prop_dim,
            space_dimensions=space_dim,
            propagate_dimensions=prop_dim,
            k=k
        )
        self.gravnet4 = the_op(
            in_channels=2 * prop_dim,
            out_channels=2 * prop_dim,
            space_dimensions=space_dim,
            propagate_dimensions=prop_dim,
            k=k
        )

        # Define two dense layers at the end
        self.fc3 = nn.Linear(2 * prop_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        # Define activation function (ReLU here, but you can choose others)
        self.relu = nn.ReLU()

        # Move all GravNet operations to the specified device
        self.to(device)

    def forward(self, x, row_splits=None):
        # Pass through the initial dense layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # Pass through the four GravNet layers
        if self.use_rs:
            x = self.gravnet1(x, row_splits)
            x = self.gravnet2(x, row_splits)
            x = self.gravnet3(x, row_splits)
            x = self.gravnet4(x, row_splits)
        else:
            x = self.gravnet1(x)
            x = self.gravnet2(x)
            x = self.gravnet3(x)
            x = self.gravnet4(x)

        # Pass through the final dense layers
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer, no activation if using a loss function directly afterward

        return x



def main(use_op='torch-geometric', no_grad=False):
    assert use_op in {'torch-geometric', 'custom'}

    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = 64       # Input dimension
    prop_dim = 32     # Propagation dimension
    space_dim = 4     # Space dimension
    k = 64            # Number of neighbors to consider
    hidden_dim = 128  # Dimension for the dense layers
    output_dim = 10   # Output dimension (you can adjust according to the task)


    my_op = None
    if use_op == 'custom':
        my_op = GravNetOp

    # print(device)
    # 0/0

    # Initialize the model
    model = GravNetModel(in_dim, prop_dim, space_dim, k, hidden_dim, output_dim, device, the_op=my_op).to(device)

    n_nodes = 200000

    # Example input data
    for i in range(100):
        x = torch.randn(n_nodes, in_dim).to(device)
        row_splits = torch.tensor([0, n_nodes], dtype=torch.int32).to(device)
        # Forward pass
        t1 = time.time()
        if no_grad:
            with torch.no_grad():
                output = model(x, row_splits)
        else:
            output = model(x, row_splits)
        print(torch.sum(output)) # THIS IS IMPORTANT | don't remove it. Otherwise the lazy execution will mess up the time estimate
        print("Took", time.time() - t1)


if __name__ == '__main__':
    # Use argh to expose the main function to the command line
    argh.dispatch_command(main)
