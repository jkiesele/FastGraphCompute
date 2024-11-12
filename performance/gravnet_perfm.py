import pickle
import time

import argh
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn  # Assuming GravNetOp is available in torch_geometric
from torch import optim

from fastgraphcompute.gnn_ops import GravNetOp

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn


class GravNetModel(nn.Module):
    def __init__(self, in_dim, prop_dim, space_dim, k, hidden_dim, output_dim, device, num_gravnet_layers=4,
                 the_op=None):
        super(GravNetModel, self).__init__()

        # Define two dense layers at the beginning
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if the_op is None:
            the_op = pyg_nn.GravNetConv
            self.use_rs = False
        else:
            self.use_rs = True

        # Save the number of GravNet layers
        self.num_gravnet_layers = num_gravnet_layers

        # Create GravNet layers and dense layers after each GravNet layer
        self.gravnet_layers = nn.ModuleList()
        self.dense_after_gravnet = nn.ModuleList()

        for _ in range(self.num_gravnet_layers):
            # Add a GravNet layer
            self.gravnet_layers.append(
                the_op(
                    in_channels=hidden_dim if _ == 0 else 2 * prop_dim,
                    out_channels=2 * prop_dim,
                    space_dimensions=space_dim,
                    propagate_dimensions=prop_dim,
                    k=k
                )
            )

            # Add 3 dense layers after each GravNet layer
            self.dense_after_gravnet.append(
                nn.Sequential(
                    nn.Linear(2 * prop_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2 * prop_dim),
                    nn.ReLU()
                )
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

        # Pass through the GravNet layers followed by dense layers
        for i in range(self.num_gravnet_layers):
            if self.use_rs:
                x = self.gravnet_layers[i](x, row_splits)
            else:
                x = self.gravnet_layers[i](x)

            # Pass through the dense layers after the GravNet layer
            x = self.dense_after_gravnet[i](x)

        # Pass through the final dense layers
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer, no activation if using a loss function directly afterward

        return x


def main(no_grad=False, reuse=False):
    # assert use_op in {'torch-geometric', 'custom'}
    n_nodes_test = [5000, 10000, 50000, 100000, 200000]

    results = {

    }

    if not reuse:
        for n_nodes in n_nodes_test:
            for use_op in ['torch-geometric', 'custom']:
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


                if not no_grad:
                    criterion = nn.MSELoss()  # Example: Mean Squared Error loss
                    optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

                timing_array = []
                # Example input data
                for i in range(5):
                    x = torch.randn(n_nodes, in_dim).to(device)
                    row_splits = torch.tensor([0, n_nodes], dtype=torch.int32).to(device)
                    # Forward pass
                    t1 = time.time()
                    if no_grad:
                        with torch.no_grad():
                            output = model(x, row_splits)
                    else:
                        output = model(x, row_splits)
                        loss = torch.sum(output)*0.0
                        loss.backward()
                        optimizer.step()

                    print(torch.sum(output)) # THIS IS IMPORTANT | don't remove it. Otherwise the lazy execution will mess up the time estimate
                    it_took = time.time() - t1
                    timing_array.append(it_took)
                    print("Took", it_took)


                if use_op not in results.keys():
                    results[use_op] = {}
                if n_nodes not in results[use_op].keys():
                    results[use_op][n_nodes] = np.mean(timing_array[1:])
        pickle.dump(results, open('gravnet_perf%s.pkl'%('_no_grad' if no_grad else ''), 'wb'))

    results = pickle.load(open('gravnet_perf%s.pkl'%('_no_grad' if no_grad else ''), 'rb'))

    # Create a single figure and axis (1 row, 1 column)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the results for Model A
    ax.plot(n_nodes_test, list(results["torch-geometric"].values()), marker='o', label="torch-geometric")

    # Plot the results for Model B
    ax.plot(n_nodes_test, list(results["custom"].values()), marker='s', label="custom", color='orange')

    # Set titles and labels
    ax.set_title('Timing Comparison')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Time (seconds)')

    # Add grid
    ax.grid(True)

    # Add legend to differentiate between models
    ax.legend()

    # Adjust layout to make the plot look clean
    plt.tight_layout()
    plt.savefig('../plots/gravnet_time_perf%s.pdf'%('_no_grad' if no_grad else ''))
    plt.close(fig)


if __name__ == '__main__':
    # Use argh to expose the main function to the command line
    argh.dispatch_command(main)
