import torch
import numpy as np
import os.path as osp
import os
import sys

# Load the shared libraries
cpu_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'bin_by_coordinates_cpu.so')
torch.ops.load_library(cpu_so_file)

def run_bin_by_coordinates():
    # Define inputs
    coordinates = torch.tensor([
        [0.1, 0.1],   # Normal case
        [2.5, 2.5],   # Normal case
        [4.9, 4.9],   # On the edge
        [-1.0, -1.0], # Out of bounds (below minimum)
        [5.1, 5.1],   # Out of bounds (above maximum)
        [0.0, 0.0],   # On the edge (lower edge)
        [5.0, 5.0]    # On the edge (upper edge, assuming bin width doesn't exactly fit)
    ], dtype=torch.float32)
    
    row_splits = torch.tensor([0, 3, 7], dtype=torch.int32)  # Adjusted to include all points
    bin_width = torch.tensor([0.5], dtype=torch.float32)
    nbins = torch.tensor([10, 10], dtype=torch.int32)
    calc_n_per_bin = True

    # Call the custom operation
    assigned_bin, flat_assigned_bin, n_per_bin = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu(
        coordinates, row_splits, bin_width, nbins, calc_n_per_bin)

    # Print outputs
    print("Assigned Bin:\n", assigned_bin)
    print("Flat Assigned Bin:\n", flat_assigned_bin)
    print("Number of points per bin:\n", n_per_bin)

if __name__ == "__main__":
    run_bin_by_coordinates()
