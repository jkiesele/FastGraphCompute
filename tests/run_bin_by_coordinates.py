import torch
import numpy as np
import os.path as osp

# Load the shared libraries
cpu_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'bin_by_coordinates_cpu.so')
torch.ops.load_library(cpu_so_file)

# Define input tensors
# Testing a range of coordinates, including edge cases at the boundaries
coordinates = torch.tensor([
    [0.1, 0.1],  # Expected to fall into the first bin in each dimension
    [2.5, 2.5],  # Mid-range, testing internal binning
    [4.9, 4.9],  # Near the edge, should be in the last bin if nbins and bin_width align correctly
    [-0.1, -0.1], # Out of bounds, should be clamped or handled according to the function's capability
    [5.1, 5.1]   # Exactly on the edge, handling depends on the edge inclusion logic
], dtype=torch.float32)

# Simulate multiple row splits, suggesting different sets of coordinates
row_splits = torch.tensor([0, 3], dtype=torch.int32)  # Three groups: first 2, next 3

bin_width = torch.tensor([0.5], dtype=torch.float32)  # Bin width of 1 unit
nbins = torch.tensor([10, 10], dtype=torch.int32)  # 5 bins in each dimension, covering [0, 5) theoretically

calc_n_per_bin = True

# Call the C++ extension function
output_assigned_bin, output_flat_assigned_bin, output_n_per_bin = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu(
    coordinates, row_splits, bin_width, nbins, calc_n_per_bin)

# Print the outputs
print("Output Assigned Bin:\n", output_assigned_bin)
print("Output Flat Assigned Bin:\n", output_flat_assigned_bin)
print("Output N Per Bin:\n", output_n_per_bin)
