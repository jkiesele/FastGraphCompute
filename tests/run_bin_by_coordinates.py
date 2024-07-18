import torch
import numpy as np
import os.path as osp

# Load the shared libraries
cpu_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'bin_by_coordinates_cpu.so')
torch.ops.load_library(cpu_so_file)

# Define input tensors
coordinates = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
row_splits = torch.tensor([0, 1, 2], dtype=torch.int32)
bin_width = torch.tensor([1.0], dtype=torch.float32)
nbins = torch.tensor([4, 4], dtype=torch.int32)
calc_n_per_bin = True

# Call the C++ extension function
output_assigned_bin, output_flat_assigned_bin, output_n_per_bin = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu(
    coordinates, row_splits, bin_width, nbins, calc_n_per_bin)

# Print the outputs
print("Output Assigned Bin:\n", output_assigned_bin)
print("Output Flat Assigned Bin:\n", output_flat_assigned_bin)
print("Output N Per Bin:\n", output_n_per_bin)
