import torch
import numpy as np
import unittest
import os.path as osp
import ml4reco_modules.extensions

# Load the shared library
cpu_so_file = osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'bin_by_coordinates_cpu.so')
torch.ops.load_library(cpu_so_file)

class TestBinByCoordinates(unittest.TestCase):
    def setUp(self):
        # Define your bin widths and number of bins per dimension
        self.bin_width = torch.tensor([0.5], dtype=torch.float32)  # Example: 0.5 units per bin
        self.nbins = torch.tensor([10, 10], dtype=torch.int32)  # Example: 10x10 grid
        self.coordinates = torch.tensor([
            [0.1, 0.1],
            [2.5, 2.5],
            [4.9, 4.9]
        ], dtype=torch.float32)
        self.row_splits = torch.tensor([0, 3], dtype=torch.int32)  # No actual split in this case, just one segment

    def test_simple_binning(self):

        # Test basic functionality
        output_assigned_bin, output_flat_assigned_bin, output_n_per_bin = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu(
            self.coordinates, self.row_splits, self.bin_width, self.nbins, True)

        # Updated expected assigned bin based on TensorFlow output
        expected_assigned_bin = torch.tensor([
            [0, 0],
            [5, 5],
            [9, 9]
        ], dtype=torch.int32)

        expected_flat_assigned_bin = torch.zeros(len(expected_assigned_bin), dtype=torch.int32)
        expected_n_per_bin = torch.zeros((100,), dtype=torch.int32)

        for i in range(len(expected_assigned_bin)):
            f = expected_assigned_bin[i,0]*10 + expected_assigned_bin[i,1]
            expected_flat_assigned_bin[i] = f
            expected_n_per_bin[f] += 1

        self.assertTrue(torch.equal(output_assigned_bin, expected_assigned_bin))
        self.assertTrue(torch.equal(output_n_per_bin, expected_n_per_bin))
        self.assertTrue(torch.equal(output_flat_assigned_bin, expected_flat_assigned_bin))

    def test_out_of_bounds(self):
        # Ensure coordinates out of bounds are handled properly
        coordinates = torch.tensor([
            [-1.0, -1.0],
            [5.1, 5.1]
        ], dtype=torch.float32)
        output_assigned_bin, output_flat_assigned_bin, output_n_per_bin = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu(
            coordinates, self.row_splits, self.bin_width, self.nbins, True)

        # Expected all coordinates to set indices to zero as per the C++ logic
        expected_assigned_bin = torch.tensor([
            [0, 0],
            [9, 9]
        ], dtype=torch.int32)

        expected_flat_assigned_bin = torch.zeros(len(expected_assigned_bin), dtype=torch.int32)
        expected_n_per_bin = torch.zeros((100,), dtype=torch.int32)

        for i in range(len(expected_assigned_bin)):
            f = expected_assigned_bin[i,0]*10 + expected_assigned_bin[i,1]
            expected_flat_assigned_bin[i] = f
            expected_n_per_bin[f] += 1

        self.assertTrue(torch.equal(output_assigned_bin, expected_assigned_bin))
        self.assertTrue(torch.equal(output_flat_assigned_bin, expected_flat_assigned_bin))
        self.assertTrue(torch.equal(output_n_per_bin, expected_n_per_bin))

    def test_large_scale(self):
        # Test with a larger scale of coordinates
        coordinates = torch.rand((1000, 2)) * 5  # Coordinates in the range [0, 5)
        output_assigned_bin, output_flat_assigned_bin, output_n_per_bin = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu(
            coordinates, self.row_splits, self.bin_width, self.nbins, True)

        # Just ensure this runs without error for a basic sanity check
        self.assertEqual(output_assigned_bin.size(0), 1000)
        self.assertEqual(output_flat_assigned_bin.size(0), 1000)
        self.assertTrue((output_n_per_bin.sum() <= 1000).item())  # Sum of indices may be less due to clamping to zero

if __name__ == '__main__':
    unittest.main()
