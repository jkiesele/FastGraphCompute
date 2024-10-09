import torch
import numpy as np
import unittest
import os.path as osp
import ml4reco_modules.extensions

# Load the shared library
cpu_so_file = osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'bin_by_coordinates_cpu.so')
torch.ops.load_library(cpu_so_file)


cuda_so_file = osp.join(osp.dirname(osp.realpath(ml4reco_modules.extensions.__file__)), 'bin_by_coordinates_cuda.so')
torch.ops.load_library(cuda_so_file)

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
        # Updated expected assigned bin based on TensorFlow output
        self.expected_assigned_bin = torch.tensor([
            [0, 0, 0],
            [0, 5, 5],
            [0, 9, 9]
        ], dtype=torch.int32)
        self.row_splits = torch.tensor([0, 3], dtype=torch.int32)  # No actual split in this case, just one segment

    def do_simple_binning(self, cuda=False):
        if not cuda:
            op = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu
            coord = self.coordinates
            rs = self.row_splits
            bin_width = self.bin_width
            nbins = self.nbins
        else:
            op = torch.ops.bin_by_coordinates_cuda.bin_by_coordinates
            coord = self.coordinates.to('cuda')
            rs = self.row_splits.to('cuda')
            bin_width = self.bin_width.to('cuda')
            nbins = self.nbins.to('cuda')

        # Test basic functionality
        output_assigned_bin, output_flat_assigned_bin, output_n_per_bin = op(
            coord, rs, bin_width, nbins, True)

        if cuda:
            output_assigned_bin = output_assigned_bin.to('cpu')
            output_flat_assigned_bin = output_flat_assigned_bin.to('cpu')
            output_n_per_bin = output_n_per_bin.to('cpu')

        expected_flat_assigned_bin = torch.zeros(len(self.expected_assigned_bin), dtype=torch.int32)
        expected_n_per_bin = torch.zeros((100,), dtype=torch.int32)

        for i in range(len(self.expected_assigned_bin)):
            f = self.expected_assigned_bin[i, 1] * 10 + self.expected_assigned_bin[i, 2]
            expected_flat_assigned_bin[i] = f
            expected_n_per_bin[f] += 1

        self.assertTrue(torch.equal(output_assigned_bin, self.expected_assigned_bin))
        self.assertTrue(torch.equal(output_n_per_bin, expected_n_per_bin))
        self.assertTrue(torch.equal(output_flat_assigned_bin, expected_flat_assigned_bin))

    def test_simple_binning_cuda(self):
        self.do_simple_binning(cuda=True)

    def test_simple_binning_cpu(self):
        self.do_simple_binning(cuda=False)

    def do_out_of_bounds(self, cuda=False):
        print("Running out of bounds test. Overflow warnings here are normal!")
        # Ensure coordinates out of bounds are handled properly
        coordinates = torch.tensor([
            [-1.0, -1.0],
            [5.1, 5.1],
            [0.7, 0.7]
        ], dtype=torch.float32)


        if not cuda:
            op = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu
            coord = coordinates
            rs = self.row_splits
            bin_width = self.bin_width
            nbins = self.nbins
        else:
            op = torch.ops.bin_by_coordinates_cuda.bin_by_coordinates
            coord = coordinates.to('cuda')
            rs = self.row_splits.to('cuda')
            bin_width = self.bin_width.to('cuda')
            nbins = self.nbins.to('cuda')


        output_assigned_bin, output_flat_assigned_bin, output_n_per_bin = op(
            coord, rs, bin_width, nbins, True)

        if cuda:
            output_assigned_bin = output_assigned_bin.to('cpu')
            output_flat_assigned_bin = output_flat_assigned_bin.to('cpu')
            output_n_per_bin = output_n_per_bin.to('cpu')

        # Expected all coordinates to set indices to zero as per the C++ logic
        expected_assigned_bin = torch.tensor([
            [0, 0, 0],
            [0, 9, 9],
            [0, 1, 1],
        ], dtype=torch.int32)

        expected_flat_assigned_bin = torch.zeros(len(expected_assigned_bin), dtype=torch.int32)
        expected_n_per_bin = torch.zeros((100,), dtype=torch.int32)

        for i in range(len(expected_assigned_bin)):
            f = expected_assigned_bin[i, 1] * 10 + expected_assigned_bin[i, 2]
            expected_flat_assigned_bin[i] = f
            expected_n_per_bin[f] += 1

        self.assertTrue(torch.equal(output_assigned_bin, expected_assigned_bin))
        self.assertTrue(torch.equal(output_flat_assigned_bin, expected_flat_assigned_bin))
        self.assertTrue(torch.equal(output_n_per_bin, expected_n_per_bin))

    def test_out_of_bounds_cuda(self):
        self.do_out_of_bounds(cuda=True)

    def test_out_of_bounds_cpu(self):
        self.do_out_of_bounds(cuda=False)


    def do_large_scale(self, cuda=False):
        # Test with a larger scale of coordinates
        coordinates = torch.rand((1000, 2)) * 5  # Coordinates in the range [0, 5)
        row_splits = torch.tensor([0, 1000], dtype=torch.int32)  # No actual split in this case, just one segment

        if not cuda:
            op = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu
            coord = coordinates
            rs = row_splits
            bin_width = self.bin_width
            nbins = self.nbins
        else:
            op = torch.ops.bin_by_coordinates_cuda.bin_by_coordinates
            coord = coordinates.to('cuda')
            rs = row_splits.to('cuda')
            bin_width = self.bin_width.to('cuda')
            nbins = self.nbins.to('cuda')


        output_assigned_bin, output_flat_assigned_bin, output_n_per_bin = op(
            coord, rs, bin_width, nbins, True)

        if cuda:
            output_assigned_bin = output_assigned_bin.to('cpu')
            output_flat_assigned_bin = output_flat_assigned_bin.to('cpu')
            output_n_per_bin = output_n_per_bin.to('cpu')

        # Just ensure this runs without error for a basic sanity check
        self.assertEqual(output_assigned_bin.size(0), 1000)
        self.assertEqual(output_flat_assigned_bin.size(0), 1000)
        self.assertTrue((output_n_per_bin.sum() <= 1000).item())  # Sum of indices may be less due to clamping to zero

    def test_large_scale_cuda(self):
        self.do_large_scale(cuda=True)

    def test_large_scale_cpu(self):
        self.do_large_scale(cuda=False)

if __name__ == '__main__':
    unittest.main()