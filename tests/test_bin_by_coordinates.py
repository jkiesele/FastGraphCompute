import torch
import numpy as np
import unittest
import os.path as osp
import fastgraphcompute.extensions
from fastgraphcompute import bin_by_coordinates

# Load the shared library
cpu_so_file = osp.join(osp.dirname(osp.realpath(fastgraphcompute.extensions.__file__)), 'bin_by_coordinates_cpu.so')
torch.ops.load_library(cpu_so_file)

if torch.cuda.is_available():
    cuda_so_file = osp.join(osp.dirname(osp.realpath(fastgraphcompute.extensions.__file__)), 'bin_by_coordinates_cuda.so')
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

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
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
            [0, 9, 9],
            [0, 9, 9],
            [0, 1, 1],
        ], dtype=torch.int32)

        expected_flat_assigned_bin = torch.zeros(len(expected_assigned_bin), dtype=torch.int32)
        expected_n_per_bin = torch.zeros((100,), dtype=torch.int32)

        for i in range(len(expected_assigned_bin)):
            f = expected_assigned_bin[i, 1] * 10 + expected_assigned_bin[i, 2]
            expected_flat_assigned_bin[i] = f
            expected_n_per_bin[f] += 1

        self.assertTrue(torch.equal(output_assigned_bin, expected_assigned_bin), f"Expected assigned bin: {expected_assigned_bin}, Got: {output_assigned_bin}")
        self.assertTrue(torch.equal(output_flat_assigned_bin, expected_flat_assigned_bin), f"Expected flat assigned bin: {expected_flat_assigned_bin}, Got: {output_flat_assigned_bin}")
        self.assertTrue(torch.equal(output_n_per_bin, expected_n_per_bin), f"Expected n per bin: {expected_n_per_bin}, Got: {output_n_per_bin}")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_out_of_bounds_cuda(self):
        self.do_out_of_bounds(cuda=True)

    def test_out_of_bounds_cpu(self):
        self.do_out_of_bounds(cuda=False)

    
    def calc_batch_index_from_rs(self, row_splits):

        # Ensure row_splits is int64 for tensor indexing
        row_splits = row_splits.to(torch.int64)
    
        # Compute lengths of each batch (difference between consecutive row splits)
        lengths = row_splits[1:] - row_splits[:-1]
        
        # Use repeat_interleave to assign batch indices
        return torch.repeat_interleave(torch.arange(len(lengths), dtype=torch.long), lengths)


    def do_large_scale(self, cuda=False, ndims=2, scaling = 5.):
        # Test with a larger scale of coordinates
        coordinates = torch.rand((1000, ndims)) * scaling  # Coordinates in the range [0, 5]
        # extend bin width and nbins
        bin_width = torch.tensor([0.5], dtype=torch.float32)  # Example: 0.5 units per bin
        nbins = torch.tensor([10]*ndims, dtype=torch.int32)  # Example: 10x10 grid
        row_splits = torch.tensor([0, 300, 700, 1000], dtype=torch.int32) 


        if not cuda:
            op = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu
            coord = coordinates
            rs = row_splits
        else:
            op = torch.ops.bin_by_coordinates_cuda.bin_by_coordinates
            coord = coordinates.to('cuda')
            rs = row_splits.to('cuda')
            bin_width = bin_width.to('cuda')
            nbins = nbins.to('cuda')


        output_assigned_bin, output_flat_assigned_bin, output_n_per_bin = op(
            coord, rs, bin_width, nbins, True)
        
        # sanity check. e.g. the entry in the first dimension of output_assigned_bin should correspond to the 
        # row split index that entry is in
        rs_index = self.calc_batch_index_from_rs(row_splits)
        rs_index = rs_index.to(torch.int32)
        rs_index = rs_index.to(coord.device)
        self.assertTrue(torch.all(output_assigned_bin[:, 0] == rs_index), f"Expected: {rs_index}, Got: {output_assigned_bin[:, 0]}")

        if cuda:
            output_assigned_bin = output_assigned_bin.to('cpu')
            output_flat_assigned_bin = output_flat_assigned_bin.to('cpu')
            output_n_per_bin = output_n_per_bin.to('cpu')

        #check if all the points are within the bounds
        self.assertTrue(torch.sum(output_n_per_bin) == row_splits[-1], "Not all points were assigned a bin, expected %d, got %d" % (row_splits[-1], torch.sum(output_n_per_bin)))

        # Just ensure this runs without error for a basic sanity check
        self.assertEqual(output_assigned_bin.size(0), 1000)
        self.assertEqual(output_flat_assigned_bin.size(0), 1000)
        self.assertTrue((output_n_per_bin.sum() <= 1000).item())  # Sum of indices may be less due to clamping to zero

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_large_scale_cuda(self):
        self.do_large_scale(cuda=True)

    def test_large_scale_cpu(self):
        self.do_large_scale(cuda=False)


    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_large_scale_large_spread_cuda(self):
        self.do_large_scale(cuda=True, scaling=1000.)

    def test_large_scale_large_spread_cpu(self):
        self.do_large_scale(cuda=False, scaling=1000.)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_large_scale_cuda4D(self):
        self.do_large_scale(cuda=True, ndims=4)

    def test_large_scale_cpu4D(self):
        self.do_large_scale(cuda=False, ndims=4)

    def do_test_with_wrapper_on_data(self, device='cpu'):

        #run the whole wrapper here
        data = np.load('test_bbc_data.npy', allow_pickle=True)
        coordinates = torch.tensor(data, device=device)
        #one row split
        row_splits = torch.tensor([0, coordinates.size(0)], dtype=torch.int32, device=device)
        #use dynamic bin width

        bin_indices, flat_bin_indices, n_bins, bin_width, n_per_bin = bin_by_coordinates(coordinates, row_splits, n_bins=21)

    def test_with_wrapper_on_data_cpu(self):
        self.do_test_with_wrapper_on_data(device='cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_with_wrapper_on_data_cuda(self):
        self.do_test_with_wrapper_on_data(device='cuda')
        


if __name__ == '__main__':
    unittest.main()
