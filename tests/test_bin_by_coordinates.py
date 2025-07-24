import torch
import numpy as np
import unittest
import os.path as osp
import fastgraphcompute.extensions
from fastgraphcompute import bin_by_coordinates
from typing import Tuple


class BinByCoordinatesModule(torch.nn.Module):
    def __init__(self, cuda=False):
        super().__init__()

    def forward(
        self,
        coordinates: torch.Tensor,
        row_splits: torch.Tensor,
        bin_width: torch.Tensor,
        nbins: torch.Tensor,
        return_all: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return bin_by_coordinates(coordinates, row_splits, bin_width, nbins, return_all)


class TestBinByCoordinates(unittest.TestCase):
    def setUp(self):
        # Define your bin widths and number of bins per dimension
        # Example: 0.5 units per bin
        self.bin_width = torch.tensor([0.5], dtype=torch.float32)
        self.nbins = torch.tensor(
            [10, 10], dtype=torch.int64)  # Example: 10x10 grid
        self.coordinates = torch.tensor([
            [0.1, 0.1],
            [2.5, 2.5],
            [4.9, 4.9]
        ], dtype=torch.float32)
        # Expected after normalization: [0,0], [2.4,2.4], [4.8,4.8]
        # With bin_width 0.5: bins are [0,0], [4,4], [9,9]
        self.expected_assigned_bin = torch.tensor([
            [0, 0, 0],
            [0, 4, 4],
            [0, 9, 9]
        ], dtype=torch.int64)
        # No actual split in this case, just one segment
        self.row_splits = torch.tensor([0, 3], dtype=torch.int64)

    def do_simple_binning(self, cuda=False):
        coord = self.coordinates.to('cuda') if cuda else self.coordinates
        rs = self.row_splits.to('cuda') if cuda else self.row_splits
        bin_width = self.bin_width.to('cuda') if cuda else self.bin_width
        nbins = self.nbins.to('cuda') if cuda else self.nbins

        output_assigned_bin, output_flat_assigned_bin, _, _, output_n_per_bin = bin_by_coordinates(
            coord, rs, bin_width, nbins, True)

        if cuda:
            output_assigned_bin = output_assigned_bin.to('cpu')
            output_flat_assigned_bin = output_flat_assigned_bin.to('cpu')
            output_n_per_bin = output_n_per_bin.to('cpu')

        expected_flat_assigned_bin = torch.zeros(
            len(self.expected_assigned_bin), dtype=torch.int64)
        expected_n_per_bin = torch.zeros((100,), dtype=torch.int64)

        for i in range(len(self.expected_assigned_bin)):
            f = self.expected_assigned_bin[i, 1] * \
                10 + self.expected_assigned_bin[i, 2]
            expected_flat_assigned_bin[i] = f
            expected_n_per_bin[f] += 1

        self.assertTrue(torch.equal(output_assigned_bin,
                        self.expected_assigned_bin))
        self.assertTrue(torch.equal(output_n_per_bin, expected_n_per_bin))
        self.assertTrue(torch.equal(output_flat_assigned_bin,
                        expected_flat_assigned_bin))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_simple_binning_cuda(self):
        self.do_simple_binning(cuda=True)

    def test_simple_binning_cpu(self):
        self.do_simple_binning(cuda=False)

    def test_jit_script_compatibility(self):
        """Test if BinByCoordinatesModule is compatible with torch.jit.script (CPU and CUDA)."""
        for cuda in (False, True) if torch.cuda.is_available() else (False,):
            device = 'cuda' if cuda else 'cpu'
            try:
                module = BinByCoordinatesModule(cuda=cuda).to(device)
                scripted_module = torch.jit.script(module)
                # Prepare inputs
                coordinates = self.coordinates.to(device)
                row_splits = self.row_splits.to(device)
                bin_width = self.bin_width.to(device)
                nbins = self.nbins.to(device)
                with torch.no_grad():
                    orig = module(coordinates, row_splits,
                                  bin_width, nbins, True)
                    scripted = scripted_module(
                        coordinates, row_splits, bin_width, nbins, True)
                # Each output is a tuple of tensors; compare each
                self.assertEqual(len(orig), len(scripted))
                for o, s in zip(orig, scripted):
                    self.assertTrue(torch.allclose(
                        o.cpu(), s.cpu(), atol=1e-6), f"Mismatch in output: {o} vs {s}")
                    self.assertEqual(o.shape, s.shape)
            except Exception as e:
                self.fail(
                    f"Failed to script BinByCoordinates op (cuda={cuda}): {str(e)}")

    def do_out_of_bounds(self, cuda=False):
        print("Running out of bounds test. Overflow warnings here are normal!")
        # Ensure coordinates out of bounds are handled properly
        coordinates = torch.tensor([
            [-1.0, -1.0],
            [5.1, 5.1],
            [0.7, 0.7]
        ], dtype=torch.float32)

        coord = coordinates.to('cuda') if cuda else coordinates
        rs = self.row_splits.to('cuda') if cuda else self.row_splits
        bin_width = self.bin_width.to('cuda') if cuda else self.bin_width
        nbins = self.nbins.to('cuda') if cuda else self.nbins

        output_assigned_bin, output_flat_assigned_bin, _, _, output_n_per_bin = bin_by_coordinates(
            coord, rs, bin_width, nbins, True)

        if cuda:
            output_assigned_bin = output_assigned_bin.to('cpu')
            output_flat_assigned_bin = output_flat_assigned_bin.to('cpu')
            output_n_per_bin = output_n_per_bin.to('cpu')

        # Expected coordinates after normalization: [[0,0], [6.1,6.1], [1.7,1.7]]
        # With bin_width 0.5: bins are [[0,0], [9,9] (clamped from 12), [3,3]]
        expected_assigned_bin = torch.tensor([
            [0, 0, 0],
            [0, 9, 9],
            [0, 3, 3],
        ], dtype=torch.int64)

        expected_flat_assigned_bin = torch.zeros(
            len(expected_assigned_bin), dtype=torch.int64)
        expected_n_per_bin = torch.zeros((100,), dtype=torch.int64)

        for i in range(len(expected_assigned_bin)):
            f = expected_assigned_bin[i, 1] * 10 + expected_assigned_bin[i, 2]
            expected_flat_assigned_bin[i] = f
            expected_n_per_bin[f] += 1

        self.assertTrue(torch.equal(output_assigned_bin, expected_assigned_bin),
                        f"Expected assigned bin: {expected_assigned_bin}, Got: {output_assigned_bin}")
        self.assertTrue(torch.equal(output_flat_assigned_bin, expected_flat_assigned_bin),
                        f"Expected flat assigned bin: {expected_flat_assigned_bin}, Got: {output_flat_assigned_bin}")
        self.assertTrue(torch.equal(output_n_per_bin, expected_n_per_bin),
                        f"Expected n per bin: {expected_n_per_bin}, Got: {output_n_per_bin}")

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

    def do_large_scale(self, cuda=False, ndims=2):
        # Test with a larger scale of coordinates
        # Coordinates in the range [0, 1]
        coordinates = torch.rand((1000, ndims))
        # extend bin width and nbins
        # Example: 0.5 units per bin
        bin_width = torch.tensor([0.5], dtype=torch.float32)
        # Example: 10x10 grid
        nbins = torch.tensor([10]*ndims, dtype=torch.int64)
        row_splits = torch.tensor([0, 300, 700, 1000], dtype=torch.int64)

        coord = coordinates.to('cuda') if cuda else coordinates
        rs = row_splits.to('cuda') if cuda else row_splits
        if cuda:
            bin_width = bin_width.to('cuda')
            nbins = nbins.to('cuda')

        output_assigned_bin, output_flat_assigned_bin, _, _, output_n_per_bin = bin_by_coordinates(
            coord, rs, bin_width, nbins, True)

        # sanity check. e.g. the entry in the first dimension of output_assigned_bin should correspond to the
        # row split index that entry is in
        rs_index = self.calc_batch_index_from_rs(row_splits)
        rs_index = rs_index.to(torch.int64)
        rs_index = rs_index.to(coord.device)
        self.assertTrue(torch.all(
            output_assigned_bin[:, 0] == rs_index), f"Expected: {rs_index}, Got: {output_assigned_bin[:, 0]}")

        if cuda:
            output_assigned_bin = output_assigned_bin.to('cpu')
            output_flat_assigned_bin = output_flat_assigned_bin.to('cpu')
            output_n_per_bin = output_n_per_bin.to('cpu')

        # check if all the points are within the bounds
        self.assertTrue(torch.sum(output_n_per_bin) == row_splits[-1], "Not all points were assigned a bin, expected %d, got %d" % (
            row_splits[-1], torch.sum(output_n_per_bin)))

        # Just ensure this runs without error for a basic sanity check
        self.assertEqual(output_assigned_bin.size(0), 1000)
        self.assertEqual(output_flat_assigned_bin.size(0), 1000)
        # Sum of indices may be less due to clamping to zero
        self.assertTrue((output_n_per_bin.sum() <= 1000).item())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_large_scale_cuda(self):
        self.do_large_scale(cuda=True)

    def test_large_scale_cpu(self):
        self.do_large_scale(cuda=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_large_scale_cuda4D(self):
        self.do_large_scale(cuda=True, ndims=4)

    def test_large_scale_cpu4D(self):
        self.do_large_scale(cuda=False, ndims=4)

    def do_test_with_wrapper_on_data(self, device='cpu'):

        # run the whole wrapper here
        data = np.load(osp.join(osp.dirname(__file__),
                       'test_bbc_data.npy'), allow_pickle=True)
        coordinates = torch.tensor(data, dtype=torch.float32, device=device)
        # one row split
        row_splits = torch.tensor(
            [0, coordinates.size(0)], dtype=torch.int64, device=device)
        # use dynamic bin width

        bin_indices, flat_bin_indices, n_bins, bin_width, n_per_bin = bin_by_coordinates(
            coordinates, row_splits, n_bins=21)

    def test_with_wrapper_on_data_cpu(self):
        self.do_test_with_wrapper_on_data(device='cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_with_wrapper_on_data_cuda(self):
        self.do_test_with_wrapper_on_data(device='cuda')


if __name__ == '__main__':
    unittest.main()
