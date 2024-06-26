import torch
import unittest
import os.path as osp


# Load the shared libraries
cpu_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'bin_by_coordinates_cpu.so')
torch.ops.load_library(cpu_so_file)

from bin_by_coordinates_cpu import bin_by_coordinates_cpu
from bin_by_coordinates_cpu import bin_by_coordinates


class TestBinByCoordinatesCPU(unittest.TestCase):
    def setUp(self):
        self.coordinates = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.rs = torch.tensor([0, 2, 3])
        self.bin_width = torch.tensor([2.0])
        self.n_bins = torch.tensor([2, 2])
        self.assigned_bin = torch.zeros((3, 3), dtype=torch.int32)
        self.flat_assigned_bin = torch.zeros(3, dtype=torch.int32)
        self.n_per_bin = torch.zeros(8, dtype=torch.int32)

    def test_bin_by_coordinates_cpu(self):
        # Convert tensors to Int type
        rs_int = self.rs.int()
        bin_width_int = self.bin_width.int()
        n_bins_int = self.n_bins.int()
        assigned_bin_int = self.assigned_bin.int()
        flat_assigned_bin_int = self.flat_assigned_bin.int()
        n_per_bin_int = self.n_per_bin.int()

        bin_by_coordinates_cpu(self.coordinates, rs_int, bin_width_int, n_bins_int, assigned_bin_int, flat_assigned_bin_int, n_per_bin_int, True)
        expected_assigned_bin = torch.tensor([[0, 0, 1], [1, 1, 2], [0, 2, 3]], dtype=torch.int32)
        expected_flat_assigned_bin = torch.tensor([1, 6, 4], dtype=torch.int32)
        expected_n_per_bin = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.int32)
        self.assertTrue(torch.equal(assigned_bin_int, expected_assigned_bin))
        self.assertTrue(torch.equal(flat_assigned_bin_int, expected_flat_assigned_bin))
        self.assertTrue(torch.equal(n_per_bin_int, expected_n_per_bin))

    def test_bin_by_coordinates(self):
        # Convert tensors to Int type
        rs_int = self.rs.int()
        bin_width_int = self.bin_width.int()
        n_bins_int = self.n_bins.int()
        assigned_bin_int = self.assigned_bin.int()
        flat_assigned_bin_int = self.flat_assigned_bin.int()
        n_per_bin_int = self.n_per_bin.int()

        bin_by_coordinates(self.coordinates, rs_int, bin_width_int, n_bins_int, assigned_bin_int, flat_assigned_bin_int, n_per_bin_int, True)
        expected_assigned_bin = torch.tensor([[0, 0, 1], [1, 1, 2], [0, 2, 3]], dtype=torch.int32)
        expected_flat_assigned_bin = torch.tensor([1, 6, 4], dtype=torch.int32)
        expected_n_per_bin = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.int32)
        self.assertTrue(torch.equal(assigned_bin_int, expected_assigned_bin))
        self.assertTrue(torch.equal(flat_assigned_bin_int, expected_flat_assigned_bin))
        self.assertTrue(torch.equal(n_per_bin_int, expected_n_per_bin))


if __name__ == '__main__':
    unittest.main()
    
    

    # def test_bin_by_coordinates_cpu(self):
    #     bin_by_coordinates_cpu(self.coordinates, self.rs, self.bin_width, self.n_bins, self.assigned_bin, self.flat_assigned_bin, self.n_per_bin, True)
    #     expected_assigned_bin = torch.tensor([[0, 0, 1], [1, 1, 2], [0, 2, 3]])
    #     expected_flat_assigned_bin = torch.tensor([1, 6, 4])
    #     expected_n_per_bin = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    #     self.assertTrue(torch.equal(self.assigned_bin, expected_assigned_bin))
    #     self.assertTrue(torch.equal(self.flat_assigned_bin, expected_flat_assigned_bin))
    #     self.assertTrue(torch.equal(self.n_per_bin, expected_n_per_bin))

    # def test_bin_by_coordinates(self):
    #     bin_by_coordinates(self.coordinates, self.rs, self.bin_width, self.n_bins, self.assigned_bin, self.flat_assigned_bin, self.n_per_bin, True)
    #     expected_assigned_bin = torch.tensor([[0, 0, 1], [1, 1, 2], [0, 2, 3]])
    #     expected_flat_assigned_bin = torch.tensor([1, 6, 4])
    #     expected_n_per_bin = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    #     self.assertTrue(torch.equal(self.assigned_bin, expected_assigned_bin))
    #     self.assertTrue(torch.equal(self.flat_assigned_bin, expected_flat_assigned_bin))
    #     self.assertTrue(torch.equal(self.n_per_bin, expected_n_per_bin))