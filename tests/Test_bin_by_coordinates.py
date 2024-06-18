import unittest
import torch
from bin_by_coordinates_cpu import bin_by_coordinates_cpu, bin_by_coordinates

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
        bin_by_coordinates_cpu(self.coordinates, self.rs, self.bin_width, self.n_bins, self.assigned_bin, self.flat_assigned_bin, self.n_per_bin, True)
        expected_assigned_bin = torch.tensor([[0, 0, 1], [1, 1, 2], [0, 2, 3]])
        expected_flat_assigned_bin = torch.tensor([1, 6, 4])
        expected_n_per_bin = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
        self.assertTrue(torch.equal(self.assigned_bin, expected_assigned_bin))
        self.assertTrue(torch.equal(self.flat_assigned_bin, expected_flat_assigned_bin))
        self.assertTrue(torch.equal(self.n_per_bin, expected_n_per_bin))

    def test_bin_by_coordinates(self):
        bin_by_coordinates(self.coordinates, self.rs, self.bin_width, self.n_bins, self.assigned_bin, self.flat_assigned_bin, self.n_per_bin, True)
        expected_assigned_bin = torch.tensor([[0, 0, 1], [1, 1, 2], [0, 2, 3]])
        expected_flat_assigned_bin = torch.tensor([1, 6, 4])
        expected_n_per_bin = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
        self.assertTrue(torch.equal(self.assigned_bin, expected_assigned_bin))
        self.assertTrue(torch.equal(self.flat_assigned_bin, expected_flat_assigned_bin))
        self.assertTrue(torch.equal(self.n_per_bin, expected_n_per_bin))

if __name__ == '__main__':
    unittest.main()