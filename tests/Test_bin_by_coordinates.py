import torch
import numpy as np
import time

# Assuming BinByCoordinates, BinnedSelectKnn, and IndexReplacer are implemented for PyTorch
from bin_by_coordinates_cpu import BinByCoordinates
from binned_select_knn import BinnedSelectKnn
from index_replacer import IndexReplacer

def create_data(nvert, ncoords):
    coords = np.random.rand(nvert, ncoords) * 1.04
    coords[:, 0] *= 2.
    coords = torch.tensor(coords, dtype=torch.float32)
    row_splits = torch.tensor([0, nvert // 2, nvert], dtype=torch.int32)
    return coords, row_splits

coordinates, row_splits = create_data(10, 3)
binwidth = torch.tensor([0.3], dtype=torch.float32)

# Assuming BinByCoordinates returns binning, fbinning, nperbin, nb
binning, fbinning, nperbin, nb = BinByCoordinates(coordinates, row_splits, binwidth)

start = time.time()
for _ in range(20):
    binning, fbinning, nperbin, nb = BinByCoordinates(coordinates, row_splits, binwidth)
ntime = time.time() - start
print(nb, ntime / 20.)

print(nperbin)

if coordinates.shape[1] == 2:
    coordinates = torch.cat([coordinates, torch.zeros_like(coordinates[:, 0:1])], axis=-1)
torch.save(coordinates, "coords.pt")
torch.save(fbinning, "binning.pt")

sorting = torch.argsort(fbinning)
scoords = coordinates[sorting]
sbinning = fbinning[sorting]

bin_boundaries = torch.cat([row_splits[0:1], torch.cumsum(nperbin, dim=0)], dim=0)

idx, dist = BinnedSelectKnn(5, scoords, sbinning, bin_boundaries=bin_boundaries, n_bins=nb, bin_width=binwidth)

print('pre-resort')
print(idx)
print(dist)

idx = IndexReplacer(idx, sorting)
dist = dist[sorting]
idx = idx[sorting]

print('post-resort')
print(idx)
print(dist)

# Assuming SelectKnn is implemented for PyTorch
from select_knn import SelectKnn
idx, dist = SelectKnn(5, coordinates, row_splits)

print('SelectKnn')
print(idx)
print(dist)









# import torch
# import unittest
# import os.path as osp


# # Load the shared libraries
# cpu_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'bin_by_coordinates_cpu.so')
# torch.ops.load_library(cpu_so_file)

# from bin_by_coordinates_cpu import bin_by_coordinates_cpu, bin_by_coordinates


# class TestBinByCoordinatesCPU(unittest.TestCase):
#     def setUp(self):
#         self.coordinates = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#         self.rs = torch.tensor([0, 2, 3])
#         self.bin_width = torch.tensor([2.0])
#         self.n_bins = torch.tensor([2, 2])
#         self.assigned_bin = torch.zeros((3, 3), dtype=torch.int32)
#         self.flat_assigned_bin = torch.zeros(3, dtype=torch.int32)
#         self.n_per_bin = torch.zeros(8, dtype=torch.int32)

#     def test_bin_by_coordinates_cpu(self):
#         # Keep coordinates as Float type
#         # Only convert tensors that need to be Int for specific operations
#         rs_int = self.rs.int()
#         # bin_width should likely remain as Float if it represents a measurement
#         n_bins_int = self.n_bins.int()  # Assuming n_bins needs to be Int for indexing
#         assigned_bin_int = self.assigned_bin.int()
#         flat_assigned_bin_int = self.flat_assigned_bin.int()
#         n_per_bin_int = self.n_per_bin.int()

#         # Call the function without changing coordinates to Int
#         bin_by_coordinates_cpu(self.coordinates, rs_int, self.bin_width, n_bins_int, assigned_bin_int, flat_assigned_bin_int, n_per_bin_int, True)
#         expected_assigned_bin = torch.tensor([[0, 0, 1], [1, 1, 2], [0, 2, 3]], dtype=torch.int32)
#         expected_flat_assigned_bin = torch.tensor([1, 6, 4], dtype=torch.int32)
#         expected_n_per_bin = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.int32)
#         self.assertTrue(torch.equal(assigned_bin_int, expected_assigned_bin))
#         self.assertTrue(torch.equal(flat_assigned_bin_int, expected_flat_assigned_bin))
#         self.assertTrue(torch.equal(n_per_bin_int, expected_n_per_bin))

#     def test_bin_by_coordinates(self):
#         # Keep coordinates as Float type
#         # Only convert tensors that need to be Int for specific operations
#         rs_int = self.rs.int()
#         # bin_width should likely remain as Float if it represents a measurement
#         n_bins_int = self.n_bins.int()  # Assuming n_bins needs to be Int for indexing
#         assigned_bin_int = self.assigned_bin.int()
#         flat_assigned_bin_int = self.flat_assigned_bin.int()
#         n_per_bin_int = self.n_per_bin.int()

#         # Call the function without changing coordinates to Int
#         bin_by_coordinates(self.coordinates, rs_int, self.bin_width, n_bins_int, assigned_bin_int, flat_assigned_bin_int, n_per_bin_int, True)
#         expected_assigned_bin = torch.tensor([[0, 0, 1], [1, 1, 2], [0, 2, 3]], dtype=torch.int32)
#         expected_flat_assigned_bin = torch.tensor([1, 6, 4], dtype=torch.int32)
#         expected_n_per_bin = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.int32)
#         self.assertTrue(torch.equal(assigned_bin_int, expected_assigned_bin))
#         self.assertTrue(torch.equal(flat_assigned_bin_int, expected_flat_assigned_bin))
#         self.assertTrue(torch.equal(n_per_bin_int, expected_n_per_bin))

# if __name__ == '__main__':
#     unittest.main()