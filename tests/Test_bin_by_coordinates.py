import torch
import numpy as np
import time
import unittest
import os.path as osp

# Load the shared libraries
cpu_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'bin_by_coordinates_cpu.so')
cuda_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'bin_by_coordinates_cuda.so')
torch.ops.load_library(cpu_so_file)
torch.ops.load_library(cuda_so_file)

def create_data(nvert, ncoords):
    coords = np.random.rand(nvert, ncoords) * 1.04
    coords[:, 0] *= 2.
    coords = torch.tensor(coords, dtype=torch.float32)
    row_splits = torch.tensor([0, nvert // 2, nvert], dtype=torch.int32)
    return coords, row_splits

class TestBinByCoordinates(unittest.TestCase):

    def test_bin_by_coordinates_cpu(self):
        coordinates, row_splits = create_data(10, 3)
        bin_width = torch.tensor([0.3], dtype=torch.float32)
        nbins = torch.tensor([int(np.ceil(2.08 / 0.3)), int(np.ceil(1.04 / 0.3)), int(np.ceil(1.04 / 0.3))], dtype=torch.int32)

        # Call the CPU function
        binning, fbinning, nperbin = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu(
            coordinates, row_splits, bin_width, nbins, True)

        print("CPU binning: ", binning)
        print("CPU fbinning: ", fbinning)
        print("CPU nperbin: ", nperbin)

        start = time.time()
        for _ in range(20):
            binning, fbinning, nperbin = torch.ops.bin_by_coordinates_cpu.bin_by_coordinates_cpu(
                coordinates, row_splits, bin_width, nbins, True)
        ntime = time.time() - start
        print("CPU Average time per run: ", ntime / 20.)

        # Save coordinates and fbinning for further processing
        if coordinates.shape[1] == 2:
            coordinates = np.concatenate([coordinates, np.zeros_like(coordinates[:, 0:1])], axis=-1)
        np.save("coords.npy", coordinates.cpu().numpy())
        np.save("binning.npy", fbinning.cpu().numpy())

        # Sort coordinates based on fbinning
        sorting = torch.argsort(fbinning)
        scoords = coordinates[sorting]
        sbinning = fbinning[sorting]

        # Create bin boundaries
        bin_boundaries = torch.cat([row_splits[0:1], nperbin])
        bin_boundaries = torch.cumsum(bin_boundaries, dim=0)
        print("CPU bin_boundaries: ", bin_boundaries)

    def test_bin_by_coordinates_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        coordinates, row_splits = create_data(10, 3)
        coordinates = coordinates.cuda()
        row_splits = row_splits.cuda()
        bin_width = torch.tensor([0.3], dtype=torch.float32).cuda()
        nbins = torch.tensor([int(np.ceil(2.08 / 0.3)), int(np.ceil(1.04 / 0.3)), int(np.ceil(1.04 / 0.3))], dtype=torch.int32).cuda()

        # Call the CUDA function
        binning, fbinning, nperbin = torch.ops.bin_by_coordinates_cuda.bin_by_coordinates(
            coordinates, row_splits, bin_width, nbins, True)

        print("CUDA binning: ", binning.cpu())
        print("CUDA fbinning: ", fbinning.cpu())
        print("CUDA nperbin: ", nperbin.cpu())

        start = time.time()
        for _ in range(20):
            binning, fbinning, nperbin = torch.ops.bin_by_coordinates_cuda.bin_by_coordinates(
                coordinates, row_splits, bin_width, nbins, True)
        ntime = time.time() - start
        print("CUDA Average time per run: ", ntime / 20.)

        # Save coordinates and fbinning for further processing
        if coordinates.shape[1] == 2:
            coordinates = torch.cat([coordinates, torch.zeros_like(coordinates[:, 0:1])], axis=-1)
        np.save("coords_cuda.npy", coordinates.cpu().numpy())
        np.save("binning_cuda.npy", fbinning.cpu().numpy())

        # Sort coordinates based on fbinning
        sorting = torch.argsort(fbinning)
        scoords = coordinates[sorting]
        sbinning = fbinning[sorting]

        # Create bin boundaries
        bin_boundaries = torch.cat([row_splits[0:1], nperbin])
        bin_boundaries = torch.cumsum(bin_boundaries, dim=0)
        print("CUDA bin_boundaries: ", bin_boundaries.cpu())

if __name__ == "__main__":
    unittest.main()
