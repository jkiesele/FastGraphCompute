import torch
import numpy as np
import unittest

from ml4reco_modules import binned_select_knn

class TestBinnedKnn(unittest.TestCase):
    def knn_pytorch_baseline(self, K, coordinates):
        """
        Simple KNN implementation using PyTorch.
        Args:
            K (int): Number of nearest neighbors.
            coordinates (torch.Tensor): The input coordinates (n_points, n_dims).
    
        Returns:
            torch.Tensor: Indices of the K nearest neighbors for each point.
            torch.Tensor: Distances to the K nearest neighbors for each point.
        """
        # Calculate the pairwise distances between points
        # dist_matrix = torch.cdist(coordinates, coordinates)
        dist_matrix = torch.cdist(coordinates, coordinates, compute_mode='donot_use_mm_for_euclid_dist')


        # Get the top-K nearest neighbors (including self at index 0)
        dist, indices = torch.topk(dist_matrix, K, largest=False)
        
        return indices, dist

    def do_large_test(self, device = 'cpu', strict=False):
        torch.manual_seed(45) # Don't change the seed. At some seeds it doesn't work for numerical reasons.
                              # Which one is closer can be ambiguous which might result in slightly different
                              # indices. The distance still remains "close".
        if device == 'cpu':
            # Parameters for the test
            n_points = 10000  # Number of points
            n_dims = 3  # Number of dimensions
            K = 50  # Number of nearest neighbors to find
            n_bins = 10  # Number of bins across each dimension
        else:
            # Parameters for the test
            n_points = 10000  # Number of points
            n_dims = 3      # Number of dimensions
            K = 100           # Number of nearest neighbors to find
            n_bins = 10      # Number of bins across each dimension

        # Generate random coordinates (3D points)
        coordinates = torch.rand((n_points, n_dims), dtype=torch.float32, device=device)
        
        # Create dummy row_splits (assuming uniform splitting for simplicity)
        row_splits = torch.tensor([0, n_points], dtype=torch.int32, device=device)
        
        # Optionally create a dummy direction tensor
        direction = None  # For this test, we won't use direction constraints
        
        # Call your binned_select_knn function
        idx_knn, dist_knn = binned_select_knn(K, coordinates, row_splits, direction=direction, n_bins=n_bins)

        dist_knn = torch.sqrt(dist_knn)
        idx_knn[idx_knn<0] = 0 # replace negative indices with 0, just here
        
        # Call the PyTorch-based baseline KNN
        idx_pytorch, dist_pytorch = self.knn_pytorch_baseline(K, coordinates)



        # Sort the distances and indices for both binned_select_knn and the PyTorch KNN
        dist_knn_sorted, knn_sorted_indices = torch.sort(dist_knn, dim=1)
        idx_knn_sorted = torch.gather(idx_knn, 1, knn_sorted_indices)

        dist_pytorch_sorted, pytorch_sorted_indices = torch.sort(dist_pytorch, dim=1)
        idx_pytorch_sorted = torch.gather(idx_pytorch, 1, pytorch_sorted_indices)

        #print them all
        print("idx_knn: ", idx_knn_sorted)
        print("idx_pytorch: ", idx_pytorch_sorted)
        print("dist_knn: ", dist_knn_sorted)
        print("dist_pytorch: ", dist_pytorch_sorted)

        distance_fn = lambda i, j: torch.sqrt(torch.sum(torch.square(coordinates[i] - coordinates[j])))

        for i in range(n_points):
            for j in range(K):
                if idx_knn_sorted[i,j] != idx_pytorch_sorted[i, j]:
                    print("Error at %dth element %dth neighbour. KNN: %d, Torch: %d. dist from KNNth: %f, dist from torchth: %f. Recomputed distance knn: %f, Recomputed distance torch: %f."
                          % (i, j, int(idx_knn_sorted[i,j]), int(idx_pytorch_sorted[i, j]), float(dist_knn_sorted[i, j]), float(dist_pytorch_sorted[i, j]), float(distance_fn(i, idx_knn_sorted[i, j])), float(distance_fn(i, idx_pytorch_sorted[i, j]))),)

        # Compare distances and indices
        if strict:
            self.assertTrue(torch.equal(idx_knn_sorted, idx_pytorch_sorted), "Indices do not match!")
        else:
            matching_indices = torch.eq(idx_knn_sorted, idx_pytorch_sorted)
            match_percentage = matching_indices.sum().item() / matching_indices.numel()
            print("Match percentage", match_percentage)
            self.assertTrue(match_percentage >= 0.999,
                            f"Only {match_percentage * 100:.2f}% of indices match, expected at least 99.9%!")

        self.assertTrue(torch.allclose(dist_knn_sorted, dist_pytorch_sorted, atol=1e-3), "Distances do not match!")


    def test_large_binned_select_knn_cpu(self):
        self.do_large_test(device='cpu', strict=False)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_large_binned_select_knn_cuda(self):
        self.do_large_test(device='cuda', strict=False)


if __name__ == '__main__':
    unittest.main()
