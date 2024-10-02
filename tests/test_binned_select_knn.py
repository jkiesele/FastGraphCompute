import torch
import numpy as np
import unittest

from ml4reco_modules import binned_select_knn

class TestBinByCoordinates(unittest.TestCase):

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
        dist_matrix = torch.cdist(coordinates, coordinates)
        
        # Get the top-K nearest neighbors (including self at index 0)
        dist, indices = torch.topk(dist_matrix, K, largest=False)
        
        return indices, dist

    def do_large_test(self, device = 'cpu'):

        # Parameters for the test
        n_points = 7  # Number of points
        n_dims = 3      # Number of dimensions
        K = 3           # Number of nearest neighbors to find
        
        # Generate random coordinates (3D points)
        coordinates = torch.rand((n_points, n_dims), dtype=torch.float32, device=device)
        
        # Create dummy row_splits (assuming uniform splitting for simplicity)
        row_splits = torch.tensor([0, n_points], dtype=torch.int32, device=device)
        
        # Optionally create a dummy direction tensor
        direction = None  # For this test, we won't use direction constraints
        
        # Call your binned_select_knn function
        idx_knn, dist_knn = binned_select_knn(K, coordinates, row_splits, direction=direction, n_bins=3)

        dist_knn = torch.sqrt(dist_knn)
        idx_knn[idx_knn<0] = 0 # replace negative indices with 0, just here
        
        # Call the PyTorch-based baseline KNN
        idx_pytorch, dist_pytorch = self.knn_pytorch_baseline(K, coordinates)

        #print them all
        print("idx_knn: ", idx_knn)
        print("idx_pytorch: ", idx_pytorch)
        print("dist_knn: ", dist_knn)
        print("dist_pytorch: ", dist_pytorch)
        

        # Sort the distances and indices for both binned_select_knn and the PyTorch KNN
        dist_knn_sorted, knn_sorted_indices = torch.sort(dist_knn, dim=1)
        idx_knn_sorted = torch.gather(idx_knn, 1, knn_sorted_indices)

        dist_pytorch_sorted, pytorch_sorted_indices = torch.sort(dist_pytorch, dim=1)
        idx_pytorch_sorted = torch.gather(idx_pytorch, 1, pytorch_sorted_indices)
        
        # Compare distances and indices
        self.assertTrue(torch.equal(idx_knn_sorted, idx_pytorch_sorted), "Indices do not match!")
        self.assertTrue(torch.allclose(dist_knn_sorted, dist_pytorch_sorted, atol=1e-3), "Distances do not match!")


    def test_large_binned_select_knn_cpu(self):
        self.do_large_test(device='cpu')
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_large_binned_select_knn_cuda(self):
        return
        self.do_large_test(device='cuda')


if __name__ == '__main__':
    unittest.main()
