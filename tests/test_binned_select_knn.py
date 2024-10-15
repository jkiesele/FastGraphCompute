import torch
import numpy as np
import unittest

from ml4reco_modules import binned_select_knn

class TestBinnedSelectKnn(unittest.TestCase):
    def knn_pytorch_baseline(self, K, coordinates, row_splits=None):
        """
        Simple KNN implementation using PyTorch.
        Args:
            K (int): Number of nearest neighbors.
            coordinates (torch.Tensor): The input coordinates (n_points, n_dims).
    
        Returns:
            torch.Tensor: Indices of the K nearest neighbors for each point.
            torch.Tensor: Distances**2 to the K nearest neighbors for each point.
        """

        if row_splits is None:
            row_splits = torch.tensor([0, coordinates.size(0)], dtype=torch.int32, device=coordinates.device)
        # Calculate the pairwise distances between points
        # dist_matrix = torch.cdist(coordinates, coordinates)

        all_indices = []
        all_dist = []

        for i in range(row_splits.size(0) - 1):
            start = row_splits[i]
            end = row_splits[i + 1]
            dist_matrix = torch.cdist(coordinates[start:end], coordinates[start:end], compute_mode='donot_use_mm_for_euclid_dist')
            # Get the top-K nearest neighbors (including self at index 0)
            dist, indices = torch.topk(dist_matrix, K, largest=False)

            #add row split offset to indices
            indices = indices + start

            all_indices.append(indices)
            all_dist.append(dist)

        all_indices = torch.cat(all_indices, dim=0)
        all_dist = torch.cat(all_dist, dim=0)

        dist_pytorch_sorted, pytorch_sorted_indices = torch.sort(all_dist, dim=1)
        idx_pytorch_sorted = torch.gather(all_indices, 1, pytorch_sorted_indices)

        #make sure the baseline makes sense
        #the first column is the distance to itself, so it should be zero and the index should be range(len(coords)), assert:
        assert torch.allclose(dist_pytorch_sorted[:, 0], torch.zeros_like(dist_pytorch_sorted[:, 0]))
        assert torch.all(idx_pytorch_sorted[:, 0] == torch.arange(len(coordinates), device=coordinates.device))
        
        return idx_pytorch_sorted, dist_pytorch_sorted**2
    
    def binned_select_knn_tester(self, K, coordinates, row_splits, direction=None, n_bins=None):
        """
        Returns:
            torch.Tensor: Indices of the K nearest neighbors for each point.
            torch.Tensor: Distances**2 to the K nearest neighbors for each point.
        """

        idx_knn, dist_knn = binned_select_knn(K, coordinates, row_splits, direction=direction, n_bins=n_bins)

        dist_knn_sorted, knn_sorted_indices = torch.sort(dist_knn, dim=1)
        idx_knn_sorted = torch.gather(idx_knn, 1, knn_sorted_indices)

        return idx_knn_sorted, dist_knn_sorted


    def do_large_test(self, device = 'cpu', strict=False, n_bins= None):
        torch.manual_seed(45) # Don't change the seed. At some seeds it doesn't work for numerical reasons.
                              # Which one is closer can be ambiguous which might result in slightly different
                              # indices. The distance still remains "close".
        if device == 'cpu':
            # Parameters for the test
            n_points = 10000  # Number of points
            n_dims = 3  # Number of dimensions
            K = 50  # Number of nearest neighbors to find
        else:
            # Parameters for the test
            n_points = 10000  # Number of points
            n_dims = 3      # Number of dimensions
            K = 50           # Number of nearest neighbors to find

        # Generate random coordinates (3D points)
        coordinates = torch.rand((n_points, n_dims), dtype=torch.float32, device='cpu') #random works differently on cpu and gpu
        coordinates = coordinates.to(device) 
        
        # Create dummy row_splits (assuming uniform splitting for simplicity)
        row_splits = torch.tensor([0, n_points//3, n_points//2,  n_points], dtype=torch.int32, device=device)
        
        # Optionally create a dummy direction tensor
        direction = None  # For this test, we won't use direction constraints
        
        # Call your binned_select_knn function
        idx_knn_sorted, dist_knn_sorted = self.binned_select_knn_tester(K, coordinates, row_splits, direction=direction, n_bins=n_bins)

        # sanity checks, for each row split, we can only have indices from that split
        for i in range(row_splits.size(0) - 1):
            start = row_splits[i]
            end = row_splits[i + 1]
            good = (idx_knn_sorted[start:end] >= start) & (idx_knn_sorted[start:end] < end)
            self.assertTrue(torch.all(good), "Indices are out of row split bounds!, these are the occurences: "+str(idx_knn_sorted[start:end][~good]))

        idx_pytorch_sorted, dist_pytorch_sorted = self.knn_pytorch_baseline(K, coordinates, row_splits)

        distance_fn = lambda i, j: (torch.sum(torch.square(coordinates[i] - coordinates[j])))

        for i in range(n_points):
            for j in range(K):
                if idx_knn_sorted[i,j] != idx_pytorch_sorted[i, j]:
                    distances = float(dist_knn_sorted[i, j]), float(dist_pytorch_sorted[i, j]), float(distance_fn(i, idx_knn_sorted[i, j])), float(distance_fn(i, idx_pytorch_sorted[i, j]))
                    # check if all the distances are very close to each other
                    self.assertTrue(np.allclose(distances, distances[0], atol=1e-3), "Distances are not close!\n"+
                                    "Error at %dth element %dth neighbour. KNN: %d, Torch: %d. dist from KNNth: %f, dist from torchth: %f. Recomputed distance knn: %f, Recomputed distance torch: %f."
                          % (i, j, int(idx_knn_sorted[i,j]), int(idx_pytorch_sorted[i, j]), float(dist_knn_sorted[i, j]), float(dist_pytorch_sorted[i, j]), float(distance_fn(i, idx_knn_sorted[i, j])), float(distance_fn(i, idx_pytorch_sorted[i, j]))))
                    #print(,)

        # Compare distances and indices
        if strict:
            self.assertTrue(torch.equal(idx_knn_sorted, idx_pytorch_sorted), "Indices do not match!")
        else:
            matching_indices = torch.eq(idx_knn_sorted, idx_pytorch_sorted)
            match_percentage = matching_indices.sum().item() / matching_indices.numel()
            #print("Match percentage", match_percentage)
            self.assertTrue(match_percentage >= 0.999,
                            f"Only {match_percentage * 100:.2f}% of indices match, expected at least 99.9%!")

        self.assertTrue(torch.allclose(dist_knn_sorted, dist_pytorch_sorted, atol=1e-3), "Distances do not match!")


    def test_large_binned_select_knn_cpu(self):
        self.do_large_test(device='cpu', strict=False)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_large_binned_select_knn_cuda(self):
        self.do_large_test(device='cuda', strict=False)


    ####### gradients
    def do_test_binned_knn_gradient(self, device, n_dims, K):
        torch.manual_seed(45)

        # Parameters for the test
        n_points = 1000  # Number of points

        # Generate random coordinates and set requires_grad=True to track gradients
        coordinates = torch.rand((n_points, n_dims), dtype=torch.float32, device=device, requires_grad=True)-0.5

        # Create dummy row_splits (assuming uniform splitting for simplicity)
        row_splits = torch.tensor([0, n_points], dtype=torch.int32, device=device)
        rand_offsets = torch.rand((n_points, K), dtype=torch.float32, device=device)-0.5

        # Optionally create a dummy direction tensor
        direction = None  # For this test, we won't use direction constraints

        idx_knn, dist_knn = self.binned_select_knn_tester(K, coordinates, row_splits, direction=direction, n_bins=10)

        #create a non-trivial gradient
        grad_knn = torch.autograd.grad(outputs=(n_points*(dist_knn+rand_offsets)**2).mean()-dist_knn.mean(), inputs=coordinates)[0]
        
        idx_pytorch, dist_pytorch = self.knn_pytorch_baseline(K, coordinates)

        #create same non-trivial gradient
        grad_pytorch = torch.autograd.grad(outputs=(n_points*(dist_pytorch+rand_offsets)**2).mean()-dist_pytorch.mean(), inputs=coordinates)[0]
        #first make sure the indices are the same
        self.assertTrue(torch.equal(idx_knn, idx_pytorch), "Indices do not match!")
        #make sure distances are very close
        self.assertTrue(torch.allclose(dist_knn, dist_pytorch, atol=1e-5, rtol=1e-3), "Distances do not match!")

        # Compare the gradients with some tolerance
        self.assertTrue(torch.allclose(grad_knn, grad_pytorch, atol=1e-5, rtol=1e-3), 
                        "Gradients from custom implementation and torch-native kNN do not match!\n"+str(grad_knn)+'\nversus target\n'+str(grad_pytorch))


    def test_binned_knn_gradient_cpu_K5(self):
        self.do_test_binned_knn_gradient(device='cpu', n_dims=2, K=5)

    def test_binned_knn_gradient_cpu_K10(self):
        self.do_test_binned_knn_gradient(device='cpu', n_dims=4, K=10)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_binned_knn_gradient_cuda_K50(self):
        self.do_test_binned_knn_gradient(device='cuda', n_dims=2, K=50)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_binned_knn_gradient_cuda_K10(self):
        self.do_test_binned_knn_gradient(device='cuda', n_dims=4, K=10)


if __name__ == '__main__':
    unittest.main()
