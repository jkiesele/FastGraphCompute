import unittest
import torch
from fastgraphcompute.object_condensation import ObjectCondensation  # Replace with the actual import path

class TestObjectCondensation(unittest.TestCase):

    def setUp(self):
        self.oc = ObjectCondensation()
        self.device = torch.device("cpu")

    def test_scatter_to_N_indices(self):
        x_k_m = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        asso_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        M = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

        expected_output = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        result = self.oc._scatter_to_N_indices(x_k_m.unsqueeze(-1), asso_indices, M)
        self.assertTrue(torch.equal(result, expected_output))

    def test_mean_per_row_split(self):
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
        row_splits = torch.tensor([0, 2, 4], dtype=torch.int32)

        expected_output = torch.tensor([1.5, 3.5], dtype=torch.float32)
        result = self.oc._mean_per_row_split(x, row_splits)
        self.assertTrue(torch.allclose(result, expected_output))

    def test_beta_loss(self):
        beta_k_m = torch.tensor([[[0.5], [0.4]], [[0.3], [0.2]]], dtype=torch.float32)
        result = self.oc._beta_loss(beta_k_m)
        expected_output = torch.tensor([[0.6], [1.2]], dtype=torch.float32)  # Derived manually
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-3))

    def test_get_alpha_indices(self):
        beta_k_m = torch.tensor([[[0.1], [0.9]], [[0.5], [0.6]]], dtype=torch.float32)
        M = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

        expected_output = torch.tensor([1, 3], dtype=torch.int32)
        result = self.oc.get_alpha_indices(beta_k_m, M)
        self.assertTrue(torch.equal(result, expected_output))

    def test_v_repulsive_func(self):
        dist = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        distsq = dist ** 2
        expected_output = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)  # Manually derived
        result = self.oc.V_repulsive_func(distsq)
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-3))

    def test_v_attractive_func(self):
        distsq = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        expected_output = distsq
        result = self.oc.V_attractive_func(distsq)
        self.assertTrue(torch.equal(result, expected_output))

    def test_alpha_features(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        x_k_m = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32)
        alpha_indices = torch.tensor([0, 2], dtype=torch.int32)

        expected_output = torch.tensor([[[1.0, 2.0]], [[5.0, 6.0]]], dtype=torch.float32)
        result = self.oc.alpha_features(x, x_k_m, alpha_indices)
        self.assertTrue(torch.allclose(result, expected_output))

if __name__ == "__main__":
    unittest.main()
