import unittest
import torch
from fastgraphcompute.object_condensation import ObjectCondensation  # Replace with the actual import path

class TestObjectCondensation(unittest.TestCase):

    def setUp(self):
        self.oc = ObjectCondensation()
        self.device = torch.device("cpu")

    def test_scatter_to_N_indices(self):
        x_k_m = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)  # K=2, M=2
        asso_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32)  # N=4
        M = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)  # K x M

        # Expected output:
        # We need to map the values in x_k_m back to the original indices in asso_indices.
        # For cluster k=0:
        #   M[0] = [0, 1], x_k_m[0] = [1.0, 2.0]
        #   So indices 0 and 1 in the output should be 1.0 and 2.0 respectively.
        # For cluster k=1:
        #   M[1] = [2, 3], x_k_m[1] = [3.0, 4.0]
        #   So indices 2 and 3 in the output should be 3.0 and 4.0 respectively.
        expected_output = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        result = self.oc._scatter_to_N_indices(x_k_m.unsqueeze(-1), asso_indices, M)
        self.assertTrue(torch.equal(result, expected_output))

    def test_mean_per_row_split(self):
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)  # N=4
        row_splits = torch.tensor([0, 2, 4], dtype=torch.int32)  # Two batches: [0,2) and [2,4)

        # Expected output:
        # For the first batch (indices 0 and 1):
        #   Mean = (1.0 + 2.0) / 2 = 1.5
        # For the second batch (indices 2 and 3):
        #   Mean = (3.0 + 4.0) / 2 = 3.5
        expected_output = torch.tensor([1.5, 3.5], dtype=torch.float32)
        result = self.oc._mean_per_row_split(x, row_splits)
        self.assertTrue(torch.allclose(result, expected_output))

    def test_beta_loss(self):
        beta_k_m = torch.tensor([[[0.5], [0.4]], [[0.3], [0.2]]], dtype=torch.float32)  # K=2, M=2

        # Manual calculation:
        # For k=0:
        #   eps = 1e-3
        #   beta_values = [0.5, 0.4]
        #   LogSumExp approximation:
        #     max_beta = 0.5 / 1e-3 = 500.0
        #     sum_exp = exp((500.0 - 500.0)) + exp((400.0 - 500.0))
        #             = 1 + exp(-100.0) ≈ 1
        #     beta_penalty = 1 - eps * (500.0 + log(1)) = 1 - 0.001 * 500.0 = 0.5
        #   Additional penalty:
        #     1 - sum(beta_values) = 1 - (0.5 + 0.4) = 0.1
        #   Total beta_loss = 0.5 + 0.1 = 0.6

        # For k=1:
        #   beta_values = [0.3, 0.2]
        #   max_beta = 0.3 / 1e-3 = 300.0
        #   sum_exp ≈ 1 (similar reasoning)
        #   beta_penalty = 1 - 0.001 * 300.0 = 0.7
        #   Additional penalty:
        #     1 - (0.3 + 0.2) = 0.5
        #   Total beta_loss = 0.7 + 0.5 = 1.2

        expected_output = torch.tensor([[0.6], [1.2]], dtype=torch.float32)
        result = self.oc._beta_loss(beta_k_m)
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-3), f"result: {result}, expected_output: {expected_output}")

    def test_get_alpha_indices(self):
        beta_k_m = torch.tensor([[[0.1], [0.9]], [[0.5], [0.6]]], dtype=torch.float32)  # K=2, M=2
        M = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

        # Expected output:
        # For k=0:
        #   beta_k_m[0] = [0.1, 0.9], max at index 1, corresponding to M[0,1] = 1
        # For k=1:
        #   beta_k_m[1] = [0.5, 0.6], max at index 1, corresponding to M[1,1] = 3
        expected_output = torch.tensor([1, 3], dtype=torch.int32)
        result = self.oc.get_alpha_indices(beta_k_m, M)
        self.assertTrue(torch.equal(result, expected_output))

    def test_v_repulsive_func(self):
        dist = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        distsq = dist ** 2

        # Manual calculation:
        # V_repulsive = ReLU(1 - sqrt(distsq + 1e-6))
        # For distsq = 0.0:
        #   sqrt(0.0 + 1e-6) ≈ 0.001
        #   V = ReLU(1 - 0.001) = 0.999
        # For distsq = 0.25:
        #   sqrt(0.25 + 1e-6) ≈ 0.500
        #   V = ReLU(1 - 0.500) = 0.5
        # For distsq = 1.0:
        #   sqrt(1.0 + 1e-6) ≈ 1.000
        #   V = ReLU(1 - 1.000) = 0.0
        expected_output = torch.tensor([0.999, 0.5, 0.0], dtype=torch.float32)  # Manually derived
        result = self.oc.V_repulsive_func(distsq)
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-3))

    def test_v_attractive_func(self):
        distsq = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        expected_output = distsq
        result = self.oc.V_attractive_func(distsq)
        self.assertTrue(torch.equal(result, expected_output))

    def test_alpha_features(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)  # N=4, F=2
        x_k_m = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32)  # K=2, M=2, F=2
        alpha_indices = torch.tensor([0, 2], dtype=torch.int32)

        # Expected output:
        # Since weighted_obj_coordinates = 0 (default), alpha features are x[alpha_indices]
        # For alpha_indices = [0, 2], x[0] = [1.0, 2.0], x[2] = [5.0, 6.0]
        expected_output = torch.tensor([[[1.0, 2.0]], [[5.0, 6.0]]], dtype=torch.float32)
        result = self.oc.alpha_features(x, x_k_m, alpha_indices)
        self.assertTrue(torch.allclose(result, expected_output))

    def test_attractive_potential(self):
        beta_scale = torch.tensor([[0.5], [0.6]], dtype=torch.float32)  # beta_scale for all points, N=2
        beta_scale_k = torch.tensor([[0.8]], dtype=torch.float32)  # K=1
        coords_k = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)  # K x 1 x C
        coords_k_m = torch.tensor([[[1.0, 1.0], [1.1, 1.1]]], dtype=torch.float32)  # K x M x C
        M = torch.tensor([[0, 1]], dtype=torch.int32)

        # Manual calculation:
        # beta_scale_k_m = beta_scale[M] = [[0.5], [0.6]]
        beta_scale_k_m = torch.tensor([[[0.5], [0.6]]], dtype=torch.float32)
        # mask_k_m = ones since M >= 0
        mask_k_m = torch.ones_like(beta_scale_k_m)
        # Compute distances squared:
        # distsq_k_m = sum((coords_k - coords_k_m)^2, dim=2, keepdim=True)
        # For m=0: (1.0 - 1.0)^2 + (1.0 - 1.0)^2 = 0.0
        # For m=1: (1.0 - 1.1)^2 + (1.0 - 1.1)^2 = 0.01 + 0.01 = 0.02
        distsq_k_m = torch.tensor([[[0.0], [0.02]]], dtype=torch.float32)
        # V_attractive = mask_k_m * beta_scale_k * V_attractive_func(distsq_k_m) * beta_scale_k_m
        # V_attractive = 1 * 0.8 * distsq_k_m * beta_scale_k_m
        # For m=0: V = 1 * 0.8 * 0.0 * 0.5 = 0.0
        # For m=1: V = 1 * 0.8 * 0.02 * 0.6 = 0.0096
        # Sum over M: total_V = 0.0 + 0.0096 = 0.0096
        # Normalize by sum(mask_k_m): total_V / 2 = 0.0096 / 2 = 0.0048
        expected_output = torch.tensor([[0.0048]], dtype=torch.float32)  # Derived manually
        result = self.oc._attractive_potential(beta_scale, beta_scale_k, coords_k, coords_k_m, M)
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-4))

    def test_repulsive_potential(self):
        beta_scale = torch.tensor([[0.5], [0.6]], dtype=torch.float32)  # N=2
        beta_scale_k = torch.tensor([[0.7]], dtype=torch.float32)  # K=1
        coords = torch.tensor([[0.5, 0.5], [0.8, 0.8]], dtype=torch.float32)  # N=2, C=2
        coords_k = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)  # K x 1 x C
        M_not = torch.tensor([[0, 1]], dtype=torch.int32)

        # Manual calculation:
        # beta_scale_k_n = beta_scale[M_not] = [[0.5], [0.6]]
        beta_scale_k_n = torch.tensor([[[0.5], [0.6]]], dtype=torch.float32)
        # mask_k_n = ones since M_not >= 0
        mask_k_n = torch.ones_like(beta_scale_k_n)
        # Compute distances squared:
        # For n=0: (0.5 - 0.0)^2 + (0.5 - 0.0)^2 = 0.25 + 0.25 = 0.5
        # For n=1: (0.8 - 0.0)^2 + (0.8 - 0.0)^2 = 0.64 + 0.64 = 1.28
        distsq_k_n = torch.tensor([[[0.5], [1.28]]], dtype=torch.float32)
        # V_repulsive_func(distsq) = ReLU(1 - sqrt(distsq + 1e-6))
        # For n=0: sqrt(0.5 + 1e-6) ≈ 0.7071, V = 1 - 0.7071 = 0.2929
        # For n=1: sqrt(1.28 + 1e-6) ≈ 1.1314, V = 1 - 1.1314 = 0, ReLU gives 0
        # V_repulsive = mask_k_n * beta_scale_k * V_repulsive_func(distsq_k_n) * beta_scale_k_n
        # For n=0: V = 1 * 0.7 * 0.2929 * 0.5 ≈ 0.1025
        # For n=1: V = 1 * 0.7 * 0 * 0.6 = 0.0
        # Sum over N': total_V = 0.1025 + 0.0 = 0.1025
        # Normalize by sum(mask_k_n): total_V / 2 = 0.1025 / 2 = 0.05125
        expected_output = torch.tensor([[0.05125]], dtype=torch.float32)  # Derived manually
        result = self.oc._repulsive_potential(beta_scale, beta_scale_k, coords, coords_k, M_not)
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-4))

    def test_payload_scaling(self):
        beta = torch.tensor([[0.5], [0.6], [0.7]], dtype=torch.float32)
        asso_idx = torch.tensor([0, 0, 1], dtype=torch.int32)
        K_k = torch.tensor([[2], [2]], dtype=torch.float32)  # K=2
        M = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
        self.oc.beta_scaling_epsilon = 0.0  # To simplify calculations

        # Expected values computed manually:
        # pl_scaling = p_beta_scaling(beta / (1 + epsilon))
        # Since epsilon=0, beta / 1 = beta
        # p_beta_scaling is arctanhsq(beta), i.e., arctanh(beta)^2
        # arctanh(0.5) ≈ 0.5493, square ≈ 0.301
        # arctanh(0.6) ≈ 0.6931, square ≈ 0.4805
        # arctanh(0.7) ≈ 0.8673, square ≈ 0.7526

        # For cluster k=0:
        #   pl_scaling_k_m = [0.301, 0.4805], sum = 0.7815
        #   Normalized per object: [0.301/0.7815, 0.4805/0.7815] ≈ [0.3851, 0.6149]
        #   Divide by K_k[0] = 2: [0.3851/2., 0.6149/2.]
        # For cluster k=1:
        #   pl_scaling_k_m = [0.7526], sum = 0.7526
        #   Normalized: [1.0]
        #   Divide by K_k[0] = 2: [1.0/2.]
        # Scatter back to N indices:
        #   Index 0: pl_scaling[0] = 0.3851
        #   Index 1: pl_scaling[1] = 0.6149
        #   Index 2: pl_scaling[2] = 1.0
        pl_scaling_expected = torch.tensor([[0.3851/2.], [0.6149/2.], [1.0/2.]], dtype=torch.float32)

        result = self.oc._payload_scaling(beta, asso_idx, K_k, M).view(-1,1)
        self.assertTrue(torch.allclose(result, pl_scaling_expected, atol=1e-3), f"result: {result}, expected_output: {pl_scaling_expected}")

    def test_full_run(self):
        '''
        tests if it runs - does not test output yet
        '''
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        asso_indices = torch.tensor([[0], [1], [1], [-1]], dtype=torch.int32)
        beta = torch.tensor([[0.1], [0.9], [0.5], [0.6]], dtype=torch.float32)
        row_splits = torch.tensor([0, 2, 4], dtype=torch.int32)
        L_V, L_rep, L_b, pl_scaling, L_V_rep = self.oc(beta, x, asso_indices, row_splits)
        # expect scalar, scalar, scalar, (4, 1), (4, 1)
        self.assertIsInstance(L_V, torch.Tensor)
        self.assertIsInstance(L_rep, torch.Tensor)
        self.assertIsInstance(L_b, torch.Tensor)
        self.assertIsInstance(pl_scaling, torch.Tensor)
        self.assertIsInstance(L_V_rep, torch.Tensor)

        self.assertTrue(L_V.shape == torch.Size([]), f"L_V shape: {L_V.shape}, expected shape: {torch.Size([])}")
        self.assertTrue(L_rep.shape == torch.Size([]), f"L_rep shape: {L_rep.shape}, expected shape: {torch.Size([])}")
        self.assertTrue(L_b.shape == torch.Size([]), f"L_b shape: {L_b.shape}, expected shape: {torch.Size([])}")
        self.assertTrue(pl_scaling.shape == torch.Size([4, 1]), f"pl_scaling shape: {pl_scaling.shape}, expected shape: {torch.Size([4, 1])}")
        self.assertTrue(L_V_rep.shape == torch.Size([4, 1]), f"L_V_rep shape: {L_V_rep.shape}, expected shape: {torch.Size([4, 1])}")

if __name__ == "__main__":
    unittest.main()
