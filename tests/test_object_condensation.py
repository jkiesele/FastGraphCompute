import numpy as np
import torch.optim as optim
import torch.nn as nn
import unittest
import torch
# Replace with the actual import path
from fastgraphcompute.object_condensation import ObjectCondensation


class TestObjectCondensation(unittest.TestCase):

    def setUp(self):
        self.oc = ObjectCondensation()
        self.device = torch.device("cpu")

    def test_scatter_to_N_indices(self):
        x_k_m = torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                             dtype=torch.float32)  # K=2, M=2
        asso_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int64)  # N=4
        M = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)  # K x M

        # Expected output:
        # We need to map the values in x_k_m back to the original indices in asso_indices.
        # For cluster k=0:
        #   M[0] = [0, 1], x_k_m[0] = [1.0, 2.0]
        #   So indices 0 and 1 in the output should be 1.0 and 2.0 respectively.
        # For cluster k=1:
        #   M[1] = [2, 3], x_k_m[1] = [3.0, 4.0]
        #   So indices 2 and 3 in the output should be 3.0 and 4.0 respectively.
        expected_output = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        result = self.oc._scatter_to_N_indices(
            x_k_m.unsqueeze(-1), asso_indices, M)
        self.assertTrue(torch.equal(result, expected_output))

    def test_mean_per_row_split(self):
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]],
                         dtype=torch.float32)  # N=4
        # Two batches: [0,2) and [2,4)
        row_splits = torch.tensor([0, 2, 4], dtype=torch.int64)

        # Expected output:
        # For the first batch (indices 0 and 1):
        #   Mean = (1.0 + 2.0) / 2 = 1.5
        # For the second batch (indices 2 and 3):
        #   Mean = (3.0 + 4.0) / 2 = 3.5
        expected_output = torch.tensor([1.5, 3.5], dtype=torch.float32)
        result = self.oc._mean_per_row_split(x, row_splits)
        self.assertTrue(torch.allclose(result, expected_output))

    def test_beta_loss(self):
        beta_k_m = torch.tensor(
            [[[0.5], [0.4]], [[0.3], [0.2]]], dtype=torch.float32)  # K=2, M=2

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
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-3),
                        f"result: {result}, expected_output: {expected_output}")

    def test_get_alpha_indices(self):
        beta_k_m = torch.tensor(
            [[[0.1], [0.9]], [[0.5], [0.6]]], dtype=torch.float32)  # K=2, M=2
        M = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

        # Expected output:
        # For k=0:
        #   beta_k_m[0] = [0.1, 0.9], max at index 1, corresponding to M[0,1] = 1
        # For k=1:
        #   beta_k_m[1] = [0.5, 0.6], max at index 1, corresponding to M[1,1] = 3
        expected_output = torch.tensor([1, 3], dtype=torch.int64)
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
        expected_output = torch.tensor(
            [0.999, 0.5, 0.0], dtype=torch.float32)  # Manually derived
        result = self.oc.V_repulsive_func(distsq)
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-3))

    def test_v_attractive_func(self):
        distsq = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        expected_output = distsq
        result = self.oc.V_attractive_func(distsq)
        self.assertTrue(torch.equal(result, expected_output))

    def test_alpha_features(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [
                         7.0, 8.0]], dtype=torch.float32)  # N=4, F=2
        x_k_m = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [
                             7.0, 8.0]]], dtype=torch.float32)  # K=2, M=2, F=2
        alpha_indices = torch.tensor([0, 2], dtype=torch.int64)

        # Expected output:
        # Since weighted_obj_coordinates = 0 (default), alpha features are x[alpha_indices]
        # For alpha_indices = [0, 2], x[0] = [1.0, 2.0], x[2] = [5.0, 6.0]
        expected_output = torch.tensor(
            [[[1.0, 2.0]], [[5.0, 6.0]]], dtype=torch.float32)
        result = self.oc.alpha_features(x, x_k_m, alpha_indices)
        self.assertTrue(torch.allclose(result, expected_output))

    def test_attractive_potential(self):
        # beta_scale for all points, N=2
        beta_scale = torch.tensor([[0.5], [0.6]], dtype=torch.float32)
        beta_scale_k = torch.tensor([[0.8]], dtype=torch.float32)  # K=1
        coords_k = torch.tensor(
            [[[1.0, 1.0]]], dtype=torch.float32)  # K x 1 x C
        coords_k_m = torch.tensor(
            [[[1.0, 1.0], [1.1, 1.1]]], dtype=torch.float32)  # K x M x C
        M = torch.tensor([[0, 1]], dtype=torch.int64)

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
        expected_output = torch.tensor(
            [[0.0048]], dtype=torch.float32)  # Derived manually
        result = self.oc._attractive_potential(
            beta_scale, beta_scale_k, coords_k, coords_k_m, M)
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-4))

    def test_repulsive_potential(self):
        beta_scale = torch.tensor([[0.5], [0.6]], dtype=torch.float32)  # N=2
        beta_scale_k = torch.tensor([[0.7]], dtype=torch.float32)  # K=1
        coords = torch.tensor([[0.5, 0.5], [0.8, 0.8]],
                              dtype=torch.float32)  # N=2, C=2
        coords_k = torch.tensor(
            [[[0.0, 0.0]]], dtype=torch.float32)  # K x 1 x C
        M_not = torch.tensor([[0, 1]], dtype=torch.int64)

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
        expected_output = torch.tensor(
            [[0.05125]], dtype=torch.float32)  # Derived manually
        result = self.oc._repulsive_potential(
            beta_scale, beta_scale_k, coords, coords_k, M_not)
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-4))

    def test_payload_scaling(self):
        beta = torch.tensor([[0.5], [0.6], [0.7]], dtype=torch.float32)
        asso_idx = torch.tensor([0, 0, 1], dtype=torch.int64)
        K_k = torch.tensor([[2], [2]], dtype=torch.float32)  # K=2
        M = torch.tensor([[0, 1], [2, -1]], dtype=torch.int64)
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
        pl_scaling_expected = torch.tensor(
            [[0.3851/2.], [0.6149/2.], [1.0/2.]], dtype=torch.float32)

        result = self.oc._payload_scaling(beta, asso_idx, K_k, M).view(-1, 1)
        self.assertTrue(torch.allclose(result, pl_scaling_expected, atol=1e-3),
                        f"result: {result}, expected_output: {pl_scaling_expected}")

    def test_full_run(self):
        '''
        tests if it runs - does not test output yet
        '''
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                         [7.0, 8.0]], dtype=torch.float32)
        asso_indices = torch.tensor([[0], [1], [1], [-1]], dtype=torch.int64)
        beta = torch.tensor([[0.1], [0.9], [0.5], [0.6]], dtype=torch.float32)
        row_splits = torch.tensor([0, 2, 4], dtype=torch.int64)
        L_V, L_rep, L_b, pl_scaling, L_V_rep = self.oc(
            beta, x, asso_indices, row_splits)
        # expect scalar, scalar, scalar, (4, 1), (4, 1)
        self.assertIsInstance(L_V, torch.Tensor)
        self.assertIsInstance(L_rep, torch.Tensor)
        self.assertIsInstance(L_b, torch.Tensor)
        self.assertIsInstance(pl_scaling, torch.Tensor)
        self.assertIsInstance(L_V_rep, torch.Tensor)

        self.assertTrue(L_V.shape == torch.Size(
            []), f"L_V shape: {L_V.shape}, expected shape: {torch.Size([])}")
        self.assertTrue(L_rep.shape == torch.Size(
            []), f"L_rep shape: {L_rep.shape}, expected shape: {torch.Size([])}")
        self.assertTrue(L_b.shape == torch.Size(
            []), f"L_b shape: {L_b.shape}, expected shape: {torch.Size([])}")
        self.assertTrue(pl_scaling.shape == torch.Size(
            [4, 1]), f"pl_scaling shape: {pl_scaling.shape}, expected shape: {torch.Size([4, 1])}")
        self.assertTrue(L_V_rep.shape == torch.Size(
            [4, 1]), f"L_V_rep shape: {L_V_rep.shape}, expected shape: {torch.Size([4, 1])}")

    def test_full_run_cuda(self):
        '''
        test if it runs on CUDA without errors
        '''
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [
                         7.0, 8.0]], dtype=torch.float32).to('cuda')
        asso_indices = torch.tensor(
            [[0], [1], [1], [-1]], dtype=torch.int64).to('cuda')
        beta = torch.tensor([[0.1], [0.9], [0.5], [0.6]],
                            dtype=torch.float32).to('cuda')
        row_splits = torch.tensor([0, 2, 4], dtype=torch.int64).to('cuda')
        L_V, L_rep, L_b, pl_scaling, L_V_rep = self.oc(
            beta, x, asso_indices, row_splits)
        self.assertTrue(L_V.device.type == 'cuda')
        self.assertTrue(L_rep.device.type == 'cuda')
        self.assertTrue(L_b.device.type == 'cuda')
        self.assertTrue(pl_scaling.device.type == 'cuda')
        self.assertTrue(L_V_rep.device.type == 'cuda')


class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc_beta = nn.Linear(16, 1)    # Outputs beta
        self.fc_coords = nn.Linear(16, 2)  # Outputs coordinates in 2D space

    def forward(self, x):
        x = self.relu(self.fc1(x))
        beta = torch.sigmoid(self.fc_beta(x))  # Beta values between 0 and 1
        coords = self.fc_coords(x)             # Coordinates in 2D space
        return beta, coords


class TestObjectCondensationTraining(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.num_points = 100
        self.num_objects = 5

        # Generate synthetic data
        torch.manual_seed(42)  # For reproducibility

        # Generate cluster centers
        cluster_centers = torch.randn(
            self.num_objects, 2) * 5.0  # 5 clusters in 2D space

        # Assign points to clusters
        points_per_cluster = self.num_points // self.num_objects
        data = []
        asso_indices = []
        for i in range(self.num_objects):
            # Generate points around the cluster center
            cluster_points = cluster_centers[i] + \
                torch.randn(points_per_cluster, 2)
            data.append(cluster_points)
            asso_indices.extend([i] * points_per_cluster)
        data = torch.vstack(data)
        asso_indices = torch.tensor(
            asso_indices, dtype=torch.int64).unsqueeze(1)

        # Add noise points
        num_noise = self.num_points - len(asso_indices)
        if num_noise > 0:
            noise_points = torch.randn(
                num_noise, 2) * 10.0  # Noise spread out more
            data = torch.vstack([data, noise_points])
            asso_indices = torch.vstack(
                [asso_indices, torch.full((num_noise, 1), -1, dtype=torch.int64)])

        # Shuffle the data
        perm = torch.randperm(len(data))
        data = data[perm]
        asso_indices = asso_indices[perm]

        # Concatenate data and asso_indices
        self.features = torch.cat([data, asso_indices.float()], dim=1).to(
            self.device)  # Input features
        self.asso_indices = asso_indices.to(self.device)

        # Update model input dimension
        input_dim = self.features.shape[1]
        self.model = SimpleNet(input_dim).to(self.device)
        self.oc_loss = ObjectCondensation().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        # Define row splits for a single batch (since data is small)
        self.row_splits = torch.tensor(
            [0, len(self.features)], dtype=torch.int64).to(self.device)

    def test_training(self):
        num_epochs = 500
        self.model.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            # Forward pass
            beta, coords = self.model(self.features)

            # Compute loss
            L_V, L_rep, L_b, pl_scaling, L_V_rep = self.oc_loss(
                beta, coords, self.asso_indices, self.row_splits)
            loss = L_V + L_rep + L_b  # For simplicity, not including payload loss

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Optionally, print loss every 10 epochs
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # After training, evaluate the model
        self.model.eval()
        with torch.no_grad():
            beta, coords = self.model(self.features)

            # Identify alpha indices (points with highest beta in each object)
            beta_np = beta.cpu().numpy()
            asso_np = self.asso_indices.cpu().numpy().squeeze()
            coords_np = coords.cpu().numpy()

            # For each object, find the point with the highest beta
            alpha_indices = {}
            for idx in range(len(beta_np)):
                obj_id = asso_np[idx]
                if obj_id < 0:
                    continue  # Skip noise
                beta_value = beta_np[idx, 0]
                if obj_id not in alpha_indices or beta_value > beta_np[alpha_indices[obj_id], 0]:
                    alpha_indices[obj_id] = idx

            # Check that alpha points have beta close to 1
            for obj_id, idx in alpha_indices.items():
                beta_value = beta_np[idx, 0]
                self.assertGreaterEqual(
                    beta_value, 0.9, f"Alpha point for object {obj_id} has beta {beta_value}, expected >= 0.9")

            # Check that noise points have low beta
            noise_indices = (asso_np == -1)
            noise_beta = beta_np[noise_indices]
            self.assertTrue((noise_beta <= 0.1).all(),
                            "Not all noise points have low beta")

            # Check that points in the same object are close to the alpha point
            for obj_id in range(self.num_objects):
                alpha_idx = alpha_indices.get(obj_id, None)
                if alpha_idx is None:
                    continue  # No points for this object
                alpha_coord = coords_np[alpha_idx]
                # Get indices of points in the same object
                obj_indices = (asso_np == obj_id)
                obj_coords = coords_np[obj_indices]
                # Compute distances to alpha point
                distances = np.linalg.norm(obj_coords - alpha_coord, axis=1)
                max_distance = distances.max()
                self.assertLessEqual(
                    max_distance, 0.5, f"Points in object {obj_id} are not close to the alpha point")

            # Check that payload scaling behaves accordingly
            pl_scaling_np = pl_scaling.cpu().numpy()
            # For each object, payload scaling should be higher for points with higher beta
            for obj_id in range(self.num_objects):
                obj_indices = (asso_np == obj_id)
                obj_beta = beta_np[obj_indices]
                obj_pl_scaling = pl_scaling_np[obj_indices]
                # Check that payload scaling is monotonically increasing with beta
                sorted_indices = np.argsort(obj_beta[:, 0])
                sorted_pl_scaling = obj_pl_scaling[sorted_indices]
                self.assertTrue(np.all(np.diff(
                    sorted_pl_scaling[:, 0]) >= -1e-5), f"Payload scaling not increasing with beta in object {obj_id}")

            print("Model training and evaluation completed successfully.")


if __name__ == "__main__":
    import numpy as np
    unittest.main()
