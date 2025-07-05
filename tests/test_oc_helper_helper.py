import torch
import unittest
import fastgraphcompute
import os.path as osp

# test torch.ops.oc_helper_helper.unique_with_counts against torch.unique
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(
    fastgraphcompute.extensions.__file__)), 'oc_helper_helper.so'))


class TestObjectCondensationHelperHelper(unittest.TestCase):

    def test_unique_with_counts(self):
        x = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4], dtype=torch.int64)
        unique, counts = torch.ops.oc_helper_helper.unique_with_counts(x)
        expected_unique, expected_counts = torch.unique(x, return_counts=True)
        # print both
        self.assertTrue(torch.equal(unique, expected_unique),
                        f'{unique=}, {expected_unique=}')
        self.assertTrue(torch.equal(counts, expected_counts),
                        f"{counts=}, {expected_counts=}")

    def test_small_tensor(self):
        x = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4], dtype=torch.int64)
        unique, counts = torch.ops.oc_helper_helper.unique_with_counts(x)
        expected_unique, expected_counts = torch.unique(x, return_counts=True)
        self.assertTrue(torch.equal(unique, expected_unique),
                        f'{unique=}, {expected_unique=}')
        self.assertTrue(torch.equal(counts, expected_counts),
                        f"{counts=}, {expected_counts=}")

    def test_large_tensor(self):
        # Generate a large tensor with duplicates
        num_elements = 100_000  # 100,000 elements
        num_unique = 1000       # Number of unique elements
        x = torch.randint(0, num_unique, (num_elements,), dtype=torch.int64)
        unique, counts = torch.ops.oc_helper_helper.unique_with_counts(x)
        expected_unique, expected_counts = torch.unique(x, return_counts=True)
        self.assertTrue(torch.equal(unique, expected_unique))
        self.assertTrue(torch.equal(counts, expected_counts))

    def test_negative_values(self):
        x = torch.tensor([-3, -1, -2, -3, -1, -2, -3, -1], dtype=torch.int64)
        unique, counts = torch.ops.oc_helper_helper.unique_with_counts(x)
        expected_unique, expected_counts = torch.unique(x, return_counts=True)
        self.assertTrue(torch.equal(unique, expected_unique))
        self.assertTrue(torch.equal(counts, expected_counts))

    # def test_empty_tensor(self):
    #    x = torch.tensor([], dtype=torch.int64)
    #    unique, counts = torch.ops.oc_helper_helper.unique_with_counts(x)
    #    expected_unique, expected_counts = torch.unique(x, return_counts=True)
    #    self.assertTrue(torch.equal(unique, expected_unique))
    #    self.assertTrue(torch.equal(counts, expected_counts))

    def test_single_element_tensor(self):
        x = torch.tensor([42], dtype=torch.int64)
        unique, counts = torch.ops.oc_helper_helper.unique_with_counts(x)
        expected_unique, expected_counts = torch.unique(x, return_counts=True)
        self.assertTrue(torch.equal(unique, expected_unique))
        self.assertTrue(torch.equal(counts, expected_counts))

    def ref_max_same_valued_entries_per_row_split(self, asso_idx, row_splits, filter_negative: bool = True):
        """
        This function calculates the maximum number of the same values in each row split.

        Args:
            asso_idx (torch.Tensor): A tensor containing indices (e.g., labels or values).
            row_splits (torch.Tensor): A tensor defining the start and end points of each row split.

        Returns:
            max_per_split (torch.Tensor): A tensor containing the maximum count of the same values for each row split.
            global_max (int): The global maximum count across all row splits.
            objects_per_split (torch.Tensor): A tensor containing the number of objects in each row split.
                                              Note: affected by filter_negative

        Notes:
            FIXME: Eventually this should be replaced by a C++/CUDA implementation to avoid the Python loop and enable jit.
        """

        n_row_splits = row_splits.size(0) - 1  # number of row splits
        max_per_split = torch.zeros(
            n_row_splits, dtype=torch.int64, device=asso_idx.device)
        objects_per_split = torch.zeros(
            n_row_splits, dtype=torch.int64, device=asso_idx.device)

        for rs_idx in range(n_row_splits):
            start_vertex = row_splits[rs_idx]
            end_vertex = row_splits[rs_idx + 1]

            # Extract the slice of asso_idx for the current row split
            asso_idx_slice = asso_idx[start_vertex:end_vertex]

            # Filter out negative values (asso_idx >= 0)
            if filter_negative:
                asso_idx_filtered = asso_idx_slice[asso_idx_slice >= 0]
            else:
                asso_idx_filtered = asso_idx_slice

            if asso_idx_filtered.numel() == 0:
                continue  # Skip if no valid indices in this split

            # Perform unique operation on the filtered slice and get counts
            unique_vals, counts = torch.unique(
                asso_idx_filtered, return_counts=True)

            # Find the maximum count and store it for this row split
            max_per_split[rs_idx] = counts.max()
            objects_per_split[rs_idx] = unique_vals.size(0)

        # Return the max_per_split and the global maximum value across all splits
        global_max = max_per_split.max()
        return max_per_split, global_max, objects_per_split

    def test_cpp_vs_python(self):
        asso_idx = torch.tensor([-1, -1, 2, 3, 3, 3, 8, 1, 0,
                                 90, 2, 2, 4, 5, -1, -1, 2
                                 ], dtype=torch.int64)
        row_splits = torch.tensor([0, 9, len(asso_idx)], dtype=torch.int64)

        r_mps, r_gm, r_ops = self.ref_max_same_valued_entries_per_row_split(
            asso_idx, row_splits, True)

        o_mps, o_gm, o_ops = torch.ops.oc_helper_helper.max_same_valued_entries_per_row_split(
            asso_idx, row_splits, True)

        self.assertTrue(torch.equal(
            r_mps, o_mps), f'{o_mps=}, {r_mps=}, {o_gm=}, {r_gm=}, {o_ops=}, {r_ops=}')
        self.assertTrue(torch.equal(
            r_gm, o_gm),  f'{o_mps=}, {r_mps=}, {o_gm=}, {r_gm=}, {o_ops=}, {r_ops=}')
        self.assertTrue(torch.equal(
            r_ops, o_ops),  f'{o_mps=}, {r_mps=}, {o_gm=}, {r_gm=}, {o_ops=}, {r_ops=}')


if __name__ == '__main__':
    unittest.main()
