import torch
import unittest
from fastgraphcompute.torch_geometric_interface import row_splits_from_strict_batch, strict_batch_from_row_splits

class TestRowSplitsFromStrictBatch(unittest.TestCase):
    
    def test_basic(self):
        batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int32)
        expected_row_splits = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
        row_splits = row_splits_from_strict_batch(batch)
        self.assertTrue(torch.equal(row_splits, expected_row_splits), f"Expected: {expected_row_splits}, Got: {row_splits}")
    
    def test_single_batch(self):
        batch = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
        expected_row_splits = torch.tensor([0, 4], dtype=torch.int32)
        row_splits = row_splits_from_strict_batch(batch)
        self.assertTrue(torch.equal(row_splits, expected_row_splits), f"Expected: {expected_row_splits}, Got: {row_splits}")
    
    def test_empty_batch(self):
        batch = torch.tensor([], dtype=torch.int32)
        expected_row_splits = torch.tensor([0,0], dtype=torch.int32)
        row_splits = row_splits_from_strict_batch(batch)
        self.assertTrue(torch.equal(row_splits, expected_row_splits), f"Expected: {expected_row_splits}, Got: {row_splits}")
    
    def test_batches_with_single_element(self):
        batch = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        expected_row_splits = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        row_splits = row_splits_from_strict_batch(batch)
        self.assertTrue(torch.equal(row_splits, expected_row_splits), f"Expected: {expected_row_splits}, Got: {row_splits}")


class TestStrictBatchFromRowSplits(unittest.TestCase):

    def test_basic(self):
        row_splits = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
        expected_batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int64)
        batch = strict_batch_from_row_splits(row_splits)
        self.assertTrue(torch.equal(batch, expected_batch), f"Expected: {expected_batch}, Got: {batch}")
    
    def test_single_batch(self):
        row_splits = torch.tensor([0, 4], dtype=torch.int32)
        expected_batch = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        batch = strict_batch_from_row_splits(row_splits)
        self.assertTrue(torch.equal(batch, expected_batch), f"Expected: {expected_batch}, Got: {batch}")
    
    def test_empty_input(self):
        row_splits = torch.tensor([0,0], dtype=torch.int32)
        expected_batch = torch.tensor([], dtype=torch.int64)
        batch = strict_batch_from_row_splits(row_splits)
        self.assertTrue(torch.equal(batch, expected_batch), f"Expected: {expected_batch}, Got: {batch}")
    
    def test_batches_with_single_element(self):
        row_splits = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        expected_batch = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        batch = strict_batch_from_row_splits(row_splits)
        self.assertTrue(torch.equal(batch, expected_batch), f"Expected: {expected_batch}, Got: {batch}")


if __name__ == '__main__':
    unittest.main()
