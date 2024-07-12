import torch
import numpy as np
import unittest
import os.path as osp

# Load the shared libraries
cpu_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'index_replacer_cpu.so')
torch.ops.load_library(cpu_so_file)
cuda_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'index_replacer_cuda.so')
torch.ops.load_library(cuda_so_file)

class TestIndexReplacer(unittest.TestCase):

    # Tests for CPU
    def test_basic_index_replacer_cpu(self):
        to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
        replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        replaced = torch.ops.index_replacer_cpu.index_replacer_cpu(to_be_replaced, replacements)
        expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        self.assertTrue(torch.equal(replaced, expected_replaced), "Test basic index replacer CPU failed")

    def test_index_replacer_out_of_range_cpu(self):
        to_be_replaced = torch.tensor([0, 33, 2, 3, 17, 5, 6], dtype=torch.int32)  # 33 and 17 are out of range
        replacements = torch.tensor([10, 1, 12, 4, 2, 15], dtype=torch.int32)
        replaced = torch.ops.index_replacer_cpu.index_replacer_cpu(to_be_replaced, replacements)
        expected_replaced = torch.tensor([10, 33, 12, 4, 17, 15, 6], dtype=torch.int32)  # out-of-range should be unchanged
        self.assertTrue(torch.equal(replaced, expected_replaced), "Test index out of range CPU failed")

    def test_large_index_replacer_cpu(self):
        to_be_replaced = torch.tensor(np.random.randint(0, 10000, 10000), dtype=torch.int32)
        replacements = torch.tensor(np.random.randint(0, 10000, 10000), dtype=torch.int32)
        replaced = torch.ops.index_replacer_cpu.index_replacer_cpu(to_be_replaced, replacements)
        for i in range(10000):
            if to_be_replaced[i] < replacements.size(0):
                self.assertEqual(replaced[i].item(), replacements[to_be_replaced[i]].item(),
                                 "Test large index replacer CPU failed at index {}".format(i))
            else:
                self.assertEqual(replaced[i].item(), to_be_replaced[i].item(),
                                 "Test large index replacer CPU failed at index {}".format(i))

    # Tests for CUDA
    def test_basic_index_replacer_cuda(self):
        to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32).cuda()
        replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32).cuda()

        # Call the CUDA function via torch.ops
        replaced = torch.ops.index_replacer_cuda.index_replacer_cuda(to_be_replaced, replacements)

        expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32).cuda()
        self.assertTrue(torch.equal(replaced, expected_replaced), "Test basic index replacer CUDA failed")

    def test_index_replacer_out_of_range_cuda(self):
        to_be_replaced = torch.tensor([0, 33, 2, 3, 17, 5, 6], dtype=torch.int32).cuda()  # 33 and 17 are out of range
        replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32).cuda()

        # Call the CUDA function via torch.ops
        replaced = torch.ops.index_replacer_cuda.index_replacer_cuda(to_be_replaced, replacements)

        expected_replaced = torch.tensor([10, 33, 12, 13, 17, 15, 6], dtype=torch.int32).cuda()  # out-of-range should be unchanged
        self.assertTrue(torch.equal(replaced, expected_replaced), "Test index out of range CUDA failed")

    def test_large_index_replacer_cuda(self):
        to_be_replaced = torch.tensor(np.random.randint(0, 10000, 10000), dtype=torch.int32).cuda()
        replacements = torch.tensor(np.random.randint(0, 10000, 10000), dtype=torch.int32).cuda()

        # Call the CUDA function via torch.ops
        replaced = torch.ops.index_replacer_cuda.index_replacer_cuda(to_be_replaced, replacements)

        for i in range(10000):
            if to_be_replaced[i] < replacements.size(0):
                self.assertEqual(replaced[i].item(), replacements[to_be_replaced[i]].item(),
                                 "Test large index replacer CUDA failed at index {}".format(i))
            else:
                self.assertEqual(replaced[i].item(), to_be_replaced[i].item(),
                                 "Test large index replacer CUDA failed at index {}".format(i))

if __name__ == "__main__":
    unittest.main()