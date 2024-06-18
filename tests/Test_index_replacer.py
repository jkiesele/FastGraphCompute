from typing import Optional, Tuple
import torch
import index_replacer_cpu
import index_replacer_cuda
import os.path as osp
import unittest
import numpy as np


# Load the shared libraries
# cpu_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'index_replacer_cpu.so')
# cuda_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'index_replacer_cuda.so')



class TestIndexReplacer(unittest.TestCase):

#tests for index_replacer_cpu
    def test_index0_replacer_cpu(self):
        # Prepare input tensors
        to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
        replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        # Initialize replaced with to_be_replaced
        replaced = to_be_replaced.clone()

        # Call the function
        index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

        # Check the results
        expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        self.assertTrue(torch.equal(replaced, expected_replaced), "Test basic index replacer CPU failed")


    def test_index_replacer_out_of_range_cpu(self):
        # Prepare input tensors
        to_be_replaced = torch.tensor([0, 33, 2, 3, 17, 5, 6], dtype=torch.int32)  # 6 is out of range
        replacements = torch.tensor([10, 1, 12, 4, 2, 15], dtype=torch.int32)
        # Initialize replaced with to_be_replaced
        replaced = to_be_replaced.clone()
        
        # Call the function
        index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

        # Check the results
        expected_replaced = torch.tensor([10, 33, 12, 4, 17, 15, 6], dtype=torch.int32)  # 6 should be unchanged
        self.assertTrue(torch.equal(replaced, expected_replaced), "Test index out of range CPU failed")
        

    def test_large_index_replacer_cpu(self):
        # Prepare input tensors
        to_be_replaced = torch.tensor(np.random.randint(0, 10000, 10000), dtype=torch.int32)
        replacements = torch.tensor(np.random.randint(0, 10000, 10000), dtype=torch.int32)
        # Initialize replaced with to_be_replaced
        replaced = to_be_replaced.clone()

        # Call the function
        index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)
 
        # Check the results
        for i in range(10000):
            if to_be_replaced[i] < replacements.size(0):
                self.assertEqual(replaced[i].item(), replacements[to_be_replaced[i]].item(), "Test large index replacer CPU failed at index {}".format(i))
            else:
                self.assertEqual(replaced[i].item(), to_be_replaced[i].item(), "Test large index replacer CPU failed at index {}".format(i))

        

#tests for index_replacer_cuda
    def test_index0_replacer_cuda(self):
        # Prepare input tensors
        to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64).cuda()
        replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64).cuda()
        # Initialize replaced with to_be_replaced
        replaced = to_be_replaced.clone()

        # Call the function
        index_replacer_cuda.index_replacer(to_be_replaced, replacements, replaced, to_be_replaced.size(0), replacements.size(0))
        # torch.ops.index_replacer_cuda.index_replacer(to_be_replaced, replacements, replaced)


        # Check the results
        expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32).cuda()
        self.assertTrue(torch.equal(replaced, expected_replaced), "Test basic index replacer CUDA failed")


    def test_index_replacer_out_of_range_cuda(self):
        # Prepare input tensors
        to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64).cuda()  # 6 is out of range
        replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64).cuda()
        # Initialize replaced with to_be_replaced
        replaced = to_be_replaced.clone()

        # Call the function
        index_replacer_cuda.index_replacer(to_be_replaced, replacements, replaced, to_be_replaced.size(0), replacements.size(0))
        # torch.ops.index_replacer_cuda.index_replacer(to_be_replaced, replacements, replaced)

        # Check the results
        expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15, 6], dtype=torch.int32).cuda()  # 6 should be unchanged
        self.assertTrue(torch.equal(replaced, expected_replaced), "Test index out of range CUDA failed")


    def test_large_index_replacer_cuda(self):
        # Prepare input tensors
        to_be_replaced = torch.tensor(np.random.randint(0, 10000, 10000), dtype=torch.int64).cuda()
        replacements = torch.tensor(np.random.randint(0, 10000, 10000), dtype=torch.int64).cuda()
        # Initialize replaced with to_be_replaced
        replaced = to_be_replaced.clone()

        # Call the function
        index_replacer_cuda.index_replacer(to_be_replaced, replacements, replaced, to_be_replaced.size(0), replacements.size(0)) 

        # Check the results
        for i in range(10000):
            if to_be_replaced[i] < replacements.size(0):
                self.assertEqual(replaced[i].item(), replacements[to_be_replaced[i]].item(), "Test large index replacer CUDA failed at index {}".format(i))
            else:
                self.assertEqual(replaced[i].item(), to_be_replaced[i].item(), "Test large index replacer CUDA failed at index {}".format(i))
            
            
    if __name__ == "__main__":
        unittest.main()        
        # test_index0_replacer_cpu(self)
        # test_index_replacer_out_of_range_cpu(self)
        # test_index0_replacer_cuda(self)
        # test_index_replacer_out_of_range_cuda(self)