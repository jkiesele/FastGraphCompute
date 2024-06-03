import torch
import index_replacer_cpu
# import index_replacer_cuda
import unittest
import os.path as osp

def load_ops(so_file):
    if not osp.isfile(so_file):
        logger.error(f'Could not load op: No file {so_file}')
    else:
        torch.ops.load_library(so_file)


THISDIR = osp.dirname(osp.abspath(__file__))
load_ops(osp.join(THISDIR, "../select_knn_cpu.so"))
load_ops(osp.join(THISDIR, "../select_knn_cuda.so"))


class TestIndexReplacer(unittest.TestCase):


    #tests for index_replacer_cpu
    def test_index0_replacer_cpu():
        # Prepare input tensors
        to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
        replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        # Initialize replaced with to_be_replaced
        replaced = to_be_replaced.clone()

        # Call the function
        index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

        # Check the results
        expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        print("Test basic index replacer passed:", torch.equal(replaced, expected_replaced))

    def test_index_replacer_out_of_range_cpu():
        # Prepare input tensors
        to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32)  # 6 is out of range
        replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        # Initialize replaced with to_be_replaced
        replaced = to_be_replaced.clone()
        
        # Call the function
        index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

        # Check the results
        expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15, 6], dtype=torch.int32)  # 6 should be unchanged
        print("Test index out of range passed:", torch.equal(replaced, expected_replaced))
        
    
    
    
    
    #tests for index_replacer_cuda
    def test_index0_replacer_cuda(self):
        # Prepare input tensors
        to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64).cuda()
        replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64).cuda()
        # Initialize replaced with to_be_replaced
        replaced = to_be_replaced.clone()

        # Call the function
        index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

        # Check the results
        expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64).cuda()
        self.assertTrue(torch.equal(replaced, expected_replaced), "Test basic index replacer CUDA failed")

    def test_index_replacer_out_of_range_cuda(self):
        # Prepare input tensors
        to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64).cuda()  # 6 is out of range
        replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64).cuda()
        # Initialize replaced with to_be_replaced
        replaced = to_be_replaced.clone()

        # Call the function
        index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

        # Check the results
        expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15, 6], dtype=torch.int64).cuda()  # 6 should be unchanged
        self.assertTrue(torch.equal(replaced, expected_replaced), "Test index replacer CUDA out of range failed")


    # if __name__ == "__main__":
    #     unittest.main()
        

    if __name__ == "__main__":
        test_index0_replacer_cpu()
        test_index_replacer_out_of_range_cpu()
        test_index0_replacer_cuda()
        test_index_replacer_out_of_range_cuda()