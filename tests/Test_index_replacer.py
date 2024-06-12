from typing import Optional, Tuple
import torch
import index_replacer_cpu
import index_replacer_cuda
import os.path as osp
import unittest

# Load the shared libraries
cpu_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'index_replacer_cpu.so')
cuda_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'index_replacer_cuda.so')



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
        print("Test basic index replacer CPU passed:", torch.equal(replaced, expected_replaced))


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
        print("Test index out of range CPU passed:", torch.equal(replaced, expected_replaced))
        
        
        

#tests for index_replacer_cuda
    def test_index0_replacer_cuda():
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
        print("Test basic index replacer CUDA passed:", torch.equal(replaced, expected_replaced))        


    def test_index_replacer_out_of_range_cuda():
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
        print("Test index out of range CUDA passed:", torch.equal(replaced, expected_replaced))
        

    if __name__ == "__main__":        
        test_index0_replacer_cpu()
        test_index_replacer_out_of_range_cpu()
        test_index0_replacer_cuda()
        test_index_replacer_out_of_range_cuda()









# import torch
# import unittest
# import os.path as osp
# import index_replacer_cpu

# # Load the shared libraries
# cpu_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'index_replacer_cpu.so')
# cuda_so_file = osp.join(osp.dirname(osp.realpath(__file__)), 'index_replacer_cuda.so')

# try:
#     torch.ops.load_library(cpu_so_file)
#     torch.ops.load_library(cuda_so_file)
# except Exception as e:
#     print(f"Failed to load shared libraries: {e}")
#     exit(1)

# class TestIndexReplacer(unittest.TestCase):
#     #tests for index_replacer_cpu
#     def test_index0_replacer_cpu():
#         # Prepare input tensors
#         to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
#         replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
#         # Initialize replaced with to_be_replaced
#         replaced = to_be_replaced.clone()

#         # Call the function
#         try:
#             if to_be_replaced.device == torch.device('cpu'):
#                 torch.ops.index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)
#             else:
#                 torch.ops.index_replacer_cuda.index_replacer_cuda(to_be_replaced, replacements, replaced)
#         except AttributeError as e:
#             print(f"Failed to call the function: {e}")
#             return

#         # Check the results
#         expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
#         assert torch.equal(replaced, expected_replaced), "Test basic index replacer CPU failed"



#     def test_index_replacer_out_of_range_cpu():
#         # Prepare input tensors
#         to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32)  # 6 is out of range
#         replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
#         # Initialize replaced with to_be_replaced
#         replaced = to_be_replaced.clone()
        
        
#         # Call the function
#         torch.ops.index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)


#         # Check the results
#         expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15, 6], dtype=torch.int32)  # 6 should be unchanged
#         print("Test index out of range passed:", torch.equal(replaced, expected_replaced))
        
    
    
    
    
#     #tests for index_replacer_cuda
#     def test_index0_replacer_cuda():
#         # Prepare input tensors
#         to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64).cuda()
#         replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64).cuda()
#         # Initialize replaced with to_be_replaced
#         replaced = to_be_replaced.clone()

#         # Call the function
#         if to_be_replaced.device == torch.device('cpu'):
#             torch.ops.index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)
#         else:
#             torch.ops.index_replacer_cuda.index_replacer_cuda(to_be_replaced, replacements, replaced)
        
#         # Check the results
#         expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64).cuda()
#         assert torch.equal(replaced, expected_replaced), "Test basic index replacer CUDA failed"        
    

#     def test_index_replacer_out_of_range_cuda():
#         # Prepare input tensors
#         to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64).cuda()  # 6 is out of range
#         replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64).cuda()
#         # Initialize replaced with to_be_replaced
#         replaced = to_be_replaced.clone()

#         # Call the function
#         if to_be_replaced.device == torch.device('cpu'):
#             torch.ops.index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)
#         else:
#             torch.ops.index_replacer_cuda.index_replacer_cuda(to_be_replaced, replacements, replaced)

#         # Check the results
#         expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15, 6], dtype=torch.int64).cuda()  # 6 should be unchanged
#         assert torch.equal(replaced, expected_replaced), "Test index replacer CUDA out of range failed"


# # if __name__ == "__main__":
# #     unittest.main()
        

#     if __name__ == "__main__":
#             # Print the attributes of torch.ops
#             print(dir(torch.ops.index_replacer_cpu))
        
#         # print("Using vars():")
#         # print(vars(torch.ops))

#         # print("Using __dict__:")
#         # print(torch.ops.__dict__)
        
#         test_index0_replacer_cpu()
#         test_index_replacer_out_of_range_cpu()
#         test_index0_replacer_cuda()
#         test_index_replacer_out_of_range_cuda()