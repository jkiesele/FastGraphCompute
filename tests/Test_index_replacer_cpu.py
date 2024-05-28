import torch
import index_replacer_cpu
import unittest


class TestIndexReplacer(unittest.TestCase):

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
        

    if __name__ == "__main__":
        test_index0_replacer_cpu()
        test_index_replacer_out_of_range_cpu()
        
        
        
        
        
        
        

# import torch
# import index_replacer_cpu
# import unittest


# class TestIndexReplacer(unittest.TestCase):

#     def test_index0_replacer_cpu(self):
#         # Prepare input tensors
#         to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
#         replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
#         # Initialize replaced with to_be_replaced
#         replaced = to_be_replaced.clone()

#         # Call the function
#         index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

#         # Check the results
#         expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
#         print("Test basic index replacer passed:", torch.equal(replaced, expected_replaced))
        
#         diff = replaced - expected_replaced
#         self.assertTrue(diff==0)
        

#     def test_index_replacer_out_of_range_cpu(self):
#         # Prepare input tensors
#         to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32)  # 6 is out of range
#         replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
#         # Initialize replaced with to_be_replaced
#         replaced = to_be_replaced.clone()
        
#         # Call the function
#         index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

#         # Check the results
#         expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15, 6], dtype=torch.int32)  # 6 should be unchanged
#         print("Test index out of range passed:", torch.equal(replaced, expected_replaced))
        
#         diff = replaced - expected_replaced
#         self.assertTrue(diff==0)
        

#     if __name__ == "__main__":
#         test_index0_replacer_cpu()
#         test_index_replacer_out_of_range_cpu()
        

        










# import torch
# import index_replacer_cpu

# def test_index0_replacer_cpu():
#     # Prepare input tensors
#     to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
#     replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
#     # Initialize replaced with to_be_replaced
#     replaced = to_be_replaced.clone()

#     # Call the function
#     index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

#     # Print the tensors
#     print("to_be_replaced:", to_be_replaced)
#     print("replacements:", replacements)
#     print("replaced:", replaced)

#     # Check the results
#     expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
#     assert torch.equal(replaced, expected_replaced)

# def test_index_replacer_out_of_range_cpu():
#     # Prepare input tensors
#     to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32)  # 6 is out of range
#     replacements = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
#     # Initialize replaced with to_be_replaced
#     replaced = to_be_replaced.clone()

#     # Call the function
#     index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

#     # Print the tensors
#     print("to_be_replaced:", to_be_replaced)
#     print("replacements:", replacements)
#     print("replaced:", replaced)

#     # Check the results
#     expected_replaced = torch.tensor([10, 11, 12, 13, 14, 15, 6], dtype=torch.int32)  # 6 should be unchanged
#     assert torch.equal(replaced, expected_replaced)

# if __name__ == "__main__":
#     test_index0_replacer_cpu()
#     test_index_replacer_out_of_range_cpu()