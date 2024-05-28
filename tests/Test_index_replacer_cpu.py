import torch

def test_index_replacer_cpu():
    from ml4reco_modules import index_replacer_cpu
    to_be_replaced = torch.tensor([0, 1, 2, 3, 4])
    replacements = torch.tensor([4, 3, 2, 1, 0])
    replaced = torch.zeros_like(to_be_replaced)

    index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

    expected_replaced = torch.tensor([4, 3, 2, 1, 0])
    torch.testing.assert_allclose(replaced, expected_replaced)
    
    



# class TestIndexReplacer(unittest.TestCase):
#     def test_index_replacer_cpu(self):
#         to_be_replaced = torch.tensor([0, 1, 2, 3, 4])
#         replacements = torch.tensor([4, 3, 2, 1, 0])
#         replaced = torch.zeros_like(to_be_replaced)

#         index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

#         expected_replaced = torch.tensor([4, 3, 2, 1, 0])
#         torch.testing.assert_allclose(replaced, expected_replaced)

#     def test_index_replacer_out_of_range(self):
#         to_be_replaced = torch.tensor([0, 1, 2, 3, 4, 5])
#         replacements = torch.tensor([4, 3, 2, 1, 0])
#         replaced = torch.zeros_like(to_be_replaced)

#         index_replacer_cpu.index_replacer(to_be_replaced, replacements, replaced)

#         expected_replaced = torch.tensor([4, 3, 2, 1, 0, 5])  # 5 is out of range, so it should be unchanged
#         torch.testing.assert_allclose(replaced, expected_replaced)

# if __name__ == '__main__':
#     unittest.main()