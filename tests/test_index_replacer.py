import torch
import numpy as np
import unittest
import os.path as osp

from fastgraphcompute import index_replacer


class TestIndexReplacer(unittest.TestCase):

    def run_test(self, device):
        # set a numpy random seed
        np.random.seed(34)

        tbr = np.arange(100)
        #
        # replace a few entries with -1
        noise_entries = np.random.choice(100, 10, replace=False)
        tbr[noise_entries] = -1

        # shuffle the array
        np.random.shuffle(tbr)

        # reshape so it is nested
        tbr = tbr.reshape(20, 5)
        to_be_replaced = torch.tensor(tbr, dtype=torch.int64, device=device)

        # define replacements
        rpl = np.arange(100)
        np.random.shuffle(rpl)
        replacements = torch.tensor(rpl, dtype=torch.int64, device=device)

        # Call the function
        replaced = index_replacer(to_be_replaced, replacements)

        # check if device is correct
        self.assertTrue(replaced.device.type == device,
                        "Test index replacer device failed: device wrong")

        # check if dtype is same as to_be_replaced
        self.assertTrue(replaced.dtype == to_be_replaced.dtype,
                        "Test index replacer dtype failed: dtype wrong")

        # check if -1 are at the same place in the replaced array
        noise_ok = to_be_replaced[to_be_replaced <
                                  0] == replaced[to_be_replaced < 0]

        self.assertTrue(torch.all(noise_ok),
                        "Test index replacer noise failed")

        # now the remaining indices, simply use a reference array
        ref_array = torch.tensor([[-1, 13, 57, -1, 62],
                                  [31, 54, 16, -1, 75],
                                  [66, 19, 14, 20, 36],
                                  [46, 32, 44, 79, -1],
                                  [23, 11, 4, 72, 93],
                                  [61, 83, 27, 65, 50],
                                  [91, 92, 71, 1, 43],
                                  [89, 53, 10, 63, 7],
                                  [96, 6, 24, 64, 40],
                                  [15, 94, -1, 82, 68],
                                  [86, 90, -1, -1, 41],
                                  [97, 37, 70, 80, 78],
                                  [21, 58, 12, 28, 9],
                                  [22, 33, 51, 73, 55],
                                  [67, 76, -1, 25, 45],
                                  [85, 30, 38, 87, 81],
                                  [47, 69, 49, 48, 18],
                                  [17, 2, 74, 39, -1],
                                  [59, 84, 98, 95, 88],
                                  [29, 34, 60, 52, -1]], dtype=to_be_replaced.dtype, device=device)

        self.assertTrue(torch.all(ref_array == replaced),
                        "Test index replacer failed")

    # Tests for CPU

    def test_index_replacer_cpu(self):
        self.run_test('cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_index_replacer_cuda(self):
        self.run_test('cuda')


if __name__ == "__main__":
    unittest.main()
