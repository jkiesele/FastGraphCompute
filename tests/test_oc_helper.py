import torch
import numpy as np
import unittest
import torch
from fastgraphcompute import oc_helper_matrices, select_with_default

class TestOcHelper(unittest.TestCase):

    def sort_matrices(self, M, M_not):
        # sort M in the second dimension using masked M
        M_masked = M.clone()
        max_in_m = M.max().item()
        M_masked[M_masked == -1] = max_in_m+100
        M = torch.gather(M, 1, M_masked.argsort(dim=1))

        #do the same to M_not
        M_not_masked = M_not.clone()
        max_in_m_not = M_not.max().item()
        M_not_masked[M_not_masked == -1] = max_in_m_not+100
        M_not = torch.gather(M_not, 1, M_not_masked.argsort(dim=1))

        # sort M in the first dimension using masked M
        sorting = M[:,0].argsort()
        return M[sorting], M_not[sorting]

    def run_matrix_test(self, device):
        #create some association indices, repeating the same values a few times and also adding some -1s
                                    # 0  1   2   3   4  5   6  7  8   9  10  11  12  13  14
        asso_indices = torch.tensor([ 3, 7,  3, -1, -1, 3,  7, 1, 1,  0, 0, -1,  1,  1,  -1,
                                    # 15 16  17  18  19 20  21...
                                      3, -1, 19, 3,  2, -1, 19], dtype=torch.int32, device=device)
        
        row_splits = torch.tensor([0, 15, len(asso_indices)], dtype=torch.int32, device=device)
        
        # in first row split, we have 
        # 3 times 3, 
        # 2 times 20, 
        # 4 times 1, 
        # 1 time 0, 
        # 3 times -1

        # in second row split, we have
        # 2 times 3
        # 1 time 2
        # 2 times 19
        # 2 times -1

        # in total we have 7 unique values, with the maximum being 20, the first row split as 4 unique values, the second 3

        # now let's call the function
        M, M_not, obj_per_rs = oc_helper_matrices(asso_indices, row_splits)
        
        #check if the number of objects per row split is correct
        self.assertTrue(torch.equal(obj_per_rs, torch.tensor([4, 3], dtype=torch.int32, device=device)), "Test oc_helper_matrices obj per rs failed: data wrong")

        # check if device is correct
        self.assertTrue(M.device.type == device, "Test oc_helper_matrices M device failed: device wrong, got "+M.device.type+" but expected "+device)
        self.assertTrue(M_not.device.type == device, "Test oc_helper_matrices M_not device failed: device wrong, got "+M_not.device.type+" but expected "+device)

        # check if dtype is same as asso_indices
        self.assertTrue(M.dtype == asso_indices.dtype, "Test oc_helper_matrices M dtype failed: dtype wrong, got "+str(M.dtype)+" but expected "+str(asso_indices.dtype))
        self.assertTrue(M_not.dtype == asso_indices.dtype, "Test oc_helper_matrices M_not dtype failed: dtype wrong, got "+str(M_not.dtype)+" but expected "+str(asso_indices.dtype))

        #check if the shape is expected: M: 7 x 4, M_not: 7 x 14
        self.assertTrue(M.shape == (7, 4), "Test oc_helper_matrices shape failed: shape wrong")
        self.assertTrue(M_not.shape == (7, 15), "Test oc_helper_matrices shape failed: shape wrong")

        # check if the data is correct. We expect that the matrix contains the indices of 
        # the entries with the same value for the asso index. For M, <0 are ignored.
        # the first row should be the indices of the 3s, the second row the indices of the 20s, etc.
        # however, there is no cross-talk between the row splits
        M_exp = torch.tensor([ 
              [0, 2, 5, -1],
              [1, 6, -1, -1],
              [7, 8, 12, 13],
              [9, 10, -1, -1],
              #next row split
              [15, 18, -1, -1],
              [17, 21, -1, -1],
              [19, -1, -1, -1]
            ],
            dtype=M.dtype, device=device
        )

        M_not_exp = torch.tensor(
            [
                #all not 3 in the first row split (not 0, 2 ,5)
                [1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, -1, -1, -1],
                #all not 7 in the first row split (not 1, 6)
                [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, -1, -1],
                #all not 1 in the first row split (not 7, 8, 12, 13)
                [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 14, -1, -1, -1, -1],
                #all not 0 in the first row split (not 9, 10)
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, -1, -1],
                # next row split, here there has to be at least a padding of 14-7 = 7 -1s
                 #all not 3 in the second row split (not 15, 18)
                [16, 17, 19, 20, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                 #all not 19 in the second row split (not 17, 21)
                [15, 16, 18, 19, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                 #all not 2 in the second row split (not 19)
                [15, 16, 17, 18, 20, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            ],
            dtype=M.dtype, device=device
        )

        # also here we need to sort the rows in the same way, according to M or M_exp, not M_not as the latter is 
        # ambiguous
        M_sorted, M_not_sorted = self.sort_matrices(M, M_not)
        M_exp_sorted, M_not_exp_sorted = self.sort_matrices(M_exp, M_not_exp)

        self.assertTrue(torch.equal(M_sorted, M_exp_sorted), "Test oc_helper_matrices data failed: data wrong. Expected: \n"+str(M_exp_sorted)+"\nbut got\n"+str(M_sorted))
    
        self.assertTrue(torch.equal(M_not_sorted, M_not_exp_sorted), "Test oc_helper_matrices data failed: data wrong. Expected: \n"+str(M_not_exp_sorted)+"\nbut got\n"+str(M_not_sorted))


    def test_oc_helper_matrices_cpu(self):
        self.run_matrix_test('cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_oc_helper_matrices_cuda(self):
        self.run_matrix_test('cuda')

    def run_select_with_default_test(self, device):
        #create some association indices, repeating the same values a few times and also adding some -1s
                                    # 0  1   2   3   4  5   6  7  8   9  10  11  12  13  14
        asso_indices = torch.tensor([ 3, 7,  3, -1, -1, 3,  7, 1, 1,  0, 0, -1,  1,  1,  -1,
                                    # 15 16  17  18  19 20  21...
                                      3, -1, 19, 3,  2, -1, 19], dtype=torch.int32, device=device)
        
        row_splits = torch.tensor([0, 15, len(asso_indices)], dtype=torch.int32, device=device)
        
        M, M_not, _ = oc_helper_matrices(asso_indices, row_splits)

        #self consistency test
        sel = select_with_default(M, asso_indices.unsqueeze(-1)# add one dim
                              , -100)  # Add one dimension to asso_indices
        #check if device is correct
        self.assertTrue(sel.device.type == device, "Test select_with_default self consistency failed: device wrong, got "+sel.device.type+" but expected "+device)

        # check if dtype is same as asso_indices
        self.assertTrue(sel.dtype == asso_indices.dtype, "Test select_with_default self consistency failed: dtype wrong, got "+str(sel.dtype)+" but expected "+str(asso_indices.dtype))

        #now every row in sel should contain the same value as M[:,0] or -100, test
        ok = sel == sel[:,0:1] 
        ok = ok | (sel == -100)
        self.assertTrue(torch.all(ok).item(), "Test select_with_default self consistency failed: data wrong")

        # now do the same but casting to float32
        f_asso_indices = asso_indices.float()
        sel = select_with_default(M, f_asso_indices.unsqueeze(-1)# add one dim
                              , -100)
        #check if data type is correct
        self.assertTrue(sel.dtype == f_asso_indices.dtype, "Test select_with_default self consistency failed: dtype wrong, got "+str(sel.dtype)+" but expected "+str(f_asso_indices.dtype))
        
        #now every row in sel should contain the same value as M[:,0] or -100, test
        ok = sel == sel[:,0:1]
        ok = ok | (sel == -100.)#now a float
        self.assertTrue(torch.all(ok).item(), "Test select_with_default self consistency failed for float: data wrong")

        # now to a large scale test
        # create some random indices
        asso_indices = torch.randint(0, 1000, (100000,), dtype=torch.int32, device=device)-1 # -1 to have some -1s
        row_splits = torch.tensor([0, len(asso_indices)], dtype=torch.int32, device=device) #but take one row split only
        M, M_not, _ = oc_helper_matrices(asso_indices, row_splits)
        sel = select_with_default(M, asso_indices.unsqueeze(-1), -100)
        #now every row in sel should contain the same value as M[:,0] or -100, test
        ok = sel == sel[:,0:1]
        ok = ok | (sel == -100)
        self.assertTrue(torch.all(ok).item(), "Test select_with_default large scale failed: data wrong")

        #now test with more than 1 dimension by repeating asso_indices.unsqueeze(-1) 10 times in the last dim
        r_asso_indices = asso_indices.unsqueeze(-1).repeat(1, 10)
        sel = select_with_default(M, r_asso_indices, -100)
        #now every row in sel should contain the same value as M[:,0] or -100, test
        ok = sel == sel[:,0:1]
        ok = ok | (sel == -100)
        self.assertTrue(torch.all(ok).item(), "Test select_with_default large scale multi-dim failed: data wrong")

    def run_large_scale_test(self, device):
        # set a deterministic seed
        torch.manual_seed(42)

        # create some random indices, random seems to do different things on cpu and gpu, so keep cpu here
        asso_indices = torch.randint(0, 100, (1000,), dtype=torch.int32, device='cpu') - 1 
        asso_indices = asso_indices.to(device)
        row_splits = torch.tensor([0, len(asso_indices)//4, len(asso_indices)//2, len(asso_indices)], dtype=torch.int32, device=device)
        M, M_not, _ = oc_helper_matrices(asso_indices, row_splits)
        
        #load the expected ones
        M_exp = torch.tensor(np.load("test_oc_helper_large_M.npy"), dtype=torch.int32, device=device)
        M_not_exp = torch.tensor(np.load("test_oc_helper_large_M_not.npy"), dtype=torch.int32, device=device)

        # sort all of them
        M_sorted, M_not_sorted = self.sort_matrices(M, M_not)
        M_exp_sorted, M_not_exp_sorted = self.sort_matrices(M_exp, M_not_exp)

        self.assertTrue(torch.equal(M_sorted, M_exp_sorted), f"Test oc_helper_matrices ({device}) large scale M data failed: data wrong.\n")
        self.assertTrue(torch.equal(M_not_sorted, M_not_exp_sorted), f"Test oc_helper_matrices ({device}) large scale M_not data failed: data wrong.\n")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cpu_against_cuda_matrices(self):
        def create_input():
            asso_indices = torch.randint(0, 100, (100000,), dtype=torch.int32, device='cpu') - 1
            row_splits = torch.tensor([0, len(asso_indices)//4, len(asso_indices)//2, len(asso_indices)], dtype=torch.int32, device='cpu')
            return asso_indices, row_splits
        
        asso_indices, row_splits = create_input()
        M_cpu, M_not_cpu, nper_cpu = oc_helper_matrices(asso_indices, row_splits) #cpu
        #move inputs to cuda
        asso_indices = asso_indices.to('cuda')
        row_splits = row_splits.to('cuda')
        M_cuda, M_not_cuda, nper_cuda = oc_helper_matrices(asso_indices, row_splits) #cuda

        # move all to gpu for faster sorting and comparison
        M_cpu = M_cpu.to('cuda')
        M_not_cpu = M_not_cpu.to('cuda')

        # sort all of them
        M_cpu_sorted, M_not_cpu_sorted = self.sort_matrices(M_cpu, M_not_cpu)
        M_cuda_sorted, M_not_cuda_sorted = self.sort_matrices(M_cuda, M_not_cuda)

        self.assertTrue(torch.equal(M_cpu_sorted, M_cuda_sorted), "Test oc_helper_matrices cpu vs cuda failed: M data wrong")
        self.assertTrue(torch.equal(M_not_cpu_sorted, M_not_cuda_sorted), "Test oc_helper_matrices cpu vs cuda failed: M_not data wrong")
        self.assertTrue(torch.equal(nper_cpu, nper_cuda), "Test oc_helper_matrices obj per rs cpu vs cuda failed: nper data wrong")

    def test_matrices_large_scale_cpu(self):
        self.run_large_scale_test('cpu')
    

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_matrices_large_scale_cuda(self):
        self.run_large_scale_test('cuda')

    def test_select_with_default_cpu(self):
        self.run_select_with_default_test('cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_select_with_default_cuda(self):
        self.run_select_with_default_test('cuda')


if __name__ == '__main__':
    unittest.main()