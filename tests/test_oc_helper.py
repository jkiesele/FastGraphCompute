import torch
import numpy as np
import unittest
import torch
from ml4reco_modules import oc_helper_matrices, select_with_default

class TestOcHelper(unittest.TestCase):

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

        # in total we have 7 unique values, with the maximum being 20

        # now let's call the function
        M, M_not = oc_helper_matrices(asso_indices, row_splits)
        # sort M in the second dimension using masked M
        M_masked = M.clone()
        M_masked[M_masked == -1] = 1000
        M = torch.gather(M, 1, M_masked.argsort(dim=1))

        #do the same to M_not
        M_not_masked = M_not.clone()
        M_not_masked[M_not_masked == -1] = 1000
        M_not = torch.gather(M_not, 1, M_not_masked.argsort(dim=1))

        # this is just for comparison. The sorting is not important for the functionality


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

        # check if M and M_exp they are the same.
        # however, any permutation of the vectors in dim=0 also counts as valid, so we need to sort the rows
        # and then compare, sort M simply by using the first entry in each row, this needs an argsort
        M_sorted = M[M[:,0].argsort()]
        M_exp_sorted = M_exp[M_exp[:,0].argsort()]


        self.assertTrue(torch.equal(M_sorted, M_exp_sorted), "Test oc_helper_matrices data failed: data wrong. Expected: \n"+str(M_exp_sorted)+"\nbut got\n"+str(M_sorted))
        
        # M_not should contain the indices of the entries with different values for the asso index.
        # also here, row split boundaries are not crossed
        # The first row would correspond to all indices in the first row split that are not 3, 
        # the second row to all indices that are not 20, etc. The matrix is padded with -1s

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
        M_not_sorted = M_not[M[:,0].argsort()]
        M_not_exp_sorted = M_not_exp[M_exp[:,0].argsort()]

        self.assertTrue(torch.equal(M_not_sorted, M_not_exp_sorted), "Test oc_helper_matrices data failed: data wrong. Expected: \n"+str(M_not_exp_sorted)+"\nbut got\n"+str(M_not_sorted))


    def test_oc_helper_matrices_cpu(self):
        self.run_matrix_test('cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_oc_helper_matrices_cuda(self):
        self.run_matrix_test('cuda')



if __name__ == '__main__':
    # Example input data
    asso_indices = torch.tensor([3, 7,  3, -1, -1, 3, 7, 1, 1, 0, 0, -1, 1, 1, -1,
                                 3, -1, 19, 3, 2, -1, 19], dtype=torch.int32)
        
    row_splits = torch.tensor([0, 15, len(asso_indices)], dtype=torch.int32)

    # Assuming you have a function to create matrices M, M_not
    M, M_not = oc_helper_matrices(asso_indices, row_splits)
    print("M:", M)
    print("M_not:", M_not)

    # Create a sample 2D tensor (V x F)
    features = torch.arange(5*len(asso_indices)).view(len(asso_indices), 5) # Example tensor with shape (len(asso_indices), 5)
    #cast to float32
    features = features.to(torch.float32)

    # Select with default value of 0
    sel = select_with_default(M, features, -100)  # Add one dimension to asso_indices
    print("sel:", sel)
    sel = select_with_default(M, asso_indices.unsqueeze(-1)# add one dim
                              , -100)  # Add one dimension to asso_indices
    print("sel tidx:", sel) #this should close, so only same indices per row or -1s
