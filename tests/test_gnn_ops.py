import unittest
import torch
from ml4reco_modules.gnn_ops import GravNetOp

class TestGravNetOp(unittest.TestCase):

    def do_shape_test(self, device):

        for in_dim in [8, 12, 64]:
            for prop_dim in  [8, 12, 64]:
                for k in [8, 100]:
                    for s in [2,3,16]:
                       
                        op = GravNetOp(in_channels=in_dim,
                                       out_channels=2 * prop_dim,
                                       space_dimensions=s,
                                       k = k,
                                       propagate_dimensions=prop_dim,
                                       optimization_arguments={})
        
                        op.to(device)
                        x = torch.randn(100, in_dim).to(device)
                        row_splits = torch.tensor([0, 50, 80, 100], dtype=torch.int32).to(device)
                        output = op(x, row_splits)
                        self.assertEqual(output.shape, (100, 2*prop_dim))
    
    def test_shape_cpu(self):
        self.do_shape_test('cpu')

    def test_shape_cuda(self):
        self.do_shape_test('cuda')


if __name__ == '__main__':
    unittest.main()