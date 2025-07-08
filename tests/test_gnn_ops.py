import unittest
import torch
import io
from fastgraphcompute.gnn_ops import GravNetOp


class SimpleGravNetModel(torch.nn.Module):
    def __init__(self, in_dim, prop_dim, s, k):
        super(SimpleGravNetModel, self).__init__()
        self.gravnet = GravNetOp(
            in_channels=in_dim,
            out_channels=prop_dim,
            space_dimensions=s,
            k=k,
            propagate_dimensions=prop_dim,
            # Ensure that any additional arguments are TorchScript compatible
            optimization_arguments={}
        )
        self.scripted_gravnet = torch.jit.script(self.gravnet)
        self.fc = torch.nn.Linear(prop_dim, prop_dim)

    def forward(self, x, row_splits):
        x, *_ = self.scripted_gravnet(x, row_splits)
        x = self.fc(x)
        return x

class TestGravNetOp(unittest.TestCase):

    def test_jit_script_compatibility(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        in_dim, prop_dim, s, k = 8, 16, 3, 8
        model = SimpleGravNetModel(in_dim, prop_dim, s, k).to(device)
        
        try:
            # Test scripting the full model
            scripted_model = torch.jit.script(model)
            x = torch.randn(100, in_dim).to(device)
            row_splits = torch.tensor([0, 50, 100], dtype=torch.int32).to(device)

            # Test that both models work and produce consistent outputs
            with torch.no_grad():
                original_output = model(x, row_splits)
                scripted_output = scripted_model(x, row_splits)
            
            # Verify outputs are not None and have correct shape
            self.assertIsNotNone(original_output)
            self.assertIsNotNone(scripted_output)
            self.assertEqual(original_output.shape, (100, prop_dim))
            self.assertEqual(scripted_output.shape, (100, prop_dim))
            
            # Verify numerical equivalence
            self.assertTrue(torch.allclose(original_output, scripted_output, atol=1e-6))
            
            # Test save/load functionality
            buffer = io.BytesIO()
            torch.jit.save(scripted_model, buffer)
            buffer.seek(0)
            loaded_model = torch.jit.load(buffer)
            
            with torch.no_grad():
                loaded_output = loaded_model(x, row_splits)
            
            self.assertIsNotNone(loaded_output)
            self.assertEqual(loaded_output.shape, (100, prop_dim))
            self.assertTrue(torch.allclose(original_output, loaded_output, atol=1e-6))
            
            # Test that the GravNetOp itself can be scripted independently
            gravnet_op = GravNetOp(
                in_channels=in_dim,
                out_channels=prop_dim,
                space_dimensions=s,
                k=k,
                propagate_dimensions=prop_dim,
                optimization_arguments={}
            ).to(device)
            
            scripted_gravnet = torch.jit.script(gravnet_op)
            with torch.no_grad():
                gravnet_output, *_ = gravnet_op(x, row_splits)
                scripted_gravnet_output, *_ = scripted_gravnet(x, row_splits)
            
            self.assertTrue(torch.allclose(gravnet_output, scripted_gravnet_output, atol=1e-6))
            
        except Exception as e:
            self.fail(f"Failed to script GravNetOp: {str(e)}")

    def do_shape_test(self, device):

        for in_dim in [8, 12, 64]:
            for prop_dim in  [8, 12, 64]:
                for k in [8, 32]:
                    for s in [2,3,5]:
                       
                        op = GravNetOp(in_channels=in_dim,
                                       out_channels=2 * prop_dim,
                                       space_dimensions=s,
                                       k = k,
                                       propagate_dimensions=prop_dim,
                                       optimization_arguments={})
        
                        op.to(device)
                        x = torch.randn(100, in_dim).to(device)
                        row_splits = torch.tensor([0, 50, 80, 100], dtype=torch.int32).to(device)
                        output, *_ = op(x, row_splits)
                        self.assertEqual(output.shape, (100, 2*prop_dim))
    
    def test_shape_cpu(self):
        self.do_shape_test('cpu')
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_shape_cuda(self):
        self.do_shape_test('cuda')

    
    def test_jit_tracing_compatibility(self, device = 'cpu'):
        in_dim = 8
        prop_dim = 16
        k = 10
        s = 4

        model = SimpleGravNetModel(in_dim, prop_dim, s, k).to(device)
        model.eval()

        x = torch.randn(100, in_dim).to(device)
        row_splits = torch.tensor([0, 50, 100], dtype=torch.int32).to(device)

        model(x, row_splits) #run once in normal mode

        try:
            # Test scripting
            scripted_model = torch.jit.script(model)
            # Run the scripted model
            with torch.no_grad():
                output_scripted = scripted_model(x, row_splits)
            print("GravNet model scripting successful.")

            # Test tracing
            example_inputs = (x, row_splits)
            traced_model = torch.jit.trace(model, example_inputs)
            # Run the traced model
            with torch.no_grad():
                output_traced = traced_model(x, row_splits)
            print("GravNet model tracing successful.")

            # Compare outputs
            self.assertTrue(torch.allclose(output_scripted, output_traced, atol=1e-6), "Outputs differ between scripted and traced models.")

        except Exception as e:
            self.fail(f"GravNet model is not TorchScript compatible: {e}")



if __name__ == '__main__':
    unittest.main()