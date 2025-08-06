# FastGraphCompute

FastGraphCompute is a high-performance extension for PyTorch designed to accelerate graph-based operations. It provides custom CUDA extensions for efficient computation in Graph Neural Networks (GNNs). The algorithms in this repository will be accompanied by a paper soon. Please leave room to cite it if you use this code.

## Installation

To install FastGraphCompute, ensure you have the following dependencies installed:

### Prerequisites
- **CUDA 12.1** (e.g., from container `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`)
- **CMake**
- **Git**
- **Python3 development tools**
- **PyTorch 2.5.0 with CUDA 12.1 support**

You can install the required dependencies using:

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install setuptools>=65 wheel>=0.43.0
```

For optional `torch_geometric` support:
```bash
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

### Installing FastGraphCompute

You can install FastGraphCompute directly from GitHub:
```bash
pip install git+https://github.com/jkiesele/FastGraphCompute
```

Alternatively, if developing locally, clone the repository and install:
```bash
git clone https://github.com/jkiesele/FastGraphCompute.git
cd FastGraphCompute
pip install .
```

## Usage

FastGraphCompute provides multiple modules for efficient graph-based operations. The key modules are:

### GravNetOp
Located in `fastgraphcompute/gnn_ops.py`, the `GravNetOp` implements a layer of the GravNet algorithm [arXiv:1902.07987] designed to learn local graph structures based on spatial coordinates.

#### Example Usage:
```python
import torch
from fastgraphcompute.gnn_ops import GravNetOp

model = GravNetOp(in_channels=8, out_channels=16, space_dimensions=4, propagate_dimensions=8, k=20)
input_tensor = torch.rand(32, 8)
# row split format, cutting the 32 x 8 array into individual samples / events
# one with 18 entries, one with 14 entries. 
row_splits = torch.tensor([0, 18, 32], dtype=torch.int32) 
output, neighbor_idx, distsq, S_space = model(input_tensor, row_splits)
print(output.shape)  # Expected output: (32, 16)
```

### Object Condensation Loss
Defined in `fastgraphcompute/object_condensation.py`, this module implements the object condensation loss [arXiv:2002.03605].

#### Example Usage:
```python
from fastgraphcompute.object_condensation import ObjectCondensation

loss_fn = ObjectCondensation(q_min=0.1, s_B=1.0)
# between 0 and 1
beta = torch.rand(32).unsqueeze(1) # by convention B x 1
coords = torch.rand(32, 3) 
# integers that determine the association of points to objects (>=0), -1 is reated as noise
asso_idx = torch.randint(0, 10, (32,), dtype=torch.int32).unsqueeze(1)
row_splits = torch.tensor([0, 18, 32], dtype=torch.int32) 
L_att, L_rep, L_beta, payload_scaling, _ = loss_fn(beta, coords, asso_idx, row_splits)
```

### Torch Geometric Utilities
The module `fastgraphcompute/torch_geometric_interface.py` provides functions that convert between the
`row_splits` representation used by FastGraphCompute and PyTorch Geometric's `batch` tensors.

#### Example
```python
import torch
from fastgraphcompute.torch_geometric_interface import (
    row_splits_from_strict_batch,
    strict_batch_from_row_splits,
)

batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int64)
row_splits = row_splits_from_strict_batch(batch)
assert torch.equal(strict_batch_from_row_splits(row_splits), batch)
```

### Low-Level Extensions
FastGraphCompute bundles several custom C++/CUDA extensions that expose high-performance kernels to PyTorch:

- `bin_by_coordinates` assigns points to spatial bins for efficient neighborhood queries.
- `binned_select_knn` performs k-nearest-neighbour searches using the binning scheme.
- `index_replacer` substitutes indices in tensors, useful for manipulating graph structures.

These extensions compile automatically during installation and are accessible via the
`fastgraphcompute.extensions` namespace.



## Development Guide

### Setting Up for Development

1. Clone the repository:
   ```bash
   git clone https://github.com/jkiesele/FastGraphCompute.git
   cd FastGraphCompute
   ```

2. Install dependencies manually (as there is no `requirements.txt`):
   ```bash
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
   pip install cmake setuptools wheel torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
   ```

3. Compile extensions:
   ```bash
   python setup.py develop
   ```

### Running Tests
To verify installation and correctness, run:
```bash
pytest tests/
```

### Contributing
- Ensure code adheres to PEP8.
- Use meaningful commit messages.
- Submit pull requests with detailed explanations.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments
FastGraphCompute is developed as part of research efforts in graph-based deep learning. Contributions are welcome!

