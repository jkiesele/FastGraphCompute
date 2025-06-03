"""
fastgraphcompute.
"""

from .extensions.oc_helper import oc_helper_matrices, select_with_default
from .extensions.binned_select_knn import binned_select_knn
from .extensions.index_replacer import index_replacer
from .extensions.bin_by_coordinates import bin_by_coordinates
__version__ = "1.0"
__author__ = ''
__credits__ = ''

import torch
import os
import glob

# First, let's check if we have compiled extension libraries
extensions_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'extensions')

# Look for any compiled extension libraries
compiled_libraries = glob.glob(os.path.join(extensions_dir, '*.so'))

# If we have compiled libraries, load them
for lib_path in compiled_libraries:
    try:
        torch.ops.load_library(lib_path)
        print(f'Loaded extension: {os.path.basename(lib_path)}')
    except Exception as e:
        print(f'Failed to load {os.path.basename(lib_path)}: {e}')

print("fastgraphcompute __init__.py loaded")

# Check which extensions are actually loaded and working
print("\n=== FastGraphCompute Extension Status ===")

# List of expected operations from your extensions
expected_ops = [
    ('binned_select_knn_func', 'binned_select_knn_func'),
    ('binned_select_knn_grad_cpu', 'binned_select_knn_grad_cpu'),
    ('binned_select_knn_grad_cuda', 'binned_select_knn_grad_cuda'),
    ('bin_by_coordinates', 'bin_by_coordinates'),
    ('index_replacer', 'index_replacer'),
    ('select_knn', 'select_knn'),
    ('oc_helper', 'oc_helper'),
]

# Check which ops are available
loaded_extensions = []
missing_extensions = []

for namespace, op_name in expected_ops:
    try:
        # Check if the operation exists in torch.ops
        if hasattr(torch.ops, namespace) and hasattr(getattr(torch.ops, namespace), op_name):
            loaded_extensions.append(f"{namespace}.{op_name}")
        else:
            missing_extensions.append(f"{namespace}.{op_name}")
    except Exception as e:
        missing_extensions.append(f"{namespace}.{op_name} (error: {e})")

# Print summary
if loaded_extensions:
    print(f"✓ Loaded extensions ({len(loaded_extensions)}):")
    for ext in loaded_extensions:
        print(f"  - {ext}")

if missing_extensions:
    print(f"\n✗ Missing extensions ({len(missing_extensions)}):")
    for ext in missing_extensions:
        print(f"  - {ext}")

# Also check if Python modules imported successfully
print("\n=== Python Module Import Status ===")
try:
    from . import extensions
    print("✓ extensions module imported")
    
    # Check individual Python modules
    python_modules = ['oc_helper', 'binned_select_knn', 'index_replacer', 'bin_by_coordinates']
    for module in python_modules:
        if hasattr(extensions, module):
            print(f"✓ {module} module available")
        else:
            print(f"✗ {module} module NOT available")
except Exception as e:
    print(f"✗ Failed to import extensions module: {e}")

print("=====================================\n")
