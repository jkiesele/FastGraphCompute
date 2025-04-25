"""
fastgraphcompute.
"""

__version__ = "1.0"
__author__ = ''
__credits__ = ''

import torch
import os
import glob

# First, let's check if we have compiled extension libraries
extensions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extensions')

# Look for any compiled extension libraries
compiled_libraries = glob.glob(os.path.join(extensions_dir, '*.so'))

# If we have compiled libraries, load them
for lib_path in compiled_libraries:
    try:
        torch.ops.load_library(lib_path)
        print(f'Loaded extension: {os.path.basename(lib_path)}')
    except Exception as e:
        print(f'Failed to load {os.path.basename(lib_path)}: {e}')

# Now import the Python modules that may use these extensions
from .extensions.bin_by_coordinates import bin_by_coordinates
from .extensions.index_replacer import index_replacer
from .extensions.binned_select_knn import binned_select_knn
from .extensions.oc_helper import oc_helper_matrices, select_with_default

print("fastgraphcompute __init__.py loaded")