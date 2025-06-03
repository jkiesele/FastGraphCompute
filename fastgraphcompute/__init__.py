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

# Check for the C++ extension build directory (typically used with setup.py)
build_dirs = glob.glob(os.path.join(
    os.path.dirname(extensions_dir), 'build', '*'))
for build_dir in build_dirs:
    for lib_pattern in ['*.so', '*.dylib', '*.dll']:
        for lib_path in glob.glob(os.path.join(build_dir, lib_pattern)):
            try:
                torch.ops.load_library(lib_path)
                print(
                    f'Loaded extension from build dir: {os.path.basename(lib_path)}')
            except Exception as e:
                print(
                    f'Failed to load {os.path.basename(lib_path)} from build dir: {e}')

# Now import the Python modules that may use these extensions

print("fastgraphcompute __init__.py loaded")
