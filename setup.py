import os
import sys
import os.path as osp
from setuptools import setup, find_packages
from textwrap import dedent

import torch
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

# check if env variable DEV_MODE is set
DEV_MODE = os.getenv('DEV_MODE', '0') == '1'
if not DEV_MODE: #compile for all archs
    os.environ['TORCH_CUDA_ARCH_LIST'] = '5.2;7.0;7.5;8.0;8.6;8.9+PTX;9.0'  # 

CUDA_AVAILABLE = torch.cuda.is_available() and CUDA_HOME is not None

DO_CPU = True
DO_CUDA = CUDA_AVAILABLE

if os.getenv('FORCE_ONLY_CPU', '0') == '1':
    print('FORCE_ONLY_CPU: Only compiling CPU extensions')
    DO_CPU = True
    DO_CUDA = False
    def CUDAExtension(*args, **kwargs):
        print("CUDA is not available. Will not compile CUDA extensions.")
elif os.getenv('FORCE_ONLY_CUDA', '0') == '1':
    print('FORCE_ONLY_CUDA: Only compiling CUDA extensions')
    DO_CPU = False
    DO_CUDA = True
elif os.getenv('FORCE_CUDA', '0') == '1':
    print('FORCE_CUDA: Forcing compilation of CUDA extensions')
    if not CUDA_AVAILABLE: print(f'{CUDA_AVAILABLE=}, high chance of failure')
    DO_CUDA = True

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'

# Define extensions
extensions_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'extensions')
cpu_kwargs = dict(
    include_dirs=[extensions_dir],
    extra_compile_args={'cxx': ['-O2']},
    extra_link_args=['-s']
    )
# Check if the platform is macOS
if sys.platform == 'darwin':
    print("Setting MACOSX_DEPLOYMENT_TARGET to 10.13")
    # Append the compiler flag
    cpu_kwargs['extra_compile_args']['cxx'].append('-mmacosx-version-min=10.13')

# Define unified extensions
extensions_cpu = [
    CppExtension('fastgraphcompute.extensions.select_knn', 
                ['fastgraphcompute/extensions/select_knn.cpp'], 
                **cpu_kwargs),
    CppExtension('fastgraphcompute.extensions.index_replacer', 
                ['fastgraphcompute/extensions/index_replacer.cpp'], 
                **cpu_kwargs),
    CppExtension('fastgraphcompute.extensions.binned_select_knn_grad', 
                ['fastgraphcompute/extensions/binned_select_knn_grad.cpp'], 
                **cpu_kwargs),
    CppExtension('fastgraphcompute.extensions.oc_helper', 
                ['fastgraphcompute/extensions/oc_helper.cpp'], 
                **cpu_kwargs),
    CppExtension('fastgraphcompute.extensions.binned_knn_forward', 
                ['fastgraphcompute/extensions/binned_knn_forward.cpp'], 
                **cpu_kwargs)
]

cuda_kwargs = dict(
    include_dirs=[extensions_dir],
    extra_compile_args={
        'cxx': ['-O2'],
        'nvcc': [
            '--expt-relaxed-constexpr',
            '-O2',
            '--use_fast_math',
            # '-D__CUDACC__',  # Define CUDA compiler macro
            # '--compiler-options', '-fPIC',
            # '--extended-lambda',
            # '-I/usr/local/cuda/include'  # Add CUDA include path
        ]
    },
    extra_link_args=['-s']
    )
extensions_cuda = [
    CUDAExtension(
        'fastgraphcompute.extensions.select_knn_cuda',
        ['fastgraphcompute/extensions/select_knn_cuda.cpp', 'fastgraphcompute/extensions/select_knn_cuda_kernel.cu'],
        **cuda_kwargs
        ),
    CUDAExtension(
        'fastgraphcompute.extensions.index_replacer_cuda',
        ['fastgraphcompute/extensions/index_replacer_cuda.cpp','fastgraphcompute/extensions/index_replacer_cuda_kernel.cu'],
        **cuda_kwargs
        ),
    CUDAExtension(
        'fastgraphcompute.extensions.binned_select_knn_grad_cuda',
        ['fastgraphcompute/extensions/binned_select_knn_grad_cuda.cpp', 'fastgraphcompute/extensions/binned_select_knn_grad_cuda_kernel.cu'],
        **cuda_kwargs
        ),
    CUDAExtension(
        'fastgraphcompute.extensions.oc_helper_cuda',
        ['fastgraphcompute/extensions/oc_helper_cuda.cpp', 'fastgraphcompute/extensions/oc_helper_cuda_kernel.cu'],
        **cuda_kwargs
        ),
    ]

extensions = []

# Unified binned_select_knn
if DO_CUDA:
    extensions.append(CUDAExtension(
        'fastgraphcompute.extensions.binned_select_knn',
        ['fastgraphcompute/extensions/binned_select_knn.cpp',
         'fastgraphcompute/extensions/binned_select_knn_cpu.cpp',
         'fastgraphcompute/extensions/binned_select_knn_cuda_kernel.cu'],
        **cuda_kwargs
    ))
elif DO_CPU:
    extensions.append(CppExtension(
        'fastgraphcompute.extensions.binned_select_knn',
        ['fastgraphcompute/extensions/binned_select_knn.cpp',
         'fastgraphcompute/extensions/binned_select_knn_cpu.cpp'],
        **cpu_kwargs
    ))

# Unified bin_by_coordinates
if DO_CUDA:
    extensions.append(CUDAExtension(
        'fastgraphcompute.extensions.bin_by_coordinates',
        ['fastgraphcompute/extensions/bin_by_coordinates.cpp',
         'fastgraphcompute/extensions/bin_by_coordinates_cpu.cpp',
         'fastgraphcompute/extensions/bin_by_coordinates_cuda_kernel.cu'],
        **cuda_kwargs
    ))
elif DO_CPU:
    extensions.append(CppExtension(
        'fastgraphcompute.extensions.bin_by_coordinates',
        ['fastgraphcompute/extensions/bin_by_coordinates.cpp',
         'fastgraphcompute/extensions/bin_by_coordinates_cpu.cpp'],
        **cpu_kwargs
    ))

# Add remaining specific CPU or CUDA extensions
if DO_CPU: extensions.extend(extensions_cpu)
if DO_CUDA: extensions.extend(extensions_cuda)

# Print extensions
def repr_ext(ext):
    """
    Debug print for an extension
    """
    return dedent(f"""\
        {ext.name}
          sources: {', '.join(ext.sources)}
          extra_compile_args: {ext.extra_compile_args}
          extra_link_args: {ext.extra_link_args}
        """)

print('\n---------------------\nExtensions:')
for ext in extensions: print(repr_ext(ext))
print('---------------------')

# Number of parallel jobs, defaulting to all available CPUs if not specified
NUM_JOBS = os.getenv('NUM_PARALLEL_JOBS', str(os.cpu_count()))  # Default to all CPU cores
MAKE_ARGS = [f"-j{NUM_JOBS}"] if NUM_JOBS != "0" else []

setup(
    name='fastgraphcompute',
    ext_modules=extensions if not BUILD_DOCS else [],
    packages=find_packages(),  # Automatically find packages
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, 
                                                 use_ninja=True, 
                                                 parallel=True, 
                                                 make_args=MAKE_ARGS,
        cmake_process_args=[f"-DCMAKE_BUILD_PARALLEL_LEVEL={NUM_JOBS}"])
    },
)