import os
import os.path as osp
from setuptools import setup, find_packages
from textwrap import dedent

import torch
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

# uncomment when going to deployment mode
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.0;8.6;8.9+PTX'  # 

CUDA_AVAILABLE = torch.cuda.is_available() and CUDA_HOME is not None

DO_CPU = True
DO_CUDA = CUDA_AVAILABLE

if os.getenv('FORCE_ONLY_CPU', '0') == '1':
    print('FORCE_ONLY_CPU: Only compiling CPU extensions')
    DO_CPU = True
    DO_CUDA = False
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
extensions_cpu = [
    CppExtension('ml4reco_modules.extensions.select_knn_cpu', ['ml4reco_modules/extensions/select_knn_cpu.cpp'], **cpu_kwargs),
    CppExtension('ml4reco_modules.extensions.index_replacer_cpu', ['ml4reco_modules/extensions/index_replacer_cpu.cpp'], **cpu_kwargs),
    CppExtension('ml4reco_modules.extensions.bin_by_coordinates_cpu', ['ml4reco_modules/extensions/bin_by_coordinates_cpu.cpp'], **cpu_kwargs),
    CppExtension('ml4reco_modules.extensions.binned_select_knn_cpu', ['ml4reco_modules/extensions/binned_select_knn_cpu.cpp'], **cpu_kwargs) ,
    CppExtension('ml4reco_modules.extensions.binned_select_knn_grad_cpu', ['ml4reco_modules/extensions/binned_select_knn_grad_cpu.cpp'], **cpu_kwargs),
    CppExtension('ml4reco_modules.extensions.oc_helper_cpu', ['ml4reco_modules/extensions/oc_helper_cpu.cpp'], **cpu_kwargs) ,
    CppExtension('ml4reco_modules.extensions.select_with_default_cpu', ['ml4reco_modules/extensions/select_with_default_cpu.cpp'], **cpu_kwargs) 
]

cuda_kwargs = dict(
    include_dirs=[extensions_dir],
    extra_compile_args={'cxx': ['-O2'], 'nvcc': ['--expt-relaxed-constexpr', '-O2']},
    extra_link_args=['-s']
    )
extensions_cuda = [
    CUDAExtension(
        'ml4reco_modules.extensions.select_knn_cuda',
        ['ml4reco_modules/extensions/select_knn_cuda.cpp', 'ml4reco_modules/extensions/select_knn_cuda_kernel.cu'],
        **cuda_kwargs
        ),
    CUDAExtension(
        'ml4reco_modules.extensions.index_replacer_cuda',
        ['ml4reco_modules/extensions/index_replacer_cuda.cpp','ml4reco_modules/extensions/index_replacer_cuda_kernel.cu'],
        **cuda_kwargs
        ),
    CUDAExtension(
        'ml4reco_modules.extensions.bin_by_coordinates_cuda',
        ['ml4reco_modules/extensions/bin_by_coordinates_cuda.cpp', 'ml4reco_modules/extensions/bin_by_coordinates_cuda_kernel.cu'],
        **cuda_kwargs
        ),
    CUDAExtension(
        'ml4reco_modules.extensions.binned_select_knn_cuda',
        ['ml4reco_modules/extensions/binned_select_knn_cuda.cpp', 'ml4reco_modules/extensions/binned_select_knn_cuda_kernel.cu'],
        **cuda_kwargs
        ),
    CUDAExtension(
        'ml4reco_modules.extensions.binned_select_knn_grad_cuda',
        ['ml4reco_modules/extensions/binned_select_knn_grad_cuda.cpp', 'ml4reco_modules/extensions/binned_select_knn_grad_cuda_kernel.cu'],
        **cuda_kwargs
        ),
    CUDAExtension(
        'ml4reco_modules.extensions.oc_helper_cuda',
        ['ml4reco_modules/extensions/oc_helper_cuda.cpp', 'ml4reco_modules/extensions/oc_helper_cuda_kernel.cu'],
        **cuda_kwargs
        ),
    CUDAExtension(
        'ml4reco_modules.extensions.select_with_default_cuda',
        ['ml4reco_modules/extensions/select_with_default_cuda.cpp', 'ml4reco_modules/extensions/select_with_default_cuda_kernel.cu'],
        **cuda_kwargs
        )
    ]

extensions = []
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
num_jobs = os.getenv('NUM_PARALLEL_JOBS', '0')
make_args = []
if num_jobs != '0':
    make_args = [f'-j{num_jobs}']

setup(
    name='ml4reco_modules',
    ext_modules=extensions if not BUILD_DOCS else [],
    packages=find_packages(),  # Automatically find packages
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False, parallel=True, make_args=make_args)
    },
)


