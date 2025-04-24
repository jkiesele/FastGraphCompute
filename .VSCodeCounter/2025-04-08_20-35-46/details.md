# Details

Date : 2025-04-08 20:35:46

Directory /Users/aarushagarwal/Documents/CMU/Research/fastgraphcompute

Total : 48 files,  4553 codes, 726 comments, 1294 blanks, all 6573 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [README.md](/README.md) | Markdown | 89 | 0 | 30 | 119 |
| [fastgraphcompute/__init__.py](/fastgraphcompute/__init__.py) | Python | 31 | 5 | 8 | 44 |
| [fastgraphcompute/extensions/__init__.py](/fastgraphcompute/extensions/__init__.py) | Python | 1 | 0 | 0 | 1 |
| [fastgraphcompute/extensions/bin_by_coordinates.py](/fastgraphcompute/extensions/bin_by_coordinates.py) | Python | 69 | 20 | 22 | 111 |
| [fastgraphcompute/extensions/bin_by_coordinates_cpu.cpp](/fastgraphcompute/extensions/bin_by_coordinates_cpu.cpp) | C++ | 114 | 9 | 36 | 159 |
| [fastgraphcompute/extensions/bin_by_coordinates_cuda.cpp](/fastgraphcompute/extensions/bin_by_coordinates_cuda.cpp) | C++ | 27 | 2 | 5 | 34 |
| [fastgraphcompute/extensions/bin_by_coordinates_cuda_kernel.cu](/fastgraphcompute/extensions/bin_by_coordinates_cuda_kernel.cu) | CUDA C++ | 125 | 16 | 35 | 176 |
| [fastgraphcompute/extensions/binned_select_knn.py](/fastgraphcompute/extensions/binned_select_knn.py) | Python | 114 | 31 | 37 | 182 |
| [fastgraphcompute/extensions/binned_select_knn_cpu.cpp](/fastgraphcompute/extensions/binned_select_knn_cpu.cpp) | C++ | 226 | 14 | 52 | 292 |
| [fastgraphcompute/extensions/binned_select_knn_cuda.cpp](/fastgraphcompute/extensions/binned_select_knn_cuda.cpp) | C++ | 52 | 3 | 6 | 61 |
| [fastgraphcompute/extensions/binned_select_knn_cuda_kernel.cu](/fastgraphcompute/extensions/binned_select_knn_cuda_kernel.cu) | CUDA C++ | 272 | 11 | 73 | 356 |
| [fastgraphcompute/extensions/binned_select_knn_grad_cpu.cpp](/fastgraphcompute/extensions/binned_select_knn_grad_cpu.cpp) | C++ | 141 | 3 | 35 | 179 |
| [fastgraphcompute/extensions/binned_select_knn_grad_cuda.cpp](/fastgraphcompute/extensions/binned_select_knn_grad_cuda.cpp) | C++ | 26 | 2 | 6 | 34 |
| [fastgraphcompute/extensions/binned_select_knn_grad_cuda_kernel.cu](/fastgraphcompute/extensions/binned_select_knn_grad_cuda_kernel.cu) | CUDA C++ | 141 | 0 | 38 | 179 |
| [fastgraphcompute/extensions/binstepper.h](/fastgraphcompute/extensions/binstepper.h) | C++ | 136 | 10 | 25 | 171 |
| [fastgraphcompute/extensions/cuda_helpers.h](/fastgraphcompute/extensions/cuda_helpers.h) | C++ | 156 | 92 | 35 | 283 |
| [fastgraphcompute/extensions/helpers.h](/fastgraphcompute/extensions/helpers.h) | C++ | 9 | 6 | 6 | 21 |
| [fastgraphcompute/extensions/index_replacer.py](/fastgraphcompute/extensions/index_replacer.py) | Python | 41 | 6 | 12 | 59 |
| [fastgraphcompute/extensions/index_replacer_cpu.cpp](/fastgraphcompute/extensions/index_replacer_cpu.cpp) | C++ | 45 | 0 | 8 | 53 |
| [fastgraphcompute/extensions/index_replacer_cuda.cpp](/fastgraphcompute/extensions/index_replacer_cuda.cpp) | C++ | 19 | 2 | 4 | 25 |
| [fastgraphcompute/extensions/index_replacer_cuda_kernel.cu](/fastgraphcompute/extensions/index_replacer_cuda_kernel.cu) | CUDA C++ | 45 | 0 | 9 | 54 |
| [fastgraphcompute/extensions/oc_helper.py](/fastgraphcompute/extensions/oc_helper.py) | Python | 179 | 24 | 58 | 261 |
| [fastgraphcompute/extensions/oc_helper_cpu.cpp](/fastgraphcompute/extensions/oc_helper_cpu.cpp) | C++ | 118 | 10 | 33 | 161 |
| [fastgraphcompute/extensions/oc_helper_cuda.cpp](/fastgraphcompute/extensions/oc_helper_cuda.cpp) | C++ | 32 | 2 | 8 | 42 |
| [fastgraphcompute/extensions/oc_helper_cuda_kernel.cu](/fastgraphcompute/extensions/oc_helper_cuda_kernel.cu) | CUDA C++ | 152 | 16 | 28 | 196 |
| [fastgraphcompute/extensions/oc_helper_helper.cpp](/fastgraphcompute/extensions/oc_helper_helper.cpp) | C++ | 58 | 19 | 24 | 101 |
| [fastgraphcompute/extensions/select_knn_cpu.cpp](/fastgraphcompute/extensions/select_knn_cpu.cpp) | C++ | 170 | 11 | 32 | 213 |
| [fastgraphcompute/extensions/select_knn_cuda.cpp](/fastgraphcompute/extensions/select_knn_cuda.cpp) | C++ | 31 | 2 | 4 | 37 |
| [fastgraphcompute/extensions/select_knn_cuda_kernel.cu](/fastgraphcompute/extensions/select_knn_cuda_kernel.cu) | CUDA C++ | 156 | 13 | 36 | 205 |
| [fastgraphcompute/gnn_ops.py](/fastgraphcompute/gnn_ops.py) | Python | 69 | 20 | 22 | 111 |
| [fastgraphcompute/object_condensation.py](/fastgraphcompute/object_condensation.py) | Python | 233 | 29 | 76 | 338 |
| [fastgraphcompute/torch_geometric_interface.py](/fastgraphcompute/torch_geometric_interface.py) | Python | 48 | 7 | 16 | 71 |
| [performance/binned_knn.py](/performance/binned_knn.py) | Python | 92 | 16 | 40 | 148 |
| [performance/gravnet_perfm.py](/performance/gravnet_perfm.py) | Python | 120 | 27 | 45 | 192 |
| [performance/oc_helper_helper.py](/performance/oc_helper_helper.py) | Python | 93 | 13 | 35 | 141 |
| [performance/oc_helpers.py](/performance/oc_helpers.py) | Python | 67 | 14 | 20 | 101 |
| [pyproject.toml](/pyproject.toml) | toml | 6 | 0 | 1 | 7 |
| [setup.cfg](/setup.cfg) | Properties | 40 | 0 | 8 | 48 |
| [setup.py](/setup.py) | Python | 114 | 6 | 21 | 141 |
| [tests/test_all.py](/tests/test_all.py) | Python | 11 | 0 | 2 | 13 |
| [tests/test_bin_by_coordinates.py](/tests/test_bin_by_coordinates.py) | Python | 158 | 18 | 53 | 229 |
| [tests/test_binned_select_knn.py](/tests/test_binned_select_knn.py) | Python | 114 | 29 | 53 | 196 |
| [tests/test_gnn_ops.py](/tests/test_gnn_ops.py) | Python | 83 | 7 | 26 | 116 |
| [tests/test_index_replacer.py](/tests/test_index_replacer.py) | Python | 50 | 12 | 20 | 82 |
| [tests/test_object_condensation.py](/tests/test_object_condensation.py) | Python | 218 | 123 | 58 | 399 |
| [tests/test_oc_helper.py](/tests/test_oc_helper.py) | Python | 133 | 62 | 49 | 244 |
| [tests/test_oc_helper_helper.py](/tests/test_oc_helper_helper.py) | Python | 82 | 14 | 30 | 126 |
| [tests/test_torch_geometric_interface.py](/tests/test_torch_geometric_interface.py) | Python | 47 | 0 | 14 | 61 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)