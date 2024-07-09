#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")

__global__ void index_replacer_kernel(
    const int32_t* to_be_replaced,
    const int32_t* replacements,
    int32_t* replaced,
    const int32_t n_to_be_replaced,
    const int32_t n_replacements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_to_be_replaced) {
        int ridx = to_be_replaced[idx];
        if (ridx < 0) {
            replaced[idx] = ridx;
        } else if (ridx >= n_replacements) {
            printf("IndexReplacerOpFunctor: index out of range\n");
        } else {
            replaced[idx] = replacements[ridx];
        }
    }
}

std::vector<torch::Tensor> index_replacer_cuda_fn(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements
) {
    CHECK_CUDA(to_be_replaced);
    CHECK_CUDA(replacements);
    AT_ASSERTM(to_be_replaced.dtype() == torch::kInt32, "Input tensor must be int32");
    AT_ASSERTM(replacements.dtype() == torch::kInt32, "Replacement tensor must be int32");
    AT_ASSERTM(to_be_replaced.numel() == replacements.numel(), "Input and replacement tensors must have the same number of elements");

    auto replaced = torch::empty_like(to_be_replaced);

    const int32_t num_elements = to_be_replaced.numel();
    const int32_t threads_per_block = 256;
    const int32_t num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    index_replacer_kernel<<<num_blocks, threads_per_block>>>(
        to_be_replaced.data_ptr<int32_t>(),
        replacements.data_ptr<int32_t>(),
        replaced.data_ptr<int32_t>(),
        num_elements,
        replacements.numel()
    );

    cudaDeviceSynchronize();

    return { replaced };
}
