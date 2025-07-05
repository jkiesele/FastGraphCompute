#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

__global__ void index_replacer_kernel(
    const int64_t* to_be_replaced,
    const int64_t* replacements,
    int64_t* replaced,
    const int64_t n_to_be_replaced,
    const int64_t n_replacements
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_to_be_replaced) return;

    const int64_t ridx = to_be_replaced[i];
    if(ridx<0){
        replaced[i] = ridx;
    }
    else if(ridx>=n_replacements){
        replaced[i] = replacements[i];//out of range
    }
    else
        replaced[i] = replacements[ridx];
}

torch::Tensor index_replacer_cuda_fn(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements
) {
    CHECK_CUDA(to_be_replaced);
    CHECK_CUDA(replacements);
    TORCH_CHECK(to_be_replaced.dtype() == torch::kInt64, "Input tensor must be int64");
    TORCH_CHECK(replacements.dtype() == torch::kInt64, "Replacement tensor must be int64");

    auto replaced = torch::empty_like(to_be_replaced);

    const int64_t n_to_be_replaced = to_be_replaced.numel();
    const int64_t threads_per_block = 1024;
    const int64_t num_blocks = (n_to_be_replaced + threads_per_block - 1) / threads_per_block;

    index_replacer_kernel<<<num_blocks, threads_per_block>>>(
        to_be_replaced.data_ptr<int64_t>(),
        replacements.data_ptr<int64_t>(),
        replaced.data_ptr<int64_t>(),
        n_to_be_replaced,
        replacements.numel()
    );

    cudaDeviceSynchronize();

    return replaced;
}