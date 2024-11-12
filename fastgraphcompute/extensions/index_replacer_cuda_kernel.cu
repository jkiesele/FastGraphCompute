#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

__global__ void index_replacer_kernel(
    const int32_t* to_be_replaced,
    const int32_t* replacements,
    int32_t* replaced,
    const int32_t n_to_be_replaced,
    const int32_t n_replacements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_to_be_replaced) return;

    const int ridx = to_be_replaced[i];
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
    TORCH_CHECK(to_be_replaced.dtype() == torch::kInt32, "Input tensor must be int32");
    TORCH_CHECK(replacements.dtype() == torch::kInt32, "Replacement tensor must be int32");

    auto replaced = torch::empty_like(to_be_replaced);

    const int32_t n_to_be_replaced = to_be_replaced.numel();
    const int32_t threads_per_block = 1024;
    const int32_t num_blocks = (n_to_be_replaced + threads_per_block - 1) / threads_per_block;

    index_replacer_kernel<<<num_blocks, threads_per_block>>>(
        to_be_replaced.data_ptr<int32_t>(),
        replacements.data_ptr<int32_t>(),
        replaced.data_ptr<int32_t>(),
        n_to_be_replaced,
        replacements.numel()
    );

    cudaDeviceSynchronize();

    return replaced;
}