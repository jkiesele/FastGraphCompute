#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void calc(
        const int64_t * to_be_replaced,
        const int64_t * replacements,
        int64_t * replaced,

        const int64_t n_to_be_replaced,
        const int64_t n_replacements){

    int64_t i =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_to_be_replaced)
        return;

    const int64_t ridx = to_be_replaced[i];
    if(ridx<0){
        replaced[i] = ridx;
        return;
    }
    if(ridx>=n_replacements){
        printf("IndexReplacerOpFunctor: index out of range\n");
        return;
    }
    replaced[i] = replacements[ridx];
}

void index_replacer(
        torch::Tensor to_be_replaced,
        torch::Tensor replacements,
        torch::Tensor replaced){

    const int64_t n_to_be_replaced = to_be_replaced.size(0);
    const int64_t n_replacements = replacements.size(0);

    const int threads = 1024;
    const dim3 blocks((n_to_be_replaced + threads - 1) / threads);

    calc<<<blocks, threads>>>(
        to_be_replaced.data_ptr<int64_t>(),
        replacements.data_ptr<int64_t>(),
        replaced.data_ptr<int64_t>(),
        n_to_be_replaced,
        n_replacements);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("index_replacer", &index_replacer, "Index Replacer (CUDA)");
}