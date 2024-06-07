#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")

template <typename scalar_t>
__global__ void calc(
    scalar_t* to_be_replaced,
    scalar_t* replacements,
    scalar_t* replaced,
    int64_t n_to_be_replaced,
    int64_t n_replacements) {

    int i =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_to_be_replaced)
        return;

    const int ridx = to_be_replaced[i];
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

// template<typename T>
// void 
std::tuple<torch::Tensor, torch::Tensor> index_replacer_cuda_fn(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements,
    torch::Tensor replaced,
    int64_t n_to_be_replaced,
    int64_t n_replacements
    ){

    CHECK_CUDA(to_be_replaced);
    CHECK_CUDA(replacements);
    CHECK_CUDA(replaced);

    const int threads = 1024;
    const dim3 blocks((n_to_be_replaced + threads - 1) / threads);

    AT_DISPATCH_INTEGRAL_TYPES(to_be_replaced.scalar_type(), "calc", ([&] {
        calc <scalar_t> <<<blocks, threads>>> (
            to_be_replaced.data_ptr<scalar_t>(),
            replacements.data_ptr<scalar_t>(),
            replaced.data_ptr<scalar_t>(),
            n_to_be_replaced,
            n_replacements);
    }));

    cudaDeviceSynchronize();

    return std::make_tuple(to_be_replaced, replacements);
}


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("index_replacer", &index_replacer_wrapper, "Index Replacer (CUDA)");
// }