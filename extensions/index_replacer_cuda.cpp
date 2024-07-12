#include <torch/extension.h>

// CUDA forward declarations
torch::Tensor index_replacer_cuda_fn(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements
);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor index_replacer_cuda_interface(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements
) {
    CHECK_INPUT(to_be_replaced);
    CHECK_INPUT(replacements);
    return index_replacer_cuda_fn(to_be_replaced, replacements);
}

TORCH_LIBRARY(index_replacer_cuda, m) {
    m.def("index_replacer_cuda", index_replacer_cuda_interface);
}