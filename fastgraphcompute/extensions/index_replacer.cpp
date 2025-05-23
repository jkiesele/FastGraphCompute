#include <torch/extension.h>
#include <vector>

// Forward declaration for CUDA function (from .cu file)
torch::Tensor index_replacer_cuda_fn(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements
);

// Forward declaration for CPU function (definition in index_replacer_cpu.cpp)
torch::Tensor index_replacer_cpu_fn(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements
);

// C++ interface
#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CPU Interface
torch::Tensor index_replacer_cpu_interface(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements
) {
    CHECK_INPUT_CPU(to_be_replaced);
    CHECK_INPUT_CPU(replacements);
    return index_replacer_cpu_fn(to_be_replaced, replacements);
}

// CUDA Interface
torch::Tensor index_replacer_cuda_interface(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements
) {
    CHECK_INPUT_CUDA(to_be_replaced);
    CHECK_INPUT_CUDA(replacements);
    return index_replacer_cuda_fn(to_be_replaced, replacements);
}

TORCH_LIBRARY(index_replacer, m) {
    m.def("index_replacer(Tensor to_be_replaced, Tensor replacements) -> Tensor");
}

TORCH_LIBRARY_IMPL(index_replacer, CPU, m) {
    m.impl("index_replacer", index_replacer_cpu_interface);
}

TORCH_LIBRARY_IMPL(index_replacer, CUDA, m) {
    m.impl("index_replacer", index_replacer_cuda_interface);
}
