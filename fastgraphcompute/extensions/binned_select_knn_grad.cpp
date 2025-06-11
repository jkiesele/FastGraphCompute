#include <torch/extension.h>

// Forward declarations for the actual implementations
// (implemented in binned_select_knn_grad_cpu.cpp and binned_select_knn_grad_cuda_kernel.cu)
torch::Tensor binned_select_knn_grad_cpu(
    torch::Tensor grad_distances,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor coordinates
);

torch::Tensor binned_select_knn_grad_cuda(
    torch::Tensor grad_distances,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor coordinates
);

// Common input checking macros
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CPU_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

torch::Tensor binned_select_knn_grad_cuda_interface(
    torch::Tensor grad_distances,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor coordinates
) {
    CHECK_INPUT(grad_distances);
    CHECK_INPUT(indices);
    CHECK_INPUT(distances);
    CHECK_INPUT(coordinates);
    return binned_select_knn_grad_cuda(grad_distances, indices, distances, coordinates);
}

torch::Tensor binned_select_knn_grad_cpu_interface(
    torch::Tensor grad_distances,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor coordinates
) {
    CHECK_CPU_INPUT(grad_distances);
    CHECK_CPU_INPUT(indices);
    CHECK_CPU_INPUT(distances);
    CHECK_CPU_INPUT(coordinates);
    return binned_select_knn_grad_cpu(grad_distances, indices, distances, coordinates);
}

TORCH_LIBRARY(binned_select_knn_grad, m) {
    m.def("binned_select_knn_grad(Tensor grad_distances, Tensor indices, Tensor distances, Tensor coordinates) -> Tensor");
}

TORCH_LIBRARY_IMPL(binned_select_knn_grad, CUDA, m) {
    m.impl("binned_select_knn_grad", &binned_select_knn_grad_cuda_interface);
}

TORCH_LIBRARY_IMPL(binned_select_knn_grad, CPU, m) {
    m.impl("binned_select_knn_grad", &binned_select_knn_grad_cpu_interface);
} 