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

// User-facing C++ function
torch::Tensor binned_select_knn_grad_entry_point(
    torch::Tensor grad_distances,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor coordinates
) {
    // ensure inputs are contiguous
    grad_distances = grad_distances.contiguous();
    indices = indices.contiguous();
    distances = distances.contiguous();
    coordinates = coordinates.contiguous();

    // Common type checks
    if (indices.scalar_type() != torch::kInt32 && indices.scalar_type() != torch::kInt64) {
        throw std::invalid_argument("Unsupported tensor type for indices. Must be Int32 or Int64.");
    }
    if (grad_distances.scalar_type() != torch::kFloat32) {
        throw std::invalid_argument("Unsupported tensor type for grad_distances. Must be Float32.");
    }
    if (distances.scalar_type() != torch::kFloat32) {
        throw std::invalid_argument("Unsupported tensor type for distances. Must be Float32.");
    }
    if (coordinates.scalar_type() != torch::kFloat32) {
        throw std::invalid_argument("Unsupported tensor type for coordinates. Must be Float32.");
    }

    // Call the internal dispatchable operator
    return torch::ops::binned_select_knn_grad_op::binned_select_knn_grad_op_kernel(
        grad_distances, indices, distances, coordinates);
}

// Register the internal operator (for backend dispatch)
TORCH_LIBRARY(binned_select_knn_grad_op, m) {
    m.def("binned_select_knn_grad_op_kernel(Tensor grad_distances, Tensor indices, Tensor distances, Tensor coordinates) -> Tensor");
}

TORCH_LIBRARY_IMPL(binned_select_knn_grad_op, CPU, m) {
    m.impl("binned_select_knn_grad_op_kernel", &binned_select_knn_grad_cpu);
}

TORCH_LIBRARY_IMPL(binned_select_knn_grad_op, CUDA, m) {
    m.impl("binned_select_knn_grad_op_kernel", &binned_select_knn_grad_cuda);
}

// Register the user-facing operator
TORCH_LIBRARY(binned_select_knn_grad_lib, m) {
    m.def("binned_select_knn_grad(Tensor grad_distances, Tensor indices, Tensor distances, Tensor coordinates) -> Tensor");
}

TORCH_LIBRARY_IMPL(binned_select_knn_grad_lib, CPU, m) {
    m.impl("binned_select_knn_grad", &binned_select_knn_grad_entry_point);
}

TORCH_LIBRARY_IMPL(binned_select_knn_grad_lib, CUDA, m) {
    m.impl("binned_select_knn_grad", &binned_select_knn_grad_entry_point);
} 