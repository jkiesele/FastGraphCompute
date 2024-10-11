#include <torch/extension.h>

// CUDA forward declarations
torch::Tensor binned_select_knn_grad_cuda_fn(
    torch::Tensor grad_distances,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor coordinates
);


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor binned_select_knn_grad_cuda_interface(
    torch::Tensor grad_distances,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor coordinates
)
{
    CHECK_INPUT(grad_distances);
    CHECK_INPUT(indices);
    CHECK_INPUT(distances);
    CHECK_INPUT(coordinates);
    return binned_select_knn_grad_cuda_fn(grad_distances, indices, distances, coordinates);
}


TORCH_LIBRARY(binned_select_knn_grad_cuda, m) {
    m.def("binned_select_knn_grad_cuda", &binned_select_knn_grad_cuda_interface);
}