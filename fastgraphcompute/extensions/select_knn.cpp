#include <torch/extension.h>
#include <vector>

// Forward declarations
std::tuple<torch::Tensor, torch::Tensor> select_knn_cpu(
    torch::Tensor coords, 
    torch::Tensor row_splits,
    torch::Tensor mask, 
    int64_t n_neighbours, 
    double max_radius,
    int64_t mask_mode
);

std::tuple<torch::Tensor, torch::Tensor> select_knn_cuda_fn(
    torch::Tensor coords, 
    torch::Tensor row_splits,
    torch::Tensor mask, 
    int64_t n_neighbours, 
    double max_radius,
    int64_t mask_mode
);

// C++ interface
#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CPU Interface
std::tuple<torch::Tensor, torch::Tensor> select_knn_cpu_interface(
    torch::Tensor coords, 
    torch::Tensor row_splits,
    torch::Tensor mask, 
    int64_t n_neighbours, 
    double max_radius,
    int64_t mask_mode
) {
    CHECK_INPUT_CPU(coords);
    CHECK_INPUT_CPU(row_splits);
    CHECK_INPUT_CPU(mask);
    return select_knn_cpu_fn(coords, row_splits, mask, n_neighbours, max_radius, mask_mode);
}

// CUDA Interface
std::tuple<torch::Tensor, torch::Tensor> select_knn_cuda_interface(
    torch::Tensor coords, 
    torch::Tensor row_splits,
    torch::Tensor mask, 
    int64_t n_neighbours, 
    double max_radius,
    int64_t mask_mode
) {
    CHECK_INPUT_CUDA(coords);
    CHECK_INPUT_CUDA(row_splits);
    CHECK_INPUT_CUDA(mask);
    return select_knn_cuda_fn(coords, row_splits, mask, n_neighbours, max_radius, mask_mode);
}

TORCH_LIBRARY(select_knn, m) {
    m.def("select_knn(Tensor coords, Tensor row_splits, Tensor mask, int n_neighbours, float max_radius, int mask_mode) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(select_knn, CPU, m) {
    m.impl("select_knn", select_knn_cpu_interface);
}

TORCH_LIBRARY_IMPL(select_knn, CUDA, m) {
    m.impl("select_knn", select_knn_cuda_interface);
}
