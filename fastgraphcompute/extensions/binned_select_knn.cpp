#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::tuple<torch::Tensor, torch::Tensor> binned_select_knn_cuda_fn(
    torch::Tensor coordinates,
    torch::Tensor bin_idx,
    torch::Tensor dim_bin_idx,
    torch::Tensor bin_boundaries,
    torch::Tensor n_bins,
    torch::Tensor bin_width,
    torch::Tensor direction,
    bool tf_compat,
    bool use_direction,
    int64_t K);

// CPU forward declarations
std::tuple<torch::Tensor, torch::Tensor> binned_select_knn_cpu(
    torch::Tensor coordinates,
    torch::Tensor bin_idx,
    torch::Tensor dim_bin_idx,
    torch::Tensor bin_boundaries,
    torch::Tensor n_bins,
    torch::Tensor bin_width,
    torch::Tensor direction,
    bool tf_compat,
    bool use_direction,
    int64_t K);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CPU_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> binned_select_knn_cuda_interface(
    torch::Tensor coordinates,
    torch::Tensor bin_idx,
    torch::Tensor dim_bin_idx,
    torch::Tensor bin_boundaries,
    torch::Tensor n_bins,
    torch::Tensor bin_width,
    torch::Tensor direction,
    bool tf_compat,
    bool use_direction,
    int64_t K)
{
    CHECK_INPUT(coordinates);
    CHECK_INPUT(bin_idx);
    CHECK_INPUT(dim_bin_idx);
    CHECK_INPUT(bin_boundaries);
    CHECK_INPUT(n_bins);
    CHECK_INPUT(bin_width);
    CHECK_INPUT(direction);
    return binned_select_knn_cuda_fn(coordinates, bin_idx, dim_bin_idx, bin_boundaries, n_bins, bin_width, direction, tf_compat, use_direction, K);
}

std::tuple<torch::Tensor, torch::Tensor> binned_select_knn_cpu_interface(
    torch::Tensor coordinates,
    torch::Tensor bin_idx,
    torch::Tensor dim_bin_idx,
    torch::Tensor bin_boundaries,
    torch::Tensor n_bins,
    torch::Tensor bin_width,
    torch::Tensor direction,
    bool tf_compat,
    bool use_direction,
    int64_t K)
{
    CHECK_CPU_INPUT(coordinates);
    CHECK_CPU_INPUT(bin_idx);
    CHECK_CPU_INPUT(dim_bin_idx);
    CHECK_CPU_INPUT(bin_boundaries);
    CHECK_CPU_INPUT(n_bins);
    CHECK_CPU_INPUT(bin_width);
    CHECK_CPU_INPUT(direction);
    return binned_select_knn_cpu(coordinates, bin_idx, dim_bin_idx, bin_boundaries, n_bins, bin_width, direction, tf_compat, use_direction, K);
}

TORCH_LIBRARY(binned_select_knn, m) {
    m.def("binned_select_knn(Tensor coordinates, Tensor bin_idx, Tensor dim_bin_idx, Tensor bin_boundaries, Tensor n_bins, Tensor bin_width, Tensor direction, bool tf_compat, bool use_direction, int K) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(binned_select_knn, CUDA, m) {
    m.impl("binned_select_knn", binned_select_knn_cuda_interface);
}

TORCH_LIBRARY_IMPL(binned_select_knn, CPU, m) {
    m.impl("binned_select_knn", binned_select_knn_cpu_interface);
} 