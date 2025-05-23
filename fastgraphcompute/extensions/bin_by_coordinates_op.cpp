#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bin_by_coordinates_cuda_fn(
    torch::Tensor coords,
    torch::Tensor row_splits,
    torch::Tensor binswidth,
    torch::Tensor nbins,
    bool calc_n_per_bin);

// CPU forward declarations
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bin_by_coordinates_cpu(
    torch::Tensor coords,
    torch::Tensor row_splits,
    torch::Tensor binswidth,
    torch::Tensor nbins,
    bool calc_n_per_bin);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CPU_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bin_by_coordinates_cuda_interface(
    torch::Tensor coords,
    torch::Tensor row_splits,
    torch::Tensor binswidth,
    torch::Tensor nbins,
    bool calc_n_per_bin)
{
    CHECK_INPUT(coords);
    CHECK_INPUT(row_splits);
    CHECK_INPUT(binswidth);
    CHECK_INPUT(nbins);
    return bin_by_coordinates_cuda_fn(coords, row_splits, binswidth, nbins, calc_n_per_bin);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bin_by_coordinates_cpu_interface(
    torch::Tensor coords,
    torch::Tensor row_splits,
    torch::Tensor binswidth,
    torch::Tensor nbins,
    bool calc_n_per_bin)
{
    CHECK_CPU_INPUT(coords);
    CHECK_CPU_INPUT(row_splits);
    CHECK_CPU_INPUT(binswidth);
    CHECK_CPU_INPUT(nbins);
    return bin_by_coordinates_cpu(coords, row_splits, binswidth, nbins, calc_n_per_bin);
}

TORCH_LIBRARY(bin_by_coordinates, m) {
    m.def("bin_by_coordinates(Tensor coords, Tensor row_splits, Tensor binswidth, Tensor nbins, bool calc_n_per_bin) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(bin_by_coordinates, CUDA, m) {
    m.impl("bin_by_coordinates", bin_by_coordinates_cuda_interface);
}

TORCH_LIBRARY_IMPL(bin_by_coordinates, CPU, m) {
    m.impl("bin_by_coordinates", bin_by_coordinates_cpu_interface);
}