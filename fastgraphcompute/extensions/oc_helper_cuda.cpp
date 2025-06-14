#include <torch/extension.h>
#include <vector>


//forward declaration
std::tuple<torch::Tensor, torch::Tensor> oc_helper_cuda_fn(
    torch::Tensor asso_idx,
    torch::Tensor unique_idx,
    torch::Tensor unique_rs_asso,
    torch::Tensor rs,
    torch::Tensor max_n_unique_over_splits,
    torch::Tensor max_n_in_splits,
    bool calc_m_not);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> oc_helper_cuda_interface(
    torch::Tensor asso_idx,
    torch::Tensor unique_idx,
    torch::Tensor unique_rs_asso,
    torch::Tensor rs,
    torch::Tensor max_n_unique_over_splits,
    torch::Tensor max_n_in_splits,
    bool calc_m_not) {
    CHECK_INPUT(asso_idx);
    CHECK_INPUT(unique_idx);
    CHECK_INPUT(unique_rs_asso);
    CHECK_INPUT(rs);
    CHECK_INPUT(max_n_unique_over_splits);
    CHECK_INPUT(max_n_in_splits);

    return oc_helper_cuda_fn(asso_idx, unique_idx, unique_rs_asso, rs, max_n_unique_over_splits, max_n_in_splits, calc_m_not);

    }


TORCH_LIBRARY(oc_helper_cuda, m) {
    m.def("oc_helper_cuda(Tensor asso_idx, Tensor unique_idx, Tensor unique_rs_asso, Tensor rs, Tensor max_n_unique_over_splits, Tensor max_n_in_splits, bool calc_m_not) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(oc_helper_cuda, CUDA, m) {
    m.impl("oc_helper_cuda", &oc_helper_cuda_interface);
}