// fastgraphcompute_ops.h
#pragma once
#include <torch/extension.h>

// Declare the operators that will be available after registration
namespace torch {
namespace ops {
namespace bin_by_coordinates {
    extern std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    bin_by_coordinates(
        torch::Tensor coordinates,
        torch::Tensor row_splits,
        c10::optional<torch::Tensor> bin_width,
        c10::optional<torch::Tensor> n_bins,
        bool calc_n_per_bin,
        bool pre_normalized);
} // namespace bin_by_coordinates

namespace binned_select_knn {
    extern std::tuple<torch::Tensor, torch::Tensor> binned_select_knn(
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
} // namespace binned_select_knn

namespace index_replacer {
    extern torch::Tensor index_replacer(
        torch::Tensor to_be_replaced,
        torch::Tensor replacements);
} // namespace index_replacer

namespace binned_select_knn_grad {
    extern torch::Tensor binned_select_knn_grad(
        torch::Tensor grad_distances,
        torch::Tensor indices,
        torch::Tensor distances,
        torch::Tensor coordinates);
} // namespace binned_select_knn_grad
} // namespace ops
} // namespace torch
