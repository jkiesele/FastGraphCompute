#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_helpers.h>

#include <torch/script.h>
#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <algorithm>

#define C10_CUDA_KERNEL_LAUNCH_CHECK() {                         \
    cudaError_t err = cudaGetLastError();                        \
    if (err != cudaSuccess) {                                    \
        printf("CUDA Kernel launch error: %s\n",                 \
               cudaGetErrorString(err));                         \
    }                                                            \

// Forward declaration for the main op that will be exposed
std::tuple<torch::Tensor, torch::Tensor> binned_select_knn_cpp_op(
    torch::Tensor coords,
    torch::Tensor row_splits,
    int64_t K,
    c10::optional<torch::Tensor> direction_opt,
    c10::optional<torch::Tensor> n_bins_user,
    int64_t max_bin_dims_user,
    bool torch_compatible_indices);

struct BinnedKNNAutograd : public torch::autograd::Function<BinnedKNNAutograd> {
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor coords,
        torch::Tensor row_splits,
        int64_t K,
        c10::optional<torch::Tensor> direction,
        c10::optional<torch::Tensor> n_bins_user,
        int64_t max_bin_dims_user,
        bool torch_compatible_indices) {

        // Validate input coordinates and max_bin_dims
        TORCH_CHECK(coords.size(1) > 0, "Input coordinates must have at least one dimension. Got 0 dimensions.");
        TORCH_CHECK(max_bin_dims_user > 0, "max_bin_dims must be greater than 0. Got 0.");

        auto original_device = coords.device();
        auto int32_options = torch::TensorOptions().dtype(torch::kInt32).device(original_device);
        auto float_options = torch::TensorOptions().dtype(coords.dtype()).device(original_device);

        // Estimate a good number of bins for homogeneous distributions
        torch::Tensor elems_per_rs_float;
        if (row_splits.size(0) > 0) { // Avoid division by zero if row_splits is empty
             elems_per_rs_float = torch::max(row_splits).to(torch::kFloat32) / static_cast<float>(row_splits.size(0));
        } else {
             elems_per_rs_float = torch::tensor(0.0f, float_options);
        }
        torch::Tensor elems_per_rs = (elems_per_rs_float.to(torch::kInt32) + 1);
        
        int64_t max_bin_dims = std::min(max_bin_dims_user, coords.size(1));

        // Calculate n_bins for binning if not provided by user
        torch::Tensor n_bins;
        if (n_bins_user.has_value()) {
            n_bins = n_bins_user.value();
        } else {
            n_bins = torch::pow(elems_per_rs.to(torch::kFloat32) / (static_cast<float>(K) / 32.0f), 1.0f / static_cast<float>(max_bin_dims));
            n_bins = n_bins.to(torch::kInt32);
            n_bins = torch::where(n_bins < 5, torch::tensor(5, int32_options), n_bins);
            n_bins = torch::where(n_bins > 30, torch::tensor(30, int32_options), n_bins);
        }

        // Handle binning for the coordinates
        torch::Tensor bin_coords = coords;
        if (bin_coords.size(bin_coords.dim() - 1) > max_bin_dims) {
            bin_coords = bin_coords.slice(/*dim=*/1, /*start=*/0, /*end=*/max_bin_dims);
        }

        // - bin_coords: Input coordinates to be binned
        // - row_splits: Tensor defining the boundaries of each row
        // - n_bins: Number of bins to use for each dimension
        // - calc_n_per_bin: Set to true to calculate number of points per bin
        // - pre_normalized: Set to false as coordinates are not pre-normalized

        std::tie(dbinning, binning, nb, bin_width, nper) = torch::ops::bin_by_coordinates_cuda::bin_by_coordinates::call(
            bin_coords, row_splits, n_bins, true, false);

        // dbinning: Multi-dimensional bin indices for each point (shape: [n_points, n_dims])
        // binning: Flattened bin indices for each point (shape: [n_points])
        // nb: Final number of bins used in each dimension (shape: [n_dims])
        // bin_width: Width of each bin in each dimension (shape: [n_dims])
        // nper: Number of points in each bin (shape: [total_bins])

        // Sort points by bin assignment
        torch::Tensor sorting_indices;
        if (binning.numel() > 0) {
            sorting_indices = torch::argsort(binning, /*dim=*/0).to(torch::kInt32);
        } else { // empty input case
            sorting_indices = torch::empty({0}, binning.options().dtype(torch::kInt32));
        }

        // Gather sorted tensors
        torch::Tensor scoords = coords.index_select(0, sorting_indices);
        torch::Tensor sbinning = binning.index_select(0, sorting_indices); // Sorted flat_bin_indices
        torch::Tensor sdbinning = dbinning.index_select(0, sorting_indices);    // Sorted bin_indices (per dim)

        c10::optional<torch::Tensor> sdirection_opt;
        if (direction_opt.has_value()) {
            sdirection_opt = direction_opt.value().index_select(0, sorting_indices);
        }

        // Create bin boundaries
        torch::Tensor zero_tensor = torch::zeros({1}, int32_options.device(nper.device())); // Ensure device match
        torch::Tensor bin_boundaries_cat = torch::cat({zero_tensor, nper}, /*dim=*/0);
        torch::Tensor bin_boundaries = torch::cumsum(bin_boundaries_cat, /*dim=*/0, torch::kInt32);

        TORCH_CHECK(torch::max(bin_boundaries).item<int32_t>() == torch::max(row_splits).item<int32_t>(),
                    "Bin boundaries do not match row splits.");

        // call the _binned_select_knn kernel
        torch::Tensor idx_sorted, dist_sorted;
        torch::Tensor direction_input_for_knn;
        bool use_direction = false;

        if (direction_opt.has_value() && direction_opt.value().numel() > 0) {
            direction_input_for_knn = direction_opt.value();
            use_direction = true;
        } else {
            // Create an empty tensor with dtype of sdbinning (dim_bin_idx) and device of scoords
            direction_input_for_knn = torch::empty({0}, sdbinning.options().device(scoords.device()));
        }
        
        // Ensure kernel inputs are on the correct device
        auto kernel_device = scoords.device();
        torch::Tensor k_scoords = scoords.contiguous();
        torch::Tensor k_sbinning = sbinning.contiguous().to(kernel_device);
        torch::Tensor k_sdbinning = sdbinning.contiguous().to(kernel_device);
        torch::Tensor k_bin_boundaries = bin_boundaries.contiguous().to(kernel_device);
        torch::Tensor k_n_bins_for_knn = nb.contiguous().to(kernel_device);
        torch::Tensor k_bin_width_for_knn = bin_width.contiguous().to(kernel_device);
        torch::Tensor k_direction_input = direction_input_for_knn.contiguous().to(kernel_device);

        std::tie(idx_sorted, dist_sorted) =
            torch::ops::binned_select_knn::binned_select_knn::call(
                k_scoords, k_sbinning, k_sdbinning, k_bin_boundaries,
                k_n_bins_for_knn, k_bin_width_for_knn, k_direction_input,
                torch_compatible_indices, use_direction, K);

        // 10. Index Replacer
        // Assuming index_replacer is a registered C++ op
        // If not, its logic needs to be ported or made available as a C++ function.
        torch::Tensor idx_unsorted;
        if (idx_sorted.numel() > 0) {
             idx_unsorted = torch::ops::fastgraphcompute::index_replacer::call(idx_sorted, sorting_indices);
        } else {
             idx_unsorted = torch::empty_like(idx_sorted);
        }


        // 11. Scatter results back to original order
        torch::Tensor sorting_indices_long = sorting_indices.to(torch::kInt64);
        
        torch::Tensor dist_final = torch::empty_like(dist_sorted, dist_sorted.options().device(original_device));
        torch::Tensor idx_final = torch::empty_like(idx_unsorted, idx_unsorted.options().device(original_device));

        if (dist_sorted.numel() > 0) {
            dist_final.scatter_(/*dim=*/0, sorting_indices_long.unsqueeze(-1).expand_as(dist_sorted), dist_sorted);
        }
        if (idx_unsorted.numel() > 0) {
            idx_final.scatter_(/*dim=*/0, sorting_indices_long.unsqueeze(-1).expand_as(idx_unsorted), idx_unsorted);
        }
        
        // 12. Save for backward
        ctx->save_for_backward({idx_final, dist_final, coords});
        // Note: K, direction_opt etc. are not tensors, so not saved directly.
        // If backward needs them, they should be stored as attributes on ctx if possible,
        // or re-derived. Here, the grad op only needs idx, dist, coords.

        return {idx_final.to(original_device), dist_final.to(original_device)};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        
        auto saved_tensors = ctx->get_saved_variables();
        torch::Tensor idx = saved_tensors[0];
        torch::Tensor dist = saved_tensors[1];
        torch::Tensor coords = saved_tensors[2]; // Original coordinates

        // grad_outputs[0] is grad_idx (likely undefined/None if idx is long dtype and not requiring grad)
        // grad_outputs[1] is grad_dist
        torch::Tensor grad_dist_input = grad_outputs[1];
        torch::Tensor grad_coordinates;

        if (grad_dist_input.defined() && grad_dist_input.requires_grad()) { // Check if grad_dist itself requires grad
            // Ensure inputs to grad kernel are contiguous and on the correct device
            torch::Tensor k_grad_dist = grad_dist_input.contiguous();
            torch::Tensor k_idx = idx.contiguous().to(k_grad_dist.device());
            torch::Tensor k_dist = dist.contiguous().to(k_grad_dist.device());
            torch::Tensor k_coords = coords.contiguous().to(k_grad_dist.device());

            if (k_grad_dist.device().is_cuda()) {
                grad_coordinates = torch::ops::binned_select_knn_grad_cuda::binned_select_knn_grad_cuda::call(
                    k_grad_dist, k_idx, k_dist, k_coords);
            } else {
                grad_coordinates = torch::ops::binned_select_knn_grad_cpu::binned_select_knn_grad_cpu::call(
                    k_grad_dist, k_idx, k_dist, k_coords);
            }
        } else {
            grad_coordinates = torch::zeros_like(coords, coords.options().requires_grad(false));
        }
        
        // Gradients for: coords, row_splits, K, direction_opt, n_bins_opt, max_bin_dims, torch_compatible_indices
        // Only coords requires grad here. Others are either non-tensor or typically don't require grad.
        return {grad_coordinates,
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor()
               }; 
    }
};

// Wrapper function that calls the apply method of the BinnedKNNAutograd struct
// This is the function that will be registered and called from Python.
std::tuple<torch::Tensor, torch::Tensor> binned_select_knn_cpp_op(
    torch::Tensor coords,
    torch::Tensor row_splits,
    int64_t K,
    c10::optional<torch::Tensor> direction_opt,
    c10::optional<torch::Tensor> n_bins_user,
    int64_t max_bin_dims_user,
    bool torch_compatible_indices) {
    
    auto result_tuple = BinnedKNNAutograd::apply(coords, row_splits, K_val, direction_opt, n_bins_user, max_bin_dims_user, torch_compatible_indices);
    return std::make_tuple(result_tuple[0], result_tuple[1]);
}

// Operator Registration
TORCH_LIBRARY(fastgraphcompute_custom_ops, m) {
    m.def("binned_select_knn(Tensor coords, Tensor row_splits, int K, Tensor? direction, Tensor? n_bins, int max_bin_dims, bool torch_compatible_indices) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fastgraphcompute_custom_ops, Autograd, m) {
    m.impl("binned_select_knn", TORCH_FN(fastgraphcompute_ops::binned_select_knn_cpp_op));
}

// Note on other ops:
// Make sure the following ops (and their CPU/CUDA counterparts) are also registered via TORCH_LIBRARY
// so they can be called via torch::ops::...::call(...):
// - torch::ops::bin_by_coordinates_cuda::bin_by_coordinates
// - torch::ops::bin_by_coordinates_cpu::bin_by_coordinates_cpu
// - torch::ops::binned_select_knn_cuda::binned_select_knn_cuda
// - torch::ops::binned_select_knn_cpu::binned_select_knn_cpu
// - torch::ops::binned_select_knn_grad_cuda::binned_select_knn_grad_cuda
// - torch::ops::binned_select_knn_grad_cpu::binned_select_knn_grad_cpu
// - torch::ops::fastgraphcompute::index_replacer::call (assuming this is its registered name)