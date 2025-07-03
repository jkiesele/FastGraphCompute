#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <algorithm>

// Forward declaration for bin_by_coordinates function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
bin_by_coordinates(
    const torch::Tensor& coordinates,
    const torch::Tensor& row_splits,
    c10::optional<torch::Tensor> bin_width_opt,
    c10::optional<torch::Tensor> n_bins_opt,
    bool calc_n_per_bin,
    bool pre_normalized);

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

torch::Tensor index_replacer_cuda_fn(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements);

torch::Tensor index_replacer_cpu_fn(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements);

torch::Tensor binned_select_knn_grad_cuda_fn(
    torch::Tensor grad_distances,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor coordinates);

torch::Tensor binned_select_knn_grad_cpu_fn(
    torch::Tensor grad_distances,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor coordinates);

// Helper function to ensure tensors are contiguous and on correct device
template<typename... Tensors>
auto make_contiguous_on_device(torch::Device device, Tensors&&... tensors) {
    return std::make_tuple(std::forward<Tensors>(tensors).contiguous().to(device)...);
}

// Helper function to calculate optimal number of bins
torch::Tensor calculate_optimal_bins(const torch::Tensor& row_splits, int64_t K, int64_t max_bin_dims, 
                                     const c10::optional<torch::Tensor>& n_bins_user, 
                                     const torch::TensorOptions& int32_options) {
    if (n_bins_user.has_value()) {
        return n_bins_user.value();
    }
    
    auto elems_per_rs = row_splits.size(0) > 0 ? 
        (torch::max(row_splits).to(torch::kFloat32) / static_cast<float>(row_splits.size(0)) + 1).to(torch::kInt32) :
        torch::tensor(static_cast<int64_t>(1), int32_options);
    
    auto n_bins = torch::pow(elems_per_rs.to(torch::kFloat32) / (static_cast<float>(K) / 32.0f), 
                            1.0f / static_cast<float>(max_bin_dims)).to(torch::kInt32);
    
    return torch::clamp(n_bins, static_cast<int64_t>(5), static_cast<int64_t>(30));
}

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

        TORCH_CHECK(coords.size(1) > 0, "Input coordinates must have at least one dimension.");
        TORCH_CHECK(max_bin_dims_user > 0, "max_bin_dims must be greater than 0.");

        auto original_device = coords.device();
        auto int32_options = torch::TensorOptions().dtype(torch::kInt32).device(original_device);
        auto int64_options = torch::TensorOptions().dtype(torch::kInt64).device(original_device);
        int64_t max_bin_dims = std::min(max_bin_dims_user, coords.size(1));

        // Calculate bins and prepare coordinates for binning
        auto n_bins = calculate_optimal_bins(row_splits, K, max_bin_dims, n_bins_user, int32_options);
        auto bin_coords = coords.size(1) > max_bin_dims ? 
            coords.slice(static_cast<int64_t>(1), static_cast<int64_t>(0), max_bin_dims) : coords;

        // Perform binning - call function directly
        auto binning_result = bin_by_coordinates(
                bin_coords, row_splits, torch::Tensor(), n_bins, true, false);
        
        auto dbinning = std::get<0>(binning_result);
        auto binning = std::get<1>(binning_result);
        auto nb = std::get<2>(binning_result);
        auto bin_width = std::get<3>(binning_result);
        auto nper = std::get<4>(binning_result);

        // Sort by bin assignment
        auto sorting_indices = binning.numel() > 0 ? 
            torch::argsort(binning, static_cast<int64_t>(0)).to(torch::kInt64) :
            torch::empty({0}, binning.options().dtype(torch::kInt64));

        // Gather sorted tensors
        auto scoords = coords.index_select(static_cast<int64_t>(0), sorting_indices);
        auto sbinning = binning.index_select(static_cast<int64_t>(0), sorting_indices);
        auto sdbinning = dbinning.index_select(static_cast<int64_t>(0), sorting_indices);
        
        c10::optional<torch::Tensor> sdirection;
        if (direction.has_value()) {
            sdirection = direction.value().index_select(static_cast<int64_t>(0), sorting_indices);
        }

        // Create bin boundaries - fix torch::cat usage
        auto zero_tensor = torch::zeros({1}, int64_options.device(nper.device()));
        std::vector<torch::Tensor> tensor_list;
        tensor_list.push_back(zero_tensor);
        tensor_list.push_back(nper.to(torch::kInt64));
        auto bin_boundaries = torch::cumsum(torch::cat(tensor_list, static_cast<int64_t>(0)), static_cast<int64_t>(0), torch::kInt64);

        // Fix the template syntax for item()
        auto max_bin_boundaries = torch::max(bin_boundaries).item().toInt();
        auto max_row_splits = torch::max(row_splits).item().toInt();
        TORCH_CHECK(max_bin_boundaries == max_row_splits,
                    "Bin boundaries do not match row splits.");

        // Prepare inputs for KNN kernel
        auto direction_input = (direction.has_value() && direction.value().numel() > 0) ? 
            direction.value() : torch::empty({0}, sdbinning.options().device(scoords.device()));
        bool use_direction = direction.has_value() && direction.value().numel() > 0;

        // Ensure kernel inputs are contiguous and on correct device
        auto kernel_device = scoords.device();
        auto [k_scoords, k_sbinning, k_sdbinning, k_bin_boundaries, k_n_bins, k_bin_width, k_direction] = 
            make_contiguous_on_device(kernel_device, scoords, sbinning, sdbinning, 
                                     bin_boundaries, nb, bin_width, direction_input);

        // Call KNN kernel directly based on device type
        std::tuple<torch::Tensor, torch::Tensor> knn_result;
        if (k_scoords.device().is_cuda()) {
            knn_result = binned_select_knn_cuda_fn(
                k_scoords, k_sbinning, k_sdbinning, k_bin_boundaries,
                k_n_bins, k_bin_width, k_direction,
                torch_compatible_indices, use_direction, K);
        } else {
            knn_result = binned_select_knn_cpu(
                k_scoords, k_sbinning, k_sdbinning, k_bin_boundaries,
                k_n_bins, k_bin_width, k_direction,
                torch_compatible_indices, use_direction, K);
        }
        
        auto idx_sorted = std::get<0>(knn_result);
        auto dist_sorted = std::get<1>(knn_result);

        // Replace indices to original order - call function directly
        torch::Tensor idx_unsorted;
        if (idx_sorted.numel() > 0) {
            if (idx_sorted.device().is_cuda()) {
                idx_unsorted = index_replacer_cuda_fn(idx_sorted, sorting_indices);
            } else {
                idx_unsorted = index_replacer_cpu_fn(idx_sorted, sorting_indices);
            }
        } else {
            idx_unsorted = torch::empty_like(idx_sorted);
        }

        // Scatter results back to original order
        auto sorting_indices_long = sorting_indices; // already int64
        auto dist_final = torch::empty_like(dist_sorted, dist_sorted.options().device(original_device));
        auto idx_final = torch::empty_like(idx_unsorted, idx_unsorted.options().device(original_device));

        if (dist_sorted.numel() > 0) {
            dist_final.scatter_(static_cast<int64_t>(0), sorting_indices_long.unsqueeze(-1).expand_as(dist_sorted), dist_sorted);
        }
        if (idx_unsorted.numel() > 0) {
            idx_final.scatter_(static_cast<int64_t>(0), sorting_indices_long.unsqueeze(-1).expand_as(idx_unsorted), idx_unsorted);
        }
        
        // Fix save_for_backward - use proper variable_list
        torch::autograd::variable_list saved_tensors;
        saved_tensors.push_back(idx_final);
        saved_tensors.push_back(dist_final);
        saved_tensors.push_back(coords);
        ctx->save_for_backward(saved_tensors);
        
        // Fix return statement - create proper variable_list
        torch::autograd::variable_list outputs;
        outputs.push_back(idx_final.to(original_device));
        outputs.push_back(dist_final.to(original_device));
        return outputs;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        
        auto saved_tensors = ctx->get_saved_variables();
        auto idx = saved_tensors[0];
        auto dist = saved_tensors[1];
        auto original_coords = saved_tensors[2];
        auto grad_dist = grad_outputs[1];

        torch::Tensor grad_coordinates;
        // Fix needs_input_grad access
        if (ctx->needs_input_grad(0)) {
            if (grad_dist.defined()) {
                auto kernel_device = grad_dist.device();
                auto [k_grad_dist, k_idx, k_dist, k_coords] = 
                    make_contiguous_on_device(kernel_device, grad_dist, idx, dist, original_coords);

                // Call gradient function directly based on device type
                if (k_grad_dist.device().is_cuda()) {
                    grad_coordinates = binned_select_knn_grad_cuda_fn(
                        k_grad_dist, k_idx, k_dist, k_coords);
                } else {
                    grad_coordinates = binned_select_knn_grad_cpu_fn(
                        k_grad_dist, k_idx, k_dist, k_coords);
                }
            } else {
                grad_coordinates = torch::zeros_like(original_coords, original_coords.options().requires_grad(false));
            }
        }
        
        // Return proper variable_list
        torch::autograd::variable_list grad_inputs;
        grad_inputs.push_back(grad_coordinates);
        grad_inputs.push_back(torch::Tensor());
        grad_inputs.push_back(torch::Tensor());
        grad_inputs.push_back(torch::Tensor());
        grad_inputs.push_back(torch::Tensor());
        grad_inputs.push_back(torch::Tensor());
        grad_inputs.push_back(torch::Tensor());
        return grad_inputs;
    }
};

// Main function that applies the autograd operation
std::tuple<torch::Tensor, torch::Tensor> binned_select_knn_cpp_op(
    torch::Tensor coords,
    torch::Tensor row_splits,
    int64_t K,
    c10::optional<torch::Tensor> direction,
    c10::optional<torch::Tensor> n_bins_user,
    int64_t max_bin_dims_user,
    bool torch_compatible_indices) {
    
    auto result = BinnedKNNAutograd::apply(coords, row_splits, K, direction, 
                                          n_bins_user, max_bin_dims_user, torch_compatible_indices);
    return std::make_tuple(result[0], result[1]);
}

// Operator Registration
TORCH_LIBRARY(fastgraphcompute_custom_ops, m) {
    m.def("binned_select_knn(Tensor coords, Tensor row_splits, int K, Tensor? direction, Tensor? n_bins, int max_bin_dims, bool torch_compatible_indices) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fastgraphcompute_custom_ops, Autograd, m) {
    m.impl("binned_select_knn", TORCH_FN(binned_select_knn_cpp_op));
}
