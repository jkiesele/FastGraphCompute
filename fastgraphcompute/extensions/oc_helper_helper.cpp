#include <torch/extension.h>
#include <tuple>

// Workaround for `at::unique` as not all versions of torch have a c++ binding for it
// This function computes unique values and their counts using sorting and tensor operations
std::tuple<torch::Tensor, torch::Tensor> unique_with_counts(const torch::Tensor& input) {
    // Sort the input tensor
    torch::Tensor sorted = std::get<0>(torch::sort(input));

    // Find unique elements by checking differences
    torch::Tensor diff = sorted.diff();
    torch::Tensor one = torch::tensor({1}, diff.options());
    torch::Tensor unique_mask = torch::cat({one, diff != 0}).to(torch::kBool);
    torch::Tensor unique_vals = sorted.masked_select(unique_mask);

    // Get indices of unique elements
    torch::Tensor unique_indices = torch::nonzero(unique_mask).flatten();

    // Append the length of the sorted tensor to unique_indices
    torch::Tensor total_size = torch::tensor({sorted.size(0)}, unique_indices.options());
    torch::Tensor indices_with_end = torch::cat({unique_indices, total_size});

    // Compute counts by subtracting adjacent indices
    torch::Tensor counts = indices_with_end.slice(0, 1) - indices_with_end.slice(0, 0, -1);

    return std::make_tuple(unique_vals, counts);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> max_same_valued_entries_per_row_split(
    const torch::Tensor& asso_idx,
    torch::Tensor row_splits,  // Not const because we might modify it
    bool filter_negative) {

    // Check that asso_idx is on CPU or CUDA
    TORCH_CHECK(asso_idx.is_cuda() || asso_idx.device().is_cpu(),
                "asso_idx must be on CPU or CUDA");

    // Move row_splits to CPU if it's not already
    if (!row_splits.device().is_cpu()) {
        row_splits = row_splits.to(torch::kCPU);
    }
    // check if row splits are in32
    TORCH_CHECK(row_splits.dtype() == torch::kInt32, "row_splits must be of type int32");

    int64_t n_row_splits = row_splits.size(0) - 1;

    // Prepare output tensors on the same device as asso_idx
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(asso_idx.device());
    torch::Tensor max_per_split = torch::zeros({n_row_splits}, options);
    torch::Tensor objects_per_split = torch::zeros({n_row_splits}, options);

    // Accessor for row_splits (CPU tensor)
    auto row_splits_accessor = row_splits.accessor<int32_t, 1>();

    for (int64_t rs_idx = 0; rs_idx < n_row_splits; ++rs_idx) {
        int64_t start_vertex = row_splits_accessor[rs_idx];
        int64_t end_vertex = row_splits_accessor[rs_idx + 1];

        // Extract the slice for the current row split
        torch::Tensor asso_idx_slice = asso_idx.narrow(0, start_vertex, end_vertex - start_vertex);

        // Filter out negative values if needed
        torch::Tensor asso_idx_filtered;
        if (filter_negative) {
            asso_idx_filtered = asso_idx_slice.masked_select(asso_idx_slice >= 0);
        } else {
            asso_idx_filtered = asso_idx_slice;
        }

        if (asso_idx_filtered.numel() == 0) {
            continue;
        }

        // Get unique values and inverse indices
        torch::Tensor unique_vals, counts;
        std::tie(unique_vals, counts) = unique_with_counts(asso_idx_filtered);

        // Find the maximum count and store it
        int64_t max_count = counts.max().item().to<int64_t>();
        max_per_split[rs_idx] = max_count;
        objects_per_split[rs_idx] = unique_vals.size(0);
    }

    // Find the global maximum
    auto global_max = max_per_split.max();

    //move all back to the device of asso_idx
    max_per_split = max_per_split.to(asso_idx.device());
    objects_per_split = objects_per_split.to(asso_idx.device());
    global_max = global_max.to(asso_idx.device());

    return std::make_tuple(max_per_split, global_max, objects_per_split);
}

// Bindings
TORCH_LIBRARY(oc_helper_helper, m) {
    m.def("max_same_valued_entries_per_row_split", &max_same_valued_entries_per_row_split);
    m.def("unique_with_counts", &unique_with_counts); //expose just for testing
}
