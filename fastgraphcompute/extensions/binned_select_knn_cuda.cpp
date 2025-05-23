#include <torch/extension.h>
#include <vector>

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//forward declaration of cuda function
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
    int64_t K
);


// Function to dispatch based on input tensor types (int32 or int64)
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
    int64_t K
) {
    CHECK_INPUT(coordinates);
    CHECK_INPUT(bin_idx);
    CHECK_INPUT(dim_bin_idx);
    CHECK_INPUT(bin_boundaries);
    CHECK_INPUT(n_bins);
    CHECK_INPUT(bin_width);
    CHECK_INPUT(direction);
    
    return binned_select_knn_cuda_fn(
        coordinates,
        bin_idx,
        dim_bin_idx,
        bin_boundaries,
        n_bins,
        bin_width,
        direction,
        tf_compat,
        use_direction,
        K
    );
}