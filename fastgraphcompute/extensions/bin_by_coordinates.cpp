#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <c10/util/Optional.h>

// Forward declarations for the implementation functions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bin_by_coordinates_cpu_fn(
    torch::Tensor coords,
    torch::Tensor row_splits,
    torch::Tensor binswidth,
    torch::Tensor nbins,
    bool calc_n_per_bin);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bin_by_coordinates_cuda_fn(
    torch::Tensor coords,
    torch::Tensor row_splits,
    torch::Tensor binswidth,
    torch::Tensor nbins,
    bool calc_n_per_bin);

// Input validation macros
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

// Helper function to check if all values in a tensor are finite
bool is_finite(const torch::Tensor& tensor) {
    return torch::isfinite(tensor).all().item().to<bool>();
}

// Main bin_by_coordinates function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
bin_by_coordinates(
    const torch::Tensor& coordinates,
    const torch::Tensor& row_splits,
    c10::optional<torch::Tensor> bin_width_opt,
    c10::optional<torch::Tensor> n_bins_opt,
    bool calc_n_per_bin,
    bool pre_normalized) {
    
    // Input validation
    CHECK_INPUT(coordinates);
    CHECK_INPUT(row_splits);
    
    // Convert optional inputs to tensors (may be undefined)
    torch::Tensor bin_width = bin_width_opt.has_value() ? *bin_width_opt : torch::Tensor();
    torch::Tensor n_bins    = n_bins_opt.has_value()    ? *n_bins_opt    : torch::Tensor();
    
    if (bin_width.defined()) TORCH_CHECK(bin_width.is_contiguous(), "bin_width must be contiguous");
    if (n_bins.defined()) TORCH_CHECK(n_bins.is_contiguous(), "n_bins must be contiguous");
    
    // Check for non-finite values
    if (!is_finite(coordinates)) {
        throw std::runtime_error("BinByCoordinates: input coordinates contain non-finite values");
    }
    
    // Create a copy of coordinates for modification
    auto coords = coordinates.clone();
    
    // Normalize coordinates if not pre-normalized
    if (!pre_normalized) {
        auto min_coords = std::get<0>(torch::min(coords, 0, true));
        coords = coords - min_coords;
    }
    
    // Calculate max coordinates and handle zero-range dimensions
    auto dmax_coords = std::get<0>(torch::max(coords, 0));
    auto min_coords_per_dim = std::get<0>(torch::min(coords, 0));
    
    // Handle zero-range dimensions
    dmax_coords = torch::where(min_coords_per_dim == dmax_coords, dmax_coords + 1.0, dmax_coords);
    
    // Add small epsilon to avoid boundary issues
    dmax_coords = dmax_coords + 1e-3;
    
    // Replace non-finite values with 1.0
    auto ones = torch::ones_like(dmax_coords);
    dmax_coords = torch::where(torch::isfinite(dmax_coords), dmax_coords, ones);
    
    // Ensure maximum coordinates are greater than 0
    if (!(dmax_coords > 0).all().item().to<bool>()) {
        throw std::runtime_error("BinByCoordinates: dmax_coords must be greater than zero.");
    }
    
    // Calculate bin_width or n_bins
    torch::Tensor final_bin_width;
    torch::Tensor final_n_bins;
    
    if (bin_width.defined() && bin_width.numel() > 0) {
        final_bin_width = bin_width;
        // Calculate n_bins from bin_width if not provided (i.e. undefined or empty)
        if (!n_bins.defined() || n_bins.numel() == 0) {
            final_n_bins = (dmax_coords / final_bin_width).to(torch::kInt32) + 1;
        } else {
            final_n_bins = n_bins;
        }
    } else {
        // bin_width must be provided if n_bins is undefined / empty
        if (!n_bins.defined() || n_bins.numel() == 0) {
            throw std::runtime_error("Either bin_width or n_bins must be provided.");
        }

        final_n_bins = n_bins;
        // Ensure n_bins has the coordinate dimension
        if (final_n_bins.dim() == 0) {
            final_n_bins = final_n_bins.repeat({coords.size(1)});
        }

        // Calculate bin_width from n_bins
        final_bin_width = dmax_coords / final_n_bins.to(torch::kFloat32);
        final_bin_width = torch::max(final_bin_width).unsqueeze(-1);  // Ensure uniform bin width
    }
    
    // Validate bin dimensions
    if (!(final_n_bins > 0).all().item().to<bool>()) {
        throw std::runtime_error("BinByCoordinates: n_bins must be greater than zero.");
    }
    if (!(final_bin_width > 0).all().item().to<bool>()) {
        throw std::runtime_error("BinByCoordinates: bin_width must be greater than zero.");
    }
    
    // Call the appropriate implementation based on device type
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> result;
    if (coords.device().is_cuda()) {
        result = bin_by_coordinates_cuda_fn(coords, row_splits.to(torch::kInt32), final_bin_width, final_n_bins, calc_n_per_bin);
    } else {
        result = bin_by_coordinates_cpu_fn(coords, row_splits.to(torch::kInt32), final_bin_width, final_n_bins, calc_n_per_bin);
    }
    
    auto bin_indices = std::get<0>(result);
    auto flat_bin_indices = std::get<1>(result);
    auto n_per_bin = std::get<2>(result);
    
    return std::make_tuple(
        bin_indices,
        flat_bin_indices,
        final_n_bins,
        final_bin_width,
        n_per_bin
    );
}

// Register the function with PyTorch
TORCH_LIBRARY(bin_by_coordinates, m) {
    m.def("bin_by_coordinates(Tensor coordinates, Tensor row_splits, Tensor? bin_width, Tensor? n_bins, bool calc_n_per_bin, bool pre_normalized) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(bin_by_coordinates, CPU, m) {
    m.impl("bin_by_coordinates", TORCH_FN(bin_by_coordinates));
}

TORCH_LIBRARY_IMPL(bin_by_coordinates, CUDA, m) {
    m.impl("bin_by_coordinates", TORCH_FN(bin_by_coordinates));
}