#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <script.h>

// Input validation macros
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

// Helper function to check if all values in a tensor are finite
bool is_finite(const torch::Tensor& tensor) {
    return torch::isfinite(tensor).all().item<bool>();
}

// Main bin_by_coordinates function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
bin_by_coordinates(
    torch::Tensor coordinates,
    torch::Tensor row_splits,
    torch::Tensor bin_width,
    torch::Tensor n_bins,
    bool calc_n_per_bin,
    bool pre_normalized) {
    
    // Input validation
    CHECK_INPUT(coordinates);
    CHECK_INPUT(row_splits);
    TORCH_CHECK(bin_width.defined() ? bin_width.is_contiguous() : true, "bin_width must be contiguous");
    TORCH_CHECK(n_bins.defined() ? n_bins.is_contiguous() : true, "n_bins must be contiguous");
    
    // Check for non-finite values
    if (!is_finite(coordinates)) {
        throw std::runtime_error("BinByCoordinates: input coordinates contain non-finite values");
    }
    
    // Create a copy of coordinates for modification
    auto coords = coordinates.clone();
    
    // Normalize coordinates if not pre-normalized
    if (!pre_normalized) {
        auto min_coords = torch::min(coords, 0, true).values;
        coords = coords - min_coords;
    }
    
    // Calculate max coordinates and handle zero-range dimensions
    auto dmax_coords = torch::max(coords, 0).values;
    auto min_coords_per_dim = torch::min(coords, 0).values;
    
    // Handle zero-range dimensions
    dmax_coords = torch::where(min_coords_per_dim == dmax_coords, dmax_coords + 1.0, dmax_coords);
    
    // Add small epsilon to avoid boundary issues
    dmax_coords = dmax_coords + 1e-3;
    
    // Replace non-finite values with 1.0
    auto ones = torch::ones_like(dmax_coords);
    dmax_coords = torch::where(torch::isfinite(dmax_coords), dmax_coords, ones);
    
    // Ensure maximum coordinates are greater than 0
    if (!(dmax_coords > 0).all().item<bool>()) {
        throw std::runtime_error("BinByCoordinates: dmax_coords must be greater than zero.");
    }
    
    // Calculate bin_width or n_bins
    if (bin_width.defined()) {
        // Calculate n_bins from bin_width if not provided
        if (!n_bins.defined()) {
            n_bins = (dmax_coords / bin_width).to(torch::kInt32) + 1;
        }
    } else {
        // bin_width must be provided if n_bins is None
        if (!n_bins.defined()) {
            throw std::runtime_error("Either bin_width or n_bins must be provided.");
        }
        
        // Ensure n_bins has the coordinate dimension
        if (n_bins.dim() == 0) {
            n_bins = n_bins.repeat({coords.size(1)});
        }
        
        // Calculate bin_width from n_bins
        bin_width = dmax_coords / n_bins.to(torch::kFloat32);
        bin_width = torch::max(bin_width).unsqueeze(-1);  // Ensure uniform bin width
    }
    
    // Validate bin dimensions
    if (!(n_bins > 0).all().item<bool>()) {
        throw std::runtime_error("BinByCoordinates: n_bins must be greater than zero.");
    }
    if (!(bin_width > 0).all().item<bool>()) {
        throw std::runtime_error("BinByCoordinates: bin_width must be greater than zero.");
    }
    
    // Call the kernel through the registered operation - PyTorch will handle device dispatch
    auto [bin_indices, flat_bin_indices, n_per_bin] = torch::ops::bin_by_coordinates_func::bin_by_coordinates_func(
        coords, row_splits, bin_width, n_bins, calc_n_per_bin);
    
    return std::make_tuple(
        bin_indices,
        flat_bin_indices,
        n_bins,
        bin_width,
        n_per_bin
    );
}

// Register the function with PyTorch
TORCH_LIBRARY(bin_by_coordinates, m) {
    m.def("bin_by_coordinates(Tensor coordinates, Tensor row_splits, Tensor? bin_width, Tensor? n_bins, bool calc_n_per_bin, bool pre_normalized) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(bin_by_coordinates, CPU, m) {
    m.impl("bin_by_coordinates", bin_by_coordinates);
}

TORCH_LIBRARY_IMPL(bin_by_coordinates, CUDA, m) {
    m.impl("bin_by_coordinates", bin_by_coordinates);
}