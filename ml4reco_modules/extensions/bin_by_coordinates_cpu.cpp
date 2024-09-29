#include <torch/extension.h>
#include <vector>

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be CPU tensor")
#define I2D(i,j,Nj) ((i) * (Nj) + (j))

// Helper to compute total bins based on the product of bins per dimension
void computeTotalBins(
    const int *n_bins,
    int *out_tot_bins,
    int n_nbins,
    int nrs
) {
    int n = 1;
    for (int i = 0; i < n_nbins; i++) {
        n *= n_bins[i];
    }
    *out_tot_bins = n * nrs;
}

// Initialize number per bin array to zero
static void setDefaults(
    int32_t *d_n_per_bin,
    size_t n_total_bins
) {
    std::fill_n(d_n_per_bin, n_total_bins, 0);
}

// Main computation function
static void computeAssignments(
    const float *d_coords,
    const int32_t *d_rs,
    const float *d_binswidth,
    const int32_t *n_bins,
    int32_t *d_assigned_bin,
    int32_t *d_flat_assigned_bin,
    int32_t *d_n_per_bin,
    size_t n_vert,
    size_t n_coords,
    size_t n_rs,
    size_t n_total_bins,
    bool calc_n_per_bin
) {
    int n_tot_bins = 0;
    computeTotalBins(n_bins, &n_tot_bins, n_coords, n_rs);

    for (size_t iv = 0; iv < n_vert; ++iv) {
        int mul = 1;
        int idx = 0;

        for (int ic = n_coords - 1; ic >= 0; --ic) {
            int cidx = static_cast<int>(d_coords[I2D(iv, ic, n_coords)] / d_binswidth[0]);
            if(cidx < 0 || cidx >= n_bins[ic]){
                cidx = std::min(std::max(0, cidx), n_bins[ic] - 1);
            }
            d_assigned_bin[I2D(iv, ic, n_coords)] = cidx;
            idx += cidx * mul;
            mul *= n_bins[ic];
        }

        int rsidx = std::upper_bound(d_rs, d_rs + n_rs, iv) - d_rs - 1;
        idx += rsidx * mul;
        if (idx < n_total_bins)
        {
            d_flat_assigned_bin[iv] = idx;
            if (calc_n_per_bin) {
                d_n_per_bin[idx]++;
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bin_by_coordinates_cpu(
    torch::Tensor coordinates,
    torch::Tensor row_splits,
    torch::Tensor bin_width,
    torch::Tensor nbins,
    bool calc_n_per_bin
) {
    CHECK_CPU(coordinates);
    CHECK_CPU(row_splits);
    CHECK_CPU(bin_width);
    CHECK_CPU(nbins);

    const auto n_vert = coordinates.size(0);
    const auto n_coords = coordinates.size(1);
    const auto n_rs = row_splits.size(0);
    const auto n_total_bins = nbins.prod().item<int>();

    auto options = torch::TensorOptions().dtype(torch::kInt32);
    auto assigned_bin = torch::empty({n_vert, n_coords}, options);
    auto flat_assigned_bin = torch::empty(n_vert, options);
    auto n_per_bin = torch::zeros(n_total_bins, options);

    computeAssignments(
        coordinates.data_ptr<float>(),
        row_splits.data_ptr<int32_t>(),
        bin_width.data_ptr<float>(),
        nbins.data_ptr<int32_t>(),
        assigned_bin.data_ptr<int32_t>(),
        flat_assigned_bin.data_ptr<int32_t>(),
        n_per_bin.data_ptr<int32_t>(),
        n_vert,
        n_coords,
        n_rs,
        n_total_bins,
        calc_n_per_bin
    );

    return std::make_tuple(assigned_bin, flat_assigned_bin, n_per_bin);
}

TORCH_LIBRARY(bin_by_coordinates_cpu, m) {
    m.def("bin_by_coordinates_cpu", &bin_by_coordinates_cpu);
}
