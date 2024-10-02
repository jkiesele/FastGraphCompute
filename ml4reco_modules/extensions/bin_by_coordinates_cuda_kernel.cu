#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define I2D(i, j, Nj) ((i) * (Nj) + (j))

template <typename scalar_t>
__global__ void set_defaults(
    int *d_n_per_bin,
    const int n_total_bins)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_total_bins) {
        d_n_per_bin[i] = 0;
    }
}

template <typename scalar_t>
__global__ void calc(
    const scalar_t *d_coords,
    const int *d_rs,
    const scalar_t *d_binswidth,
    const int *n_bins,
    int *d_assigned_bin,
    int *d_flat_assigned_bin,
    int *d_n_per_bin,
    const int n_vert,
    const int n_coords,
    const int n_rs,
    const int n_total_bins,
    const bool calc_n_per_bin)
{
    const int iv = blockIdx.x * blockDim.x + threadIdx.x;
    if (iv >= n_vert) return;

    int mul = 1;
    int idx = 0;

    for (int ic = n_coords - 1; ic >= 0; --ic) {
        int cidx = d_coords[I2D(iv, ic, n_coords)] / d_binswidth[0];
        if (cidx >= n_bins[ic]) {
            printf("Overflow warning: index %d of coordinate %d exceeds n bins %d\n", cidx, ic, n_bins[ic]);
            cidx = n_bins[ic] - 1;
        }
        else if (cidx < 0) {
            printf("Underflow warning: index %d of coordinate %d less than n bins %d\n", cidx, ic, n_bins[ic]);
            cidx=0;
        }
        d_assigned_bin[I2D(iv, ic + 1, n_coords + 1)] = cidx;
        idx += cidx * mul;
        mul *= n_bins[ic];
    }

    int rsidx = 0;
    for (int irs = 1; irs < n_rs; ++irs) {
        if (d_rs[irs] > iv) {
            break;
        }
        rsidx++;
    }

    idx += rsidx * mul;

    if (idx >= n_total_bins) {
        printf("global index larger than total bins %d %d \n", idx, n_total_bins);
        return;
    }

    d_assigned_bin[I2D(iv, 0, n_coords + 1)] = rsidx;
    d_flat_assigned_bin[iv] = idx;

    if (calc_n_per_bin) {
        atomicAdd(&d_n_per_bin[idx], 1);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bin_by_coordinates_cuda_fn(
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

    const auto n_vert = coords.size(0);
    const auto n_coords = coords.size(1);
    const auto n_rs = row_splits.size(0);
    const auto n_nbins = nbins.size(0);

    int n_total_bins = 1;
    auto cpu_nbins = nbins.to(torch::kCPU);
    for (int i = 0; i < n_nbins; ++i) {
        n_total_bins *= cpu_nbins[i].item<int>();
    }
    n_total_bins *= (n_rs - 1);

    auto output_n_per_bin_tensor = torch::zeros({ n_total_bins }, torch::TensorOptions().dtype(torch::kInt32).device(coords.device()));
    auto output_assigned_bin_tensor = torch::zeros({ n_vert, n_coords + 1 }, torch::TensorOptions().dtype(torch::kInt32).device(coords.device()));
    auto output_flat_assigned_bin_tensor = torch::zeros({ n_vert }, torch::TensorOptions().dtype(torch::kInt32).device(coords.device()));

    // Initialize d_n_per_bin to zeros
    int block_size = 256;
    int num_blocks = (n_total_bins + block_size - 1) / block_size;
    AT_DISPATCH_FLOATING_TYPES(coords.type(), "set_defaults", ([&] {
        set_defaults<scalar_t> <<<num_blocks, block_size>>> (
            output_n_per_bin_tensor.data_ptr<int>(),
            n_total_bins);
    }));

    // Calculate bin assignments
    num_blocks = (n_vert + block_size - 1) / block_size;
    AT_DISPATCH_FLOATING_TYPES(coords.type(), "calc", ([&] {
        calc<scalar_t> <<<num_blocks, block_size>>> (
            coords.data_ptr<scalar_t>(),
            row_splits.data_ptr<int>(),
            binswidth.data_ptr<scalar_t>(),
            nbins.data_ptr<int>(),
            output_assigned_bin_tensor.data_ptr<int>(),
            output_flat_assigned_bin_tensor.data_ptr<int>(),
            output_n_per_bin_tensor.data_ptr<int>(),
            n_vert,
            n_coords,
            n_rs,
            n_total_bins,
            calc_n_per_bin);
    }));

    return std::make_tuple(output_assigned_bin_tensor, output_flat_assigned_bin_tensor, output_n_per_bin_tensor);
}
