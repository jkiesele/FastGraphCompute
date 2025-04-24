#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "cuda_helpers.h"
#include "helpers.h"
#include <c10/macros/Macros.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() {                         \
    cudaError_t err = cudaGetLastError();                        \
    if (err != cudaSuccess) {                                    \
        printf("CUDA Kernel launch error: %s\n",                 \
               cudaGetErrorString(err));                         \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
}

__global__
void calc(
        const float * d_coords,
        const int * d_rs,
        const float * d_binswidth, //singleton
        const int * n_bins,

        int * d_assigned_bin,
        int * d_flat_assigned_bin,
        int * d_n_per_bin,

        const int n_vert,
        const int n_coords,
        const int n_rs,
        const int n_total_bins,
        const bool calc_n_per_bin
){
    int iv=blockIdx.x * blockDim.x + threadIdx.x;
    if(iv>=n_vert)
        return;

    ///same for cu

    int mul = 1;
    int idx = 0;
    const float epsilon = 1e-3f;

    for (int ic = n_coords-1; ic != -1; ic--) {
        
        float coord = d_coords[I2D(iv, ic, n_coords)]; 
        float scaled = coord / d_binswidth[0]; 
        int cidx = (int)floorf(scaled + epsilon);  

        if(cidx < 0) {
            printf("Warning: Vertex %d, coordinate %d (%f) yields negative bin index (%d). Clamping to 0.\n", iv, ic, coord, cidx);
            cidx = 0;
        }
        else if(cidx >= n_bins[ic]) {
            /* For upper boundary, check if the coordinate is within a small threshold of the boundary.
             * If so, silently clamp; otherwise, print a warning. */
            float upper_bound = n_bins[ic] * d_binswidth[0];
            if (cidx == n_bins[ic] && fabs(coord - upper_bound) < epsilon * 10) {
                // Silent clamp for coordinate exactly at the upper boundary
                cidx = n_bins[ic] - 1;
            } else {
                printf("Warning: Vertex %d, coordinate %d (%f) yields bin index %d out of range [0, %d). Clamping to %d.\n", 
                       iv, ic, coord, cidx, n_bins[ic], n_bins[ic]-1);
                cidx = n_bins[ic] - 1;
            }
        }
        d_assigned_bin[I2D(iv,ic+1,n_coords+1)]=cidx;

        if(n_bins[ic] > 0 && mul > INT_MAX / n_bins[ic]) {
            printf("ERROR: Integer overflow detected in thread %d at coordinate %d during multiplication. Aborting computation.\n", iv, ic);
            return;
        }
        idx += cidx * mul;
        mul *= n_bins[ic];

    }

    //get row split index last
    int rsidx=0;
    for(int irs=1 ; irs < n_rs ; irs++){
        if(d_rs[irs] > iv){
            break;
        }
        rsidx++;
    }
    
    if(mul > INT_MAX / (rsidx + 1)) {
        printf("ERROR: Integer overflow detected in thread %d when adding row-split index. Aborting computation.\n", iv);
        return;
    }
    idx += rsidx * mul;

    if(idx>=n_total_bins){
        printf("\nERROR: BinByCoordinatesOpFunctor: global index larger than total bins\n");//DEBUG if you see this you're screwed
        return;
    }

    d_assigned_bin[I2D(iv,0,n_coords+1)]=rsidx; //now this is c-style ordering with [rs, c_N, c_N-1, ..., c_0]
    d_flat_assigned_bin[iv]=idx;


    if(calc_n_per_bin){
        //atomic in parallel!
        atomicAdd(&d_n_per_bin[idx] , 1);

    }
    //end same for cu

}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bin_by_coordinates_cuda_fn(
    torch::Tensor coordinates,
    torch::Tensor row_splits,
    torch::Tensor bin_width,
    torch::Tensor nbins,
    bool calc_n_per_bin)
{
    CHECK_INPUT(coordinates);
    CHECK_INPUT(row_splits);
    CHECK_INPUT(bin_width);
    CHECK_INPUT(nbins);

    const auto n_vert = coordinates.size(0);
    const auto n_coords = coordinates.size(1);
    const auto n_rs = row_splits.size(0);
    const auto n_nbins = nbins.size(0);

    //check if bin_width is a singleton
    TORCH_CHECK(bin_width.size(0) == 1, "bin_by_coordinates_cuda: bin_width must be a singleton tensor");

    // Check if n_coords is equal to nbins.size(0)
    TORCH_CHECK(n_coords == nbins.size(0), "bin_by_coordinates_cuda: coordinates.size(1) must be equal to nbins.size(0)");

    const auto n_total_bins = nbins.to(torch::kCPU).prod().item<int>() * (n_rs - 1);
    
    auto output_n_per_bin_tensor = torch::zeros({ n_total_bins }, torch::TensorOptions().dtype(torch::kInt32).device(coordinates.device()));
    auto output_assigned_bin_tensor = torch::zeros({ n_vert, n_coords + 1 }, torch::TensorOptions().dtype(torch::kInt32).device(coordinates.device()));
    auto output_flat_assigned_bin_tensor = torch::zeros({ n_vert }, torch::TensorOptions().dtype(torch::kInt32).device(coordinates.device()));

    grid_and_block gb(n_vert,512);

    calc<<<gb.grid(),gb.block()>>>(
        coordinates.data_ptr<float>(),
        row_splits.data_ptr<int32_t>(),
        bin_width.data_ptr<float>(),
        nbins.data_ptr<int32_t>(),

        output_assigned_bin_tensor.data_ptr<int32_t>(),
        output_flat_assigned_bin_tensor.data_ptr<int32_t>(),
        output_n_per_bin_tensor.data_ptr<int32_t>(),

        n_vert,
        n_coords,
        n_rs,
        n_total_bins,
        calc_n_per_bin);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(output_assigned_bin_tensor, output_flat_assigned_bin_tensor, output_n_per_bin_tensor);
}
