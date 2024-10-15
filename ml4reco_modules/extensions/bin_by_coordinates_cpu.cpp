#include <torch/extension.h>
#include <vector>

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be CPU tensor")
#define I2D(i,j,Nj) ((i) * (Nj) + (j))


static void calc(
        const float * d_coords,
        const int * d_rs,

        const float * d_binswidth, //singleton
        const int * n_bins,//singleton

        int * d_assigned_bin,
        int * d_flat_assigned_bin,
        int * d_n_per_bin,

        const int n_vert,
        const int n_coords,
        const int n_rs,
        const int n_total_bins,
        const bool calc_n_per_bin
){

    for(int iv=0; iv<n_vert; iv++){

        ///same for cu

        int mul = 1;
        int idx = 0;

        for (int ic = n_coords-1; ic > -1; ic--) {

            int cidx = d_coords[I2D(iv,ic,n_coords)] / d_binswidth[0];

            if(cidx < 0 || cidx >= n_bins[ic]){
                printf("Fatal error: index %d of coordinate %d exceeds n bins %d or below 0, coord %e\n",cidx,ic,n_bins[ic],d_coords[I2D(iv,ic,n_coords)]);
                cidx = n_bins[ic] - 1;//stable, but will create bogus later
            }
            d_assigned_bin[I2D(iv,ic+1,n_coords+1)]=cidx;

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

        idx += rsidx * mul;

        if(idx>=n_total_bins){
            printf("Fatal error:  global index larger than total bins\n");//DEBUG if you see this you're screwed
            continue;
        }

        d_assigned_bin[I2D(iv,0,n_coords+1)]=rsidx; //now this is c-style ordering with [rs, c_N, c_N-1, ..., c_0]
        d_flat_assigned_bin[iv]=idx;


        if(calc_n_per_bin){
            //atomic in parallel!
            d_n_per_bin[idx] += 1;

        }
        //end same for cu

    }//iv loop
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


    //check if bin_width is a singleton
    if (bin_width.size(0) != 1) {
        throw std::invalid_argument("bin_by_coordinates_cpu: bin_width must be a singleton tensor");
    }

    // throw exception if n_coords is not nbins.size(0)
    if (n_coords != nbins.size(0)) {
        throw std::invalid_argument("bin_by_coordinates_cpu: coordinates.size(1) must be equal to nbins.size(0)");
    }
    const auto n_rs = row_splits.size(0);
    const auto n_total_bins = nbins.prod().item<int>() * (n_rs - 1);

    auto options = torch::TensorOptions().dtype(torch::kInt32);
    auto assigned_bin = torch::empty({n_vert, n_coords+1}, options);
    auto flat_assigned_bin = torch::empty(n_vert, options);
    auto n_per_bin = torch::zeros(n_total_bins, options);

    calc(
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
        calc_n_per_bin);


    return std::make_tuple(assigned_bin, flat_assigned_bin, n_per_bin);
}

TORCH_LIBRARY(bin_by_coordinates_cpu, m) {
    m.def("bin_by_coordinates_cpu", &bin_by_coordinates_cpu);
}
