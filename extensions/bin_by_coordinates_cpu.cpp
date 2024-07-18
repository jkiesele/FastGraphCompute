#include <torch/extension.h>

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define I2D(i,j,Nj) j + Nj*i

void computeHelper(
    const int *n_bins,
    int *out_tot_bins,
    int n_nbins,
    int nrs
) {
    int n = 1;
    printf("n bins:");
    for (int i = 0; i < n_nbins; i++) {
        n *= n_bins[i];
        printf(" %d ,", n_bins[i]);
    }
    printf("\n");

    *out_tot_bins = n * (nrs - 1);
}

void set_defaults(
    int32_t *d_n_per_bin,
    const size_t n_total_bins)
{
    for (size_t i = 0; i < n_total_bins; ++i) {
        d_n_per_bin[i] = 0;
    }
}

void calc(
    const float *d_coords,
    const int32_t *d_rs,
    const float *d_binswidth, // singleton
    const int32_t *n_bins,    // singleton
    int32_t *d_assigned_bin,
    int32_t *d_flat_assigned_bin,
    int32_t *d_n_per_bin,
    const size_t n_vert,
    const size_t n_coords,
    const size_t n_rs,
    const size_t n_total_bins,
    const bool calc_n_per_bin)
{
    for (size_t iv = 0; iv < n_vert; ++iv) {
        int mul = 1;
        int idx = 0;

        for (int ic = n_coords-1; ic > -1; ic--) {

            int cidx = d_coords[I2D(iv,ic,n_coords)] / d_binswidth[0];

            if(cidx < 0 || cidx >= n_bins[ic]){
                printf("index %d of coordinate %d exceeds n bins %d or below 0, coord %e\n",cidx,ic,n_bins[ic],d_coords[I2D(iv,ic,n_coords)]);
                cidx = 0; //stable, but will create bogus later
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
            printf("global index larger than total bins\n");//DEBUG if you see this you're screwed
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

void compute(
    const float *d_coords,
    const int32_t *d_rs,
    const float *d_binswidth, // singleton
    const int32_t *n_bins,    // singleton
    int32_t *d_assigned_bin,
    int32_t *d_flat_assigned_bin,
    int32_t *d_n_per_bin,
    const size_t n_vert,
    const size_t n_coords,
    const size_t n_rs,
    const size_t n_total_bins,
    const bool calc_n_per_bin)
{
    set_defaults(d_n_per_bin, n_total_bins);
    calc(d_coords, d_rs, d_binswidth, n_bins,
         d_assigned_bin, d_flat_assigned_bin, d_n_per_bin,
         n_vert, n_coords, n_rs, n_total_bins, calc_n_per_bin);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
bin_by_coordinates_cpu(
    torch::Tensor coordinates,
    torch::Tensor row_splits,
    torch::Tensor bin_width,
    torch::Tensor nbins,
    bool calc_n_per_bin)
{
    const size_t n_vert = coordinates.size(0);
    const size_t n_coords = coordinates.size(1);
    const size_t n_rs = row_splits.size(0);
    const size_t n_total_bins = nbins.numel();

    auto options = torch::TensorOptions().dtype(torch::kInt32);
    auto output_assigned_bin = torch::empty({n_vert, n_coords + 1}, options);
    auto output_flat_assigned_bin = torch::empty({n_vert}, options);
    auto output_n_per_bin = torch::zeros({n_total_bins}, options);

    auto d_coords = coordinates.data_ptr<float>();
    auto d_rs = row_splits.data_ptr<int32_t>();
    auto d_bin_width = bin_width.data_ptr<float>();
    auto d_nbins = nbins.data_ptr<int32_t>();
    auto d_assigned_bin = output_assigned_bin.data_ptr<int32_t>();
    auto d_flat_assigned_bin = output_flat_assigned_bin.data_ptr<int32_t>();
    auto d_n_per_bin = output_n_per_bin.data_ptr<int32_t>();

    compute(d_coords, d_rs, d_bin_width, d_nbins,
            d_assigned_bin, d_flat_assigned_bin, d_n_per_bin,
            n_vert, n_coords, n_rs, n_total_bins, calc_n_per_bin);

    return std::make_tuple(output_assigned_bin, output_flat_assigned_bin, output_n_per_bin);
}

TORCH_LIBRARY(bin_by_coordinates_cpu, m) {
    m.def("bin_by_coordinates_cpu", bin_by_coordinates_cpu);
}