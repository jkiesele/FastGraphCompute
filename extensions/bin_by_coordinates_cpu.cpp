#include <torch/extension.h>
#include <cmath> // For std::floor

#define CHECK_INPUT(x) AT_ASSERTM(!x.is_cuda(), #x " must be a CPU tensor")
#define I2D(i, j, n) ((i)*(n) + (j))

void computeHelper(
            const int32_t * n_bins,
            int32_t * out_tot_bins,
            int32_t n_nbins,
            int32_t nrs)
{
    int32_t n=1;
    printf("n bins:");
    for(int32_t i=0;i<n_nbins;i++){
        n*=n_bins[i];
        printf(" %d ,", n_bins[i]);
    }
    printf("\n");

    *out_tot_bins=n*(nrs-1);
}


static void set_defaults(
        int32_t * d_n_per_bin,
        const int32_t n_total_bins
       )
{
    for(int32_t i=0;i<n_total_bins;i++)
        d_n_per_bin[i]=0;
}

static void calc(
        const float_t * d_coords,
        const int32_t * d_rs,

        const float_t * d_binswidth, //singleton
        const int32_t * n_bins,//singleton

        int32_t * d_assigned_bin,
        int32_t * d_flat_assigned_bin,
        int32_t * d_n_per_bin,

        const int32_t n_vert,
        const int32_t n_coords,
        const int32_t n_rs,
        const int32_t n_total_bins,
        const bool calc_n_per_bin)
{

    for(int32_t iv=0; iv<n_vert; iv++){

        ///same for cu

        int32_t mul = 1;
        int32_t idx = 0;

        for (int32_t ic = n_coords-1; ic > -1; ic--) {

            int32_t cidx = d_coords[I2D(iv,ic,n_coords)] / d_binswidth[0];

            if(cidx < 0 || cidx >= n_bins[ic]){
                printf("index %d of coordinate %d exceeds n bins %d or below 0, coord %e\n",cidx,ic,n_bins[ic],d_coords[I2D(iv,ic,n_coords)]);
                cidx = 0; //stable, but will create bogus later
            }
            d_assigned_bin[I2D(iv,ic+1,n_coords+1)]=cidx;

            idx += cidx * mul;
            mul *= n_bins[ic];

        }

        //get row split index last
        int32_t rsidx=0;
        for(int32_t irs=1 ; irs < n_rs ; irs++){
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

void compute_bin_by_coordinates(
    const float_t * d_coords,
    const int32_t * d_rs,
    const float_t * d_binswidth,
    const int32_t * n_bins,
    int32_t * d_assigned_bin,
    int32_t * d_flat_assigned_bin,
    int32_t * d_n_per_bin,
    const int32_t n_vert,
    const int32_t n_coords,
    const int32_t n_rs,
    const int32_t n_total_bins,
    const bool calc_n_per_bin
) {
    set_defaults(d_n_per_bin, n_total_bins);
    calc(d_coords, d_rs, d_binswidth, n_bins, 
            
            d_assigned_bin, 
            d_flat_assigned_bin, 
            d_n_per_bin, 
            
            n_vert, 
            n_coords, 
            n_rs, 
            n_total_bins, 
            calc_n_per_bin);
}


void bin_by_coordinates_cpu(
    const at::Tensor& t_coords,
    const at::Tensor& t_rs,
    const at::Tensor& t_binwdith,
    const at::Tensor& t_nbins,
    at::Tensor& t_assigned_bin,
    at::Tensor& t_flat_assigned_bin,
    at::Tensor& t_nper_bin,
    bool calc_n_per_bin
) {
    CHECK_INPUT(t_coords);
    CHECK_INPUT(t_rs);
    CHECK_INPUT(t_binwdith);
    CHECK_INPUT(t_nbins);
    CHECK_INPUT(t_assigned_bin);
    CHECK_INPUT(t_flat_assigned_bin);
    CHECK_INPUT(t_nper_bin);

    auto t_coords_a = t_coords.accessor<float,2>();
    auto t_rs_a = t_rs.accessor<int,1>();
    auto t_binwdith_a = t_binwdith.accessor<float,1>();
    auto t_nbins_a = t_nbins.accessor<int,1>();
    auto t_assigned_bin_a = t_assigned_bin.accessor<int,2>();
    auto t_flat_assigned_bin_a = t_flat_assigned_bin.accessor<int,1>();
    auto t_nper_bin_a = t_nper_bin.accessor<int,1>();

    int n_vert = t_coords.size(0);
    int n_coords = t_coords.size(1);
    int n_rs = t_rs.size(0);
    int n_nbins = n_coords;
    int n_tot_bins = 0;

    computeHelper(t_nbins_a.data(), &n_tot_bins, n_nbins, n_rs);
    compute_bin_by_coordinates(t_coords_a.data(), t_rs_a.data(), t_binwdith_a.data(), t_nbins_a.data(), t_assigned_bin_a.data(), t_flat_assigned_bin_a.data(), t_nper_bin_a.data(), n_vert, n_coords, n_rs, n_tot_bins, calc_n_per_bin);
}

#ifdef CUDA_AVAILABLE
void bin_by_coordinates_cuda(
    const at::Tensor& t_coords,
    const at::Tensor& t_rs,
    const at::Tensor& t_binwdith,
    const at::Tensor& t_nbins,
    at::Tensor& t_assigned_bin,
    at::Tensor& t_flat_assigned_bin,
    at::Tensor& t_nper_bin,
    bool calc_n_per_bin
);
#endif

void bin_by_coordinates(
    const at::Tensor& t_coords,
    const at::Tensor& t_rs,
    const at::Tensor& t_binwdith,
    const at::Tensor& t_nbins,
    at::Tensor& t_assigned_bin,
    at::Tensor& t_flat_assigned_bin,
    at::Tensor& t_nper_bin,
    bool calc_n_per_bin
) {
    if (t_assigned_bin.is_cuda()) {
        #ifdef CUDA_AVAILABLE
        bin_by_coordinates_cuda(t_coords, t_rs, t_binwdith, t_nbins, t_assigned_bin, t_flat_assigned_bin, t_nper_bin, calc_n_per_bin);
        #else
        AT_ERROR("bin_by_coordinates is not available on CUDA");
        #endif
    } else {
        bin_by_coordinates_cpu(t_coords, t_rs, t_binwdith, t_nbins, t_assigned_bin, t_flat_assigned_bin, t_nper_bin, calc_n_per_bin);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bin_by_coordinates", &bin_by_coordinates, "Bin by coordinates");
    m.def("bin_by_coordinates_cpu", &bin_by_coordinates_cpu, "Bin by coordinates CPU");
}