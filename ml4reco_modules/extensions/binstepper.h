/*
 * only use bare c++ so that it compiled with nvcc
 * assert might need to be removed/exchanged
 * template only
 * for complaints: jkiesele
 */

#include <cuda_runtime.h>
#include <cmath>

// some cuda/cpu compatibility defines
#ifdef __CUDA_ARCH__
    #define BINSTEPPER_FLOOR(x) floorf(x)
    #define BINSTEPPER_CEIL(x) ceilf(x)
#else
    #define BINSTEPPER_FLOOR(x) std::floor(x)
    #define BINSTEPPER_CEIL(x) std::ceil(x)
#endif

//now some host device statement compatibility
#ifdef __CUDACC__
    #define BINSTEPPER_HOST_DEVICE __host__ __device__
#else
    #define BINSTEPPER_HOST_DEVICE
#endif

template<int N_dims>
class binstepper{
public:

    BINSTEPPER_HOST_DEVICE
    binstepper(const int * bins_per_dim, const int* glidxs){
        total_cap_=1;
        for(int i=0;i<N_dims;i++){
            total_cap_ *= bins_per_dim[i];
            total_bins_[i] = bins_per_dim[i];//just a copy to local memory
            glidxs_[i] = glidxs[i];
        }
        set_d(0);
    }


    BINSTEPPER_HOST_DEVICE
    void set_d(int distance){
        d_ = distance;
        cube_cap_ = 2*d_+1;
        for(int i=1;i<N_dims;i++)
            cube_cap_*=2*d_+1;
        step_no_=0;
    }

    BINSTEPPER_HOST_DEVICE
    int step(){
        if(step_no_==cube_cap_)
            return -1;

        //read to inner cube
        int mul=cube_cap_;
        int cidx = step_no_;
        for(int i=0;i<N_dims;i++){
            mul /= 2*d_+1;
            cubeidxs_[i] = cidx / mul;
            cidx -= cubeidxs_[i] * mul;
        }

        step_no_++;

        //check if it is a step on surface
        bool on_surface=false;
        for(int i=0;i<N_dims;i++){
            if(abs(cubeidxs_[i]-d_)==d_)
                on_surface = true; //any abs is d
        }

        for(int i=0;i<N_dims;i++){
            if(abs(cubeidxs_[i]-d_)>d_)
                on_surface = false; //any abs is larger d

        }
        if(!on_surface)
            return step();

        //valid local step in cube: ok

        //apply to global index and check
        mul = 1;
        int glidx=0;
        for(int i = N_dims-1; i != -1; i--){//go backwards and make flat index in situ
            int iidx = cubeidxs_[i] - d_ + glidxs_[i];
            if(iidx<0 || iidx>=total_bins_[i])
                return step();
            glidx+=iidx*mul;
            mul*=total_bins_[i];
        }
        if(glidx>=total_cap_ || glidx<0)
            return step();

        return glidx;
    }

private:

    int step_no_;
    int total_cap_;
    int cube_cap_;
    int d_;
    int cubeidxs_[N_dims]; //temp
    int glidxs_[N_dims];
    int total_bins_[N_dims];
};

// To be replaced by binstepper.h
struct ccoords2flat_binstepper {
    int dims;
    const int*n_bins;
    int *low_bin_indices;
    int *high_bin_indices;
    int total_bins_to_search;
    int index;
    int flat_bin_index;
    int total_bins;

//  BINSTEPPER_HOST_DEVICE
    ccoords2flat_binstepper(const int dims) {
        this->dims = dims;
        low_bin_indices = new int[dims];
        high_bin_indices = new int[dims];
    }

    ~ccoords2flat_binstepper() {
        delete low_bin_indices;
        delete high_bin_indices;
    }
//  BINSTEPPER_HOST_DEVICE
    void set(const float*min_, const float*max_, const float bin_width, const int*n_bins) {
        total_bins_to_search=1;
        index = 0;
        this->n_bins = n_bins;
        flat_bin_index=0;
        total_bins = 1;

        for(int id=0;id<dims;id++) {
            low_bin_indices[id] = BINSTEPPER_FLOOR(min_[id] / bin_width);
            high_bin_indices[id] = BINSTEPPER_CEIL(max_[id] / bin_width);
            total_bins_to_search *= high_bin_indices[id] - low_bin_indices[id] + 1;
            total_bins*=n_bins[id];

        }
    }
//  BINSTEPPER_HOST_DEVICE
    int step() {
        while(true) {
            int offset1 = 1;
            int offset2 = 1;
            flat_bin_index = 0;
            for(int id=dims-1;id>-1;id--) {
                int dim_bin_index = low_bin_indices[id] + (index / offset2) % (
                            high_bin_indices[id] - low_bin_indices[id] + 1);
                flat_bin_index += dim_bin_index * offset1;
                offset1 *= n_bins[id];
                offset2 *= high_bin_indices[id] - low_bin_indices[id] + 1;
            }

            if(index >= total_bins_to_search)
                return -1;

            if (0 <= flat_bin_index && flat_bin_index < total_bins) {
                index += 1;
                return flat_bin_index;
            }

            index += 1;
        }
    }
};

#undef BINSTEPPER_HOST_DEVICE
#undef BINSTEPPER_FLOOR
#undef BINSTEPPER_CEIL