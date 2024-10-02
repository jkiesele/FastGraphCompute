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

template<int N_dims, typename T>
class binstepper {
public:
    BINSTEPPER_HOST_DEVICE
    binstepper(const T* bins_per_dim, const T* glidxs) {
        total_cap_ = 1;
        for (T i = 0; i < N_dims; i++) {
            total_cap_ *= bins_per_dim[i];
            total_bins_[i] = bins_per_dim[i];  // just a copy to local memory
            glidxs_[i] = glidxs[i];
        }
        set_d(0);
    }

    BINSTEPPER_HOST_DEVICE
    void set_d(T distance) {
        d_ = distance;
        cube_cap_ = 2 * d_ + 1;
        for (T i = 1; i < N_dims; i++)
            cube_cap_ *= 2 * d_ + 1;
        step_no_ = 0;
    }

    BINSTEPPER_HOST_DEVICE
    T step() {
        if (step_no_ == cube_cap_)
            return -1;

        // read to inner cube
        T mul = cube_cap_;
        T cidx = step_no_;
        for (T i = 0; i < N_dims; i++) {
            mul /= 2 * d_ + 1;
            cubeidxs_[i] = cidx / mul;
            cidx -= cubeidxs_[i] * mul;
        }

        step_no_++;

        // check if it is a step on surface
        bool on_surface = false;
        for (T i = 0; i < N_dims; i++) {
            if (abs(cubeidxs_[i] - d_) == d_)
                on_surface = true;  // any abs is d
        }

        for (T i = 0; i < N_dims; i++) {
            if (abs(cubeidxs_[i] - d_) > d_)
                on_surface = false;  // any abs is larger than d
        }
        if (!on_surface)
            return step();

        // valid local step in cube: ok

        // apply to global index and check
        mul = 1;
        T glidx = 0;
        for (T i = N_dims - 1; i >= 0; i--) {  // go backwards and make flat index in situ
            T iidx = cubeidxs_[i] - d_ + glidxs_[i];
            if (iidx < 0 || iidx >= total_bins_[i])
                return step();
            glidx += iidx * mul;
            mul *= total_bins_[i];
        }
        if (glidx >= total_cap_ || glidx < 0)
            return step();

        return glidx;
    }

private:
    T step_no_;
    T total_cap_;
    T cube_cap_;
    T d_;
    T cubeidxs_[N_dims];  // temp
    T glidxs_[N_dims];
    T total_bins_[N_dims];
};

// To be replaced by binstepper.h
template <typename T>
struct ccoords2flat_binstepper {
    T dims;
    const T* n_bins;
    T* low_bin_indices;
    T* high_bin_indices;
    T total_bins_to_search;
    T index;
    T flat_bin_index;
    T total_bins;

    // BINSTEPPER_HOST_DEVICE
    ccoords2flat_binstepper(const T dims) {
        this->dims = dims;
        low_bin_indices = new T[dims];
        high_bin_indices = new T[dims];
    }

    ~ccoords2flat_binstepper() {
        delete[] low_bin_indices;
        delete[] high_bin_indices;
    }

    // BINSTEPPER_HOST_DEVICE
    void set(const float* min_, const float* max_, const float bin_width, const T* n_bins) {
        total_bins_to_search = 1;
        index = 0;
        this->n_bins = n_bins;
        flat_bin_index = 0;
        total_bins = 1;

        for (T id = 0; id < dims; id++) {
            low_bin_indices[id] = BINSTEPPER_FLOOR(min_[id] / bin_width);
            high_bin_indices[id] = BINSTEPPER_CEIL(max_[id] / bin_width);
            total_bins_to_search *= high_bin_indices[id] - low_bin_indices[id] + 1;
            total_bins *= n_bins[id];
        }
    }

    // BINSTEPPER_HOST_DEVICE
    T step() {
        while (true) {
            T offset1 = 1;
            T offset2 = 1;
            flat_bin_index = 0;
            for (T id = dims - 1; id >= 0; id--) {
                T dim_bin_index = low_bin_indices[id] + (index / offset2) % (high_bin_indices[id] - low_bin_indices[id] + 1);
                flat_bin_index += dim_bin_index * offset1;
                offset1 *= n_bins[id];
                offset2 *= high_bin_indices[id] - low_bin_indices[id] + 1;
            }

            if (index >= total_bins_to_search)
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
