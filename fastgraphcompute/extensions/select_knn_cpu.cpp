#include <torch/extension.h>

// #include <string> //size_t, just for helper function
#include <cmath>

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be CPU tensor")
#define I2D(i,j,Nj) j + Nj*i


float calculateDistance(
    int64_t i_v, 
    int64_t j_v, 
    const float *d_coord, 
    int64_t n_coords) 
{
    float distsq = 0;
    if (i_v == j_v)
        return 0;
    for (int64_t i = 0; i < n_coords; i++) {
        float dist = d_coord[I2D(i_v, i, n_coords)] - d_coord[I2D(j_v, i, n_coords)];
        distsq += dist * dist;
    }
    return distsq;
}

int64_t searchLargestDistance(
    int64_t i_v, 
    float *d_dist, 
    int64_t n_neigh, 
    float& maxdist) 
{
    maxdist = 0;
    int64_t maxidx = 0;
    if (n_neigh < 2)
        return maxidx;
    for (int64_t n = 1; n < n_neigh; n++) { //0 is self
        float distsq = d_dist[I2D(i_v, n, n_neigh)];
        if (distsq > maxdist) {
            maxdist = distsq;
            maxidx = n;
        }
    }
    return maxidx;
}

void set_defaults(
    int64_t *d_indices,
    float *d_dist,
    const int64_t n_vert,
    const int64_t n_neigh)
{
    for (int64_t i_v = 0; i_v < n_vert; i_v++) {
        for (int64_t n = 0; n < n_neigh; n++)
        {
            if (n)
            {
                d_indices[I2D(i_v, n, n_neigh)] = -1;
            }
            else
            {
                d_indices[I2D(i_v, n, n_neigh)] = i_v;
            }
            d_dist[I2D(i_v, n, n_neigh)] = 0;
        }
    }
}

void select_knn_kernel(
    const float *d_coord,
    const int64_t *d_row_splits,
    const int64_t *d_mask,
    int64_t *d_indices,
    float *d_dist,

    const int64_t n_vert,
    const int64_t n_neigh,
    const int64_t n_coords,

    const int64_t j_rs,
    const float max_radius) {

    const int64_t start_vert = d_row_splits[j_rs];
    const int64_t end_vert = d_row_splits[j_rs + 1];

    for (int64_t i_v = start_vert; i_v < end_vert; i_v++) {
        if (i_v >= n_vert)
            return;//this will be a problem with actual RS, just a safety net
        
        //protection against n_vert<n_neigh
        
        int64_t nvert_in_row = end_vert - start_vert;
        int64_t max_neighbours = n_neigh;
        //set default to self
        if (nvert_in_row < n_neigh) {
            max_neighbours = nvert_in_row;
        }
        int64_t nfilled = 1;
        int64_t maxidx_local = 0;
        float maxdistsq = 0;
        for (int64_t j_v = start_vert; j_v < end_vert; j_v++) {
            if (i_v == j_v)
                continue;
            //fill up
            float distsq = calculateDistance(i_v, j_v, d_coord, n_coords);
            if (nfilled < max_neighbours && (max_radius <= 0 || max_radius >= distsq)) {
                d_indices[I2D(i_v, nfilled, n_neigh)] = j_v;
                d_dist[I2D(i_v, nfilled, n_neigh)] = distsq;
                if (distsq > maxdistsq) {
                    maxdistsq = distsq;
                    maxidx_local = nfilled;
                }
                nfilled++;
                continue;
            }
            if (distsq < maxdistsq) {// automatically applies to max radius
                //replace former max
                d_indices[I2D(i_v, maxidx_local, n_neigh)] = j_v;
                d_dist[I2D(i_v, maxidx_local, n_neigh)] = distsq;
                //search new max
                maxidx_local = searchLargestDistance(i_v, d_dist, n_neigh, maxdistsq);
            }
        }

    }

}

void compute(const float *d_coord,
    const int64_t *d_row_splits,
    const int64_t *d_mask,
    int64_t *d_indices,
    float *d_dist,

    const int64_t n_vert,
    const int64_t n_neigh,
    const int64_t n_coords,

    const int64_t n_rs,
    const double max_radius)
{
    set_defaults(d_indices,
        d_dist,
        n_vert,
        n_neigh);

    for (int64_t j_rs = 0; j_rs < n_rs - 1; j_rs++) {
        select_knn_kernel(d_coord,
            d_row_splits,
            d_mask,
            d_indices,
            d_dist,

            n_vert,
            n_neigh,
            n_coords,

            j_rs,
            max_radius);
    }
}

std::tuple<torch::Tensor, torch::Tensor>
select_knn_cpu(torch::Tensor coords, 
	       torch::Tensor row_splits,
	       torch::Tensor mask, 
	       int64_t n_neighbours, 
	       double max_radius,
	       int64_t mask_mode) {

    // CHECK_CPU(coords);
    // CHECK_CPU(row_splits);
    // CHECK_CPU(mask);

    const auto n_vert = coords.size(0);
    const auto n_coords = coords.size(1);
    const auto n_rs = row_splits.size(0);

    if (max_radius > 0) {
        max_radius *= max_radius;
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto output_dist_tensor = torch::zeros({ n_vert, n_neighbours }, options);
    auto optionsIdx = torch::TensorOptions().dtype(torch::kInt64);
    auto output_idx_tensor = torch::zeros({ n_vert, n_neighbours }, optionsIdx);

    // Input pointers to the compute function
    auto d_coords = coords.data_ptr<float>();
    auto d_row_splits = row_splits.data_ptr<int64_t>();
    auto d_mask = mask.data_ptr<int64_t>();
    auto d_output_dist = output_dist_tensor.data_ptr<float>();
    auto d_output_idx = output_idx_tensor.data_ptr<int64_t>();

    // Calling compute
    compute(d_coords, 
        d_row_splits,
        d_mask,
        d_output_idx, 
        d_output_dist,
        n_vert, 
        n_neighbours, 
        n_coords,
        n_rs, 
        max_radius);

    return std::make_tuple(output_idx_tensor, output_dist_tensor);
}