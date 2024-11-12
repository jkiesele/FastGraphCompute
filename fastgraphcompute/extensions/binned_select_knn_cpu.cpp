#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include "binstepper.h"
#include "helpers.h"

// Templated Helper to compute squared distances between two points

static float calculateDistance(size_t i_v, size_t j_v, const float * d_coord, size_t n_coords){
    float distsq=0;
    if(i_v == j_v)
        return 0;
    for(size_t i=0;i<n_coords;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist*dist;
    }
    return distsq;
}


static int searchLargestDistance(int i_v, float* d_dist, int n_neigh, float& maxdist){

    maxdist=0;
    int maxidx=0;
    if(n_neigh < 2)
        return maxidx;
    for(size_t n=1;n<n_neigh;n++){ //0 is self
        float distsq = d_dist[I2D(i_v,n,n_neigh)];
        if(distsq > maxdist){
            maxdist = distsq;
            maxidx = n;
        }
    }
    return maxidx;
}

// Templated setDefaults function
template <typename T>
static void setDefaults(T* d_indices, float* d_dist, bool tf_compat, T n_vert, T n_neigh) {
    for (T i_v = 0; i_v < n_vert; i_v++) {
        for (T n = 0; n < n_neigh; n++) {
            d_indices[i_v * n_neigh + n] = (n == 0) ? i_v : (tf_compat ? i_v : -1);
            d_dist[i_v * n_neigh + n] = 0.0f;
        }
    }
}

// Main templated computation function (CPU version)
template<int N_bin_dims, typename T>
static void select_knn_kernel_cpu(
    const float* d_coord,
    const T* d_bin_idx,
    const T* d_direction,
    const T* d_dim_bin_idx,
    const T* d_bin_boundaries,
    const T* d_n_bins,
    const float* d_bin_width,
    T* d_indices,
    float* d_dist,
    T n_vert,
    T n_neigh,
    T n_coords,
    T n_bin_dim,
    T n_bboundaries,
    bool use_direction
) {

    T n_bins_total=1;
    for (T b=0;b < n_bin_dim; b++)
        n_bins_total = n_bins_total * d_n_bins[b];

    for (T i_v = 0; i_v < n_vert; i_v++) {
        // 0: can only be neighbour, 1: can only have neighbour, 2: neither
        if(use_direction &&
                (d_direction[i_v] == 0 || d_direction[i_v] == 2))
            continue;
    
        //continue;//do nothing
    
    
        size_t nfilled=1;//self-reference from defaults
        size_t maxidx_local=0;
        float maxdistsq=0;
    
        int total_subbins = 1;
        for(int sbi=0;sbi<n_bin_dim;sbi++)
            total_subbins *= d_n_bins[sbi];
    
        int iv_bin = d_bin_idx[i_v];
        int gbin_offset = total_subbins*(iv_bin / total_subbins);
        int sb_flat_offset = iv_bin - gbin_offset;
    
        // printf("considering vertex %d, bin %d, flat offset %d, global bin offset %d\n",i_v,iv_bin,sb_flat_offset,gbin_offset);
    
    
        binstepper<N_bin_dims, T> stepper(d_n_bins, &d_dim_bin_idx[I2D(i_v,1,n_bin_dim+1)]);
    
        bool continue_search = true;
        int distance = 0;
        while(continue_search){
    
            stepper.set_d(distance);
    
            continue_search=false;
    
            while(true){
                int idx = stepper.step();
                if(idx<0){//not valid
                    if(!continue_search && !distance){//this should not happen
                        printf("\nERROR: binned_select_knn.cu: stopping search for vtx %d at distance %d\n",i_v,distance);
                    }
                    break;
    
                }
    
                idx+=gbin_offset;
    
                if(idx>=n_bboundaries-1){
                    printf("\nERROR: binned_select_knn.cu: boundary issue: idx %d out of range, gb offset %d, distance %d, sb_flat_offset %d, nbb %d\n", idx, gbin_offset, distance, sb_flat_offset,n_bboundaries);
                    continue;
                }
    
                int start_vertex = d_bin_boundaries[idx];
                int end_vertex = d_bin_boundaries[idx+1];
    
                if(start_vertex == end_vertex){ //empty bin
                    continue_search=true; //correct?
                    continue;
                }
    
                if(start_vertex>=n_vert || end_vertex>n_vert){
                    printf("\nERROR: binned_select_knn.cu: start_vertex %d or end_vertex %d out of range %d \n", start_vertex, end_vertex, n_vert);
                    continue;//safe guard
                }
    
                for(size_t j_v=start_vertex;j_v<end_vertex;j_v++){
                    if(i_v == j_v)
                        continue;
    
                    // 0: can only be neighbour, 1: can only have neighbour, 2: neither
                    if(use_direction &&
                            (d_direction[j_v] == 1 || d_direction[j_v] == 2))
                        continue;
    
                    //fill up
                    float distsq = calculateDistance(i_v,j_v,d_coord,n_coords);
                    if(nfilled< n_neigh){
                        d_indices[I2D(i_v,nfilled,n_neigh)] = j_v;
                        d_dist[I2D(i_v,nfilled,n_neigh)] = distsq;
                        if(distsq > maxdistsq){
                            maxdistsq = distsq;
                            maxidx_local = nfilled;
                        }
                        nfilled++;
                        continue;
                    }
                    if(distsq < maxdistsq){// automatically applies to max radius
                        //replace former max
                        d_indices[I2D(i_v,maxidx_local,n_neigh)] = j_v;
                        d_dist[I2D(i_v,maxidx_local,n_neigh)] = distsq;
                        //search new max
                        maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
                    }
                }
    
                continue_search=true;//at least one was valid
    
            }
            // debug: never stop unless all bins exhausted DEBUG FIXME
            if(nfilled==n_neigh && d_bin_width[0]*distance * d_bin_width[0]*distance > maxdistsq)
                break;//done
    
            distance++;
        }
    }
}

// Function to dispatch based on input tensor types (int32 or int64)
std::tuple<torch::Tensor, torch::Tensor> binned_select_knn_cpu(
    torch::Tensor coordinates,
    torch::Tensor bin_idx,
    torch::Tensor dim_bin_idx,
    torch::Tensor bin_boundaries,
    torch::Tensor n_bins,
    torch::Tensor bin_width,
    torch::Tensor direction,
    bool tf_compat,
    bool use_direction,
    int64_t K
) {
    const auto n_vert = coordinates.size(0);
    const auto n_coords = coordinates.size(1);
    const auto n_bboundaries = bin_boundaries.size(0);
    const auto n_bin_dims = n_bins.size(0);

    auto options_int = torch::TensorOptions().dtype(torch::kInt32);
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32);

    torch::Tensor indices = torch::empty({n_vert, K}, options_int);
    torch::Tensor distances = torch::empty({n_vert, K}, options_float);

    if (bin_idx.scalar_type() == torch::kInt32) {
        // For int32 inputs
        setDefaults<int32_t>(indices.data_ptr<int32_t>(), distances.data_ptr<float>(), tf_compat, n_vert, K);


        if (n_bin_dims == 2)
            select_knn_kernel_cpu<2, int32_t>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int32_t>(),
                direction.data_ptr<int32_t>(), dim_bin_idx.data_ptr<int32_t>(),
                bin_boundaries.data_ptr<int32_t>(), n_bins.data_ptr<int32_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);
        
        else if (n_bin_dims == 3)
            select_knn_kernel_cpu<3, int32_t>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int32_t>(),
                direction.data_ptr<int32_t>(), dim_bin_idx.data_ptr<int32_t>(),
                bin_boundaries.data_ptr<int32_t>(), n_bins.data_ptr<int32_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);

        else if (n_bin_dims == 4)
            select_knn_kernel_cpu<4, int32_t>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int32_t>(),
                direction.data_ptr<int32_t>(), dim_bin_idx.data_ptr<int32_t>(),
                bin_boundaries.data_ptr<int32_t>(), n_bins.data_ptr<int32_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);

        else if (n_bin_dims == 5)
            select_knn_kernel_cpu<5, int32_t>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int32_t>(),
                direction.data_ptr<int32_t>(), dim_bin_idx.data_ptr<int32_t>(),
                bin_boundaries.data_ptr<int32_t>(), n_bins.data_ptr<int32_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);

        else{
            throw std::invalid_argument("Unsupported number of binning dimensions.");
        }
                
    } else if (bin_idx.scalar_type() == torch::kInt64) {
        // For int64 inputs
        setDefaults<int64_t>(indices.data_ptr<int64_t>(), distances.data_ptr<float>(), tf_compat, n_vert, K);

        if (n_bin_dims == 2)
            select_knn_kernel_cpu<2, int64_t>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int64_t>(),
                direction.data_ptr<int64_t>(), dim_bin_idx.data_ptr<int64_t>(),
                bin_boundaries.data_ptr<int64_t>(), n_bins.data_ptr<int64_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int64_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);
       
        else if (n_bin_dims == 3)
            select_knn_kernel_cpu<3, int64_t>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int64_t>(),
                direction.data_ptr<int64_t>(), dim_bin_idx.data_ptr<int64_t>(),
                bin_boundaries.data_ptr<int64_t>(), n_bins.data_ptr<int64_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int64_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);

        else if (n_bin_dims == 4)
            select_knn_kernel_cpu<4, int64_t>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int64_t>(),
                direction.data_ptr<int64_t>(), dim_bin_idx.data_ptr<int64_t>(),
                bin_boundaries.data_ptr<int64_t>(), n_bins.data_ptr<int64_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int64_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);

        else if (n_bin_dims == 5)
            select_knn_kernel_cpu<5, int64_t>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int64_t>(),
                direction.data_ptr<int64_t>(), dim_bin_idx.data_ptr<int64_t>(),
                bin_boundaries.data_ptr<int64_t>(), n_bins.data_ptr<int64_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int64_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);

        else{
            throw std::invalid_argument("Unsupported number of binning dimensions.");
        }
    } else {
        throw std::invalid_argument("Unsupported tensor type for bin_idx.");
    }

    return std::make_tuple(indices, distances);
}

TORCH_LIBRARY(binned_select_knn_cpu, m) {
    m.def("binned_select_knn_cpu", &binned_select_knn_cpu);
}
