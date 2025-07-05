#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <limits>
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


template<typename T>
static T searchLargestDistance(T i_v, float* d_dist, T n_neigh, float& maxdist){

    maxdist=0;
    T maxidx=0;
    if(n_neigh < 2)
        return maxidx;
    for(T n=1;n<n_neigh;n++){ //0 is self
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
            T idx = i_v * n_neigh + n;
            if (idx >= n_vert * n_neigh) {
                printf("ERROR: Buffer overflow in setDefaults at index %ld\n", (long)idx);
                return;
            }
            d_indices[idx] = (n == 0) ? i_v : (tf_compat ? i_v : -1);
            d_dist[idx] = 0.0f;
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
    
    
        T nfilled=1;//self-reference from defaults
        T maxidx_local=0;
        float maxdistsq=0;
    
        T total_subbins = 1;
        for(T sbi=0;sbi<n_bin_dim;sbi++)
            total_subbins *= d_n_bins[sbi];
    
        T iv_bin = d_bin_idx[i_v];
        T gbin_offset = total_subbins*(iv_bin / total_subbins);
        T sb_flat_offset = iv_bin - gbin_offset;
    
        // printf("considering vertex %d, bin %d, flat offset %d, global bin offset %d\n",i_v,iv_bin,sb_flat_offset,gbin_offset);
    
    
        binstepper<N_bin_dims, T> stepper(d_n_bins, &d_dim_bin_idx[I2D(i_v,1,n_bin_dim+1)]);
    
        bool continue_search = true;
        int64_t distance = 0;
        while(continue_search){
    
            stepper.set_d(distance);
    
            continue_search=false;
    
            while(true){
                T idx = stepper.step();
                if(idx<0){//not valid
                    if(!continue_search && !distance){//this should not happen
                        printf("\nERROR: binned_select_knn_cpu.cpp: stopping search for vtx %ld at distance %ld\n",(long)i_v,(long)distance);
                    }
                    break;
    
                }
    
                // Check for integer overflow before addition
                if (idx > (std::numeric_limits<T>::max() - gbin_offset)) {
                    printf("\nERROR: binned_select_knn_cpu.cpp: integer overflow prevented in idx calculation\n");
                    continue;
                }
                idx += gbin_offset;
    
                if(idx >= n_bboundaries-1 || idx < 0){
                    printf("\nERROR: binned_select_knn_cpu.cpp: boundary issue: idx %ld out of range, gb offset %ld, distance %ld, sb_flat_offset %ld, nbb %ld\n", (long)idx, (long)gbin_offset, (long)distance, (long)sb_flat_offset,(long)n_bboundaries);
                    continue;
                }
    
                // Bounds check before accessing array
                if (idx + 1 >= n_bboundaries) {
                    printf("\nERROR: binned_select_knn_cpu.cpp: boundary array access out of bounds\n");
                    continue;
                }
                T start_vertex = d_bin_boundaries[idx];
                T end_vertex = d_bin_boundaries[idx+1];
    
                if(start_vertex == end_vertex){ //empty bin
                    continue_search=true; //correct?
                    continue;
                }
    
                if(start_vertex>=n_vert || end_vertex>n_vert){
                    printf("\nERROR: binned_select_knn_cpu.cpp: start_vertex %ld or end_vertex %ld out of range %ld \n", (long)start_vertex, (long)end_vertex, (long)n_vert);
                    continue;//safe guard
                }
    
                for(T j_v=start_vertex;j_v<end_vertex;j_v++){
                    if(i_v == j_v)
                        continue;
    
                    // 0: can only be neighbour, 1: can only have neighbour, 2: neither
                    if(use_direction &&
                            (d_direction[j_v] == 1 || d_direction[j_v] == 2))
                        continue;
    
                    //fill up
                    float distsq = calculateDistance(i_v,j_v,d_coord,n_coords);
                    if(nfilled < n_neigh){
                        T idx_write = I2D(i_v,nfilled,n_neigh);
                        if (idx_write >= n_vert * n_neigh) {
                            printf("\nERROR: binned_select_knn_cpu.cpp: buffer overflow prevented in indices write\n");
                            continue;
                        }
                        d_indices[idx_write] = j_v;
                        d_dist[idx_write] = distsq;
                        if(distsq > maxdistsq){
                            maxdistsq = distsq;
                            maxidx_local = nfilled;
                        }
                        nfilled++;
                        continue;
                    }
                    if(distsq < maxdistsq){// automatically applies to max radius
                        //replace former max
                        T idx_replace = I2D(i_v,maxidx_local,n_neigh);
                        if (idx_replace >= n_vert * n_neigh) {
                            printf("\nERROR: binned_select_knn_cpu.cpp: buffer overflow prevented in indices replacement\n");
                            continue;
                        }
                        d_indices[idx_replace] = j_v;
                        d_dist[idx_replace] = distsq;
                        //search new max
                        maxidx_local = searchLargestDistance<T>(i_v,d_dist,n_neigh,maxdistsq);
                    }
                }
    
                continue_search=true;//at least one was valid
    
            }
            // debug: never stop unless all bins exhausted DEBUG FIXME
            if(nfilled==n_neigh && d_bin_width[0]*distance * d_bin_width[0]*distance > maxdistsq)
                break;//done
    
            // Prevent infinite loop and integer overflow
            if (distance >= std::numeric_limits<int64_t>::max() - 1) {
                printf("\nERROR: binned_select_knn_cpu.cpp: distance overflow prevented\n");
                break;
            }
            distance++;
        }
    }
}

// Function to dispatch based on input tensor types (int64 or int64)
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

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor distances = torch::empty({n_vert, K}, options_float);
    torch::Tensor indices;
    
    // Create indices tensor with same dtype as input
    if (bin_idx.scalar_type() == torch::kInt64) {
        auto options_int64 = torch::TensorOptions().dtype(torch::kInt64);
        indices = torch::empty({n_vert, K}, options_int64);
    } else {
        auto options_int32 = torch::TensorOptions().dtype(torch::kInt32);
        indices = torch::empty({n_vert, K}, options_int32);
    }

    if (bin_idx.scalar_type() == torch::kInt64) {
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
                
    } else if (bin_idx.scalar_type() == torch::kInt32) {
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
    } else {
        throw std::invalid_argument("Unsupported tensor type for bin_idx.");
    }

    return std::make_tuple(indices, distances);
}
