
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "binstepper.h"
#include "cuda_helpers.h"
#include "helpers.h"
#include <vector>


__device__
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


__device__
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


__global__
static void setDefaults(
        int *d_indices,
        float *d_dist,
        const bool tf_compat,
        const int n_vert,
        const int n_neigh
){
    const size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;
    const size_t n =  blockIdx.y * blockDim.y + threadIdx.y;
    if(n >= n_neigh)
        return;

    if(n){
        if(tf_compat)
            d_indices[I2D(i_v,n,n_neigh)] = i_v;
        else
            d_indices[I2D(i_v,n,n_neigh)] = -1;
    }
    else{
        d_indices[I2D(i_v,n,n_neigh)] = i_v;
    }
    d_dist[I2D(i_v,n,n_neigh)] = 0;


}



template<int N_bin_dims, typename T>
__global__
static void select_knn_kernel(

        const float * d_coord,
        const T * d_bin_idx,
        const T * d_direction,
        const T * d_dim_bin_idx,

        const T * d_bin_boundaries,
        const T * d_n_bins,

        const float* d_bin_width,

        int *d_indices,
        float *d_dist,

        const int n_vert,
        const int n_neigh,
        const int n_coords,
        const int n_bin_dim,

        const int n_bboundaries,
        bool use_direction) {

    //bin boundaries [i] [i+1] describe the scan ranges


    //really no buffering at all here


    int i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v>=n_vert)
        return;//safe guard

    // 0: can only be neighbour, 1: can only have neighbour, 2: neither
    if(use_direction &&
            (d_direction[i_v] == 0 || d_direction[i_v] == 2))
        return;

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


// Function to dispatch based on input tensor types (int32 or int64)
std::tuple<torch::Tensor, torch::Tensor> binned_select_knn_cuda_fn(
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

    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(coordinates.device());
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(coordinates.device());

    torch::Tensor indices = torch::empty({n_vert, K}, options_int);
    torch::Tensor distances = torch::empty({n_vert, K}, options_float);

    grid_and_block gb_set_def(n_vert,256,K,4);
    grid_and_block gb(n_vert,512);

    setDefaults<<<gb_set_def.grid(),gb_set_def.block()>>>(indices.data_ptr<int32_t>(), distances.data_ptr<float>(), tf_compat, n_vert, K);

    if (bin_idx.scalar_type() == torch::kInt32) {

        if (n_bin_dims == 2)
            select_knn_kernel<2, int32_t><<<gb.grid(),gb.block()>>>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int32_t>(),
                direction.data_ptr<int32_t>(), dim_bin_idx.data_ptr<int32_t>(),
                bin_boundaries.data_ptr<int32_t>(), n_bins.data_ptr<int32_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);
        
        else if (n_bin_dims == 3)
            select_knn_kernel<3, int32_t><<<gb.grid(),gb.block()>>>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int32_t>(),
                direction.data_ptr<int32_t>(), dim_bin_idx.data_ptr<int32_t>(),
                bin_boundaries.data_ptr<int32_t>(), n_bins.data_ptr<int32_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);

        else if (n_bin_dims == 4)
            select_knn_kernel<4, int32_t><<<gb.grid(),gb.block()>>>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int32_t>(),
                direction.data_ptr<int32_t>(), dim_bin_idx.data_ptr<int32_t>(),
                bin_boundaries.data_ptr<int32_t>(), n_bins.data_ptr<int32_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);

        else if (n_bin_dims == 5)
            select_knn_kernel<5, int32_t><<<gb.grid(),gb.block()>>>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int32_t>(),
                direction.data_ptr<int32_t>(), dim_bin_idx.data_ptr<int32_t>(),
                bin_boundaries.data_ptr<int32_t>(), n_bins.data_ptr<int32_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);

        else{
            throw std::invalid_argument("Unsupported number of binning dimensions.");
        }
                
    } else if (bin_idx.scalar_type() == torch::kInt64) {

        if (n_bin_dims == 2)
            select_knn_kernel<2, int64_t><<<gb.grid(),gb.block()>>>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int64_t>(),
                direction.data_ptr<int64_t>(), dim_bin_idx.data_ptr<int64_t>(),
                bin_boundaries.data_ptr<int64_t>(), n_bins.data_ptr<int64_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);
       
        else if (n_bin_dims == 3)
            select_knn_kernel<3, int64_t><<<gb.grid(),gb.block()>>>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int64_t>(),
                direction.data_ptr<int64_t>(), dim_bin_idx.data_ptr<int64_t>(),
                bin_boundaries.data_ptr<int64_t>(), n_bins.data_ptr<int64_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);

        else if (n_bin_dims == 4)
            select_knn_kernel<4, int64_t><<<gb.grid(),gb.block()>>>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int64_t>(),
                direction.data_ptr<int64_t>(), dim_bin_idx.data_ptr<int64_t>(),
                bin_boundaries.data_ptr<int64_t>(), n_bins.data_ptr<int64_t>(),
                bin_width.data_ptr<float>(), indices.data_ptr<int32_t>(),
                distances.data_ptr<float>(), n_vert, K, n_coords, n_bin_dims, n_bboundaries, use_direction);

        else if (n_bin_dims == 5)
            select_knn_kernel<5, int64_t><<<gb.grid(),gb.block()>>>(
                coordinates.data_ptr<float>(), bin_idx.data_ptr<int64_t>(),
                direction.data_ptr<int64_t>(), dim_bin_idx.data_ptr<int64_t>(),
                bin_boundaries.data_ptr<int64_t>(), n_bins.data_ptr<int64_t>(),
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