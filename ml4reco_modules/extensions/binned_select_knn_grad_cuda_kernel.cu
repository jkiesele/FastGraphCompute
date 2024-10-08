
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helpers.h"
#include "cuda_helpers.h"


template <typename T>
__global__
static void select_knn_grad_selfloop_kernel(
        const float *d_grad_dist, // V x N
        const T *d_neigh_indices,
        const float *d_dist,
        const float *d_coord,

        float * d_grad_coord,

        const T n_vert,
        const T n_neigh,
        const T n_coords) {


      size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;

    size_t nu_c= blockIdx.y * blockDim.y + threadIdx.y;
    if(nu_c >= n_coords)
        return;

    const float xinu = d_coord[I2D(i_v,nu_c,n_coords)];

    float self_contrib=0;
    for(T i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        T k = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        if(k<0 || k>= n_vert){
            if( k>= n_vert)
                printf("select_knn_grad_kernel: k out of range\n");
            continue;
        }

        const float gik = d_grad_dist[I2D(i_v,i_i_n,n_neigh)];
        const float xknu = d_coord[I2D(k,nu_c,n_coords)];


        self_contrib -= 2. * gik * (xknu - xinu);

    }
    d_grad_coord[I2D(i_v,nu_c,n_coords)] = self_contrib;

}


template <typename T>
__global__
static void select_knn_grad_neighloop_kernel(
        const float *d_grad_dist, // V x N
        const T *d_neigh_indices,
        const float *d_dist,
        const float *d_coord,

        float * d_grad_coord,

        const T n_vert,
        const T n_neigh,
        const T n_coords){

    size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;

    size_t nu_c= blockIdx.y * blockDim.y + threadIdx.y;
    if(nu_c >= n_coords)
        return;

    const float xinu = d_coord[I2D(i_v,nu_c,n_coords)];

    for(T i_i_n = 0; i_i_n < n_neigh; i_i_n++){

        T m = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        if(m<0 || m>= n_vert){
            if(m>= n_vert)
                printf("select_knn_grad_kernel: m out of range\n");
            continue;
        }

        const float gim = d_grad_dist[I2D(i_v,i_i_n,n_neigh)];
        const float xmnu = d_coord[I2D(m,nu_c,n_coords)];

        float add = 2. * gim * (xmnu - xinu);
        atomicAdd( &d_grad_coord[I2D(m, nu_c, n_coords)], add);

    }
}


torch::Tensor binned_select_knn_grad_cuda(
    torch::Tensor grad_distances,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor coordinates
) {
    const auto n_vert = coordinates.size(0);
    const auto n_coords = coordinates.size(1);
    const auto K = indices.size(1);

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(coordinates.device());
    torch::Tensor grad_coords = torch::empty({n_vert, n_coords}, options_float);

    grid_and_block gb(n_vert,256,n_coords,4);

    if (indices.scalar_type() == torch::kInt32) {
        select_knn_grad_selfloop_kernel<int32_t><<<gb.grid(),gb.block()>>>(
            grad_distances.data_ptr<float>(),
            indices.data_ptr<int32_t>(),
            distances.data_ptr<float>(),
            coordinates.data_ptr<float>(),
            grad_coords.data_ptr<float>(),
            n_vert,
            K,
            n_coords
        );
    
        select_knn_grad_neighloop_kernel<int32_t><<<gb.grid(),gb.block()>>>(
            grad_distances.data_ptr<float>(),
            indices.data_ptr<int32_t>(),
            distances.data_ptr<float>(),
            coordinates.data_ptr<float>(),
            grad_coords.data_ptr<float>(),
            n_vert,
            K,
            n_coords
        );
    }
    else if (indices.scalar_type() == torch::kInt64) {
            select_knn_grad_selfloop_kernel<int64_t><<<gb.grid(),gb.block()>>>(
            grad_distances.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            distances.data_ptr<float>(),
            coordinates.data_ptr<float>(),
            grad_coords.data_ptr<float>(),
            n_vert,
            K,
            n_coords
        );
    
        select_knn_grad_neighloop_kernel<int64_t><<<gb.grid(),gb.block()>>>(
            grad_distances.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            distances.data_ptr<float>(),
            coordinates.data_ptr<float>(),
            grad_coords.data_ptr<float>(),
            n_vert,
            K,
            n_coords
        );
    }
    else {
        throw std::invalid_argument("Unsupported tensor type for bin_idx.");
    }

    return grad_coords;
}