#include <torch/extension.h>
#include "helpers.h"
#include <vector>
#include <algorithm>

template <typename T>
static void select_knn_grad_selfloop_kernel(
        const float *d_grad_dist, // V x N
        const T *d_neigh_indices,
        const float *d_dist,
        const float *d_coord,

        float * d_grad_coord,

        const T n_vert,
        const T n_neigh,
        const T n_coords) {

    for (T i_v = 0; i_v < n_vert; i_v++){

        for (T nu_c = 0; nu_c < n_coords; nu_c++){
    
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
        }//nu_c
    }//i_v 
}


template <typename T>
static void select_knn_grad_neighloop_kernel(
        const float *d_grad_dist, // V x N
        const T *d_neigh_indices,
        const float *d_dist,
        const float *d_coord,

        float * d_grad_coord,

        const T n_vert,
        const T n_neigh,
        const T n_coords){

    for (T i_v = 0; i_v < n_vert; i_v++){

        for (T nu_c = 0; nu_c < n_coords; nu_c++){

            const float xinu = d_coord[I2D(i_v, nu_c, n_coords)];

            for (T i_i_n = 0; i_i_n < n_neigh; i_i_n++){

                T m = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
                if (m < 0 || m >= n_vert)
                {
                    if (m >= n_vert)
                        printf("select_knn_grad_kernel: m out of range\n");
                    continue;
                }

                const float gim = d_grad_dist[I2D(i_v, i_i_n, n_neigh)];
                const float xmnu = d_coord[I2D(m, nu_c, n_coords)];

                float add = 2. * gim * (xmnu - xinu);
                d_grad_coord[I2D(m, nu_c, n_coords)] += add;
            } // i_i_n
        } // nu_c
    } // i_v
}


torch::Tensor binned_select_knn_grad_cpu(
    torch::Tensor grad_distances,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor coordinates
) {
    const auto n_vert = coordinates.size(0);
    const auto n_coords = coordinates.size(1);
    const auto K = indices.size(1);

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor grad_coords = torch::zeros({n_vert, n_coords}, options_float);

    //make sure coordinates, distances, and grad_distances are contiguous and float32
    grad_distances = grad_distances.contiguous();
    indices = indices.contiguous();
    distances = distances.contiguous();
    coordinates = coordinates.contiguous();

    if (indices.scalar_type() == torch::kInt32) {
        select_knn_grad_selfloop_kernel<int32_t>(
            grad_distances.data_ptr<float>(),
            indices.data_ptr<int32_t>(),
            distances.data_ptr<float>(),
            coordinates.data_ptr<float>(),
            grad_coords.data_ptr<float>(),
            n_vert,
            K,
            n_coords
        );
    
        select_knn_grad_neighloop_kernel<int32_t>(
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
            select_knn_grad_selfloop_kernel<int64_t>(
            grad_distances.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            distances.data_ptr<float>(),
            coordinates.data_ptr<float>(),
            grad_coords.data_ptr<float>(),
            n_vert,
            K,
            n_coords
        );
    
        select_knn_grad_neighloop_kernel<int64_t>(
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