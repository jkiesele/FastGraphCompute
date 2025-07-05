#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helpers.h"
#include "cuda_helpers.h"
#include <c10/macros/Macros.h>

#define C10_CUDA_KERNEL_LAUNCH_CHECK() {                         \
    cudaError_t err = cudaGetLastError();                        \
    if (err != cudaSuccess) {                                    \
        printf("CUDA Kernel launch error: %s\n",                 \
               cudaGetErrorString(err));                         \
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err)); \
    }                                                            \
}

#ifndef CHECK_INPUT
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

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
    int64_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;
    int64_t nu_c= blockIdx.y * blockDim.y + threadIdx.y;
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
    int64_t i_v =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i_v >= n_vert)
        return;
    int64_t nu_c= blockIdx.y * blockDim.y + threadIdx.y;
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

torch::Tensor binned_select_knn_grad_cuda_fn(
    torch::Tensor grad_distances,
    torch::Tensor indices,
    torch::Tensor distances,
    torch::Tensor coordinates
) {
    CHECK_INPUT(grad_distances);
    CHECK_INPUT(indices);
    CHECK_INPUT(distances);
    CHECK_INPUT(coordinates);
    const auto n_vert = coordinates.size(0);
    const auto n_coords = coordinates.size(1);
    const auto K = indices.size(1);
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(coordinates.device());
    torch::Tensor grad_coords = torch::empty({n_vert, n_coords}, options_float);
    grid_and_block gb(n_vert,256,n_coords,4);
    if (indices.scalar_type() == torch::kInt64) {
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cudaDeviceSynchronize();
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cudaDeviceSynchronize();
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (indices.scalar_type() == torch::kInt64) {
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cudaDeviceSynchronize();
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cudaDeviceSynchronize();
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return grad_coords;
}
