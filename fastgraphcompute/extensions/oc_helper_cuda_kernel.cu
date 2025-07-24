/*

Credit to:
fantastic-fabian, jkiesele
Complaints to:
jkiesele
*/

#include <torch/extension.h>
#include "helpers.h"
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_helpers.h"

template <typename T>
__global__
static void calc_m(
            T * asso_idx, 
            T * unique_idx, 
            T * unique_rs_asso, 
            T * rs,
            
            T * M,
            T * M_not,

            int64_t n_vert,
            int64_t n_unique,
            int64_t n_maxuq,
            int64_t n_maxrs,
            bool calc_m_not) {

        int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
        int64_t tid = threadIdx.x; //for coallesced memory access
        if(k >= n_unique)
            return;

        int64_t uqidx = unique_idx[k];
        int64_t uqidx_i_rs = unique_rs_asso[k]; // which row split it belongs to
        int64_t start_vertex = rs[uqidx_i_rs];
        int64_t end_vertex = rs[uqidx_i_rs+1];
        if(end_vertex > n_vert){
            printf("Error: end_vertex %lld is larger than n_vert %lld, setting end_vertex to n_vert . Check inputs!\n", end_vertex, n_vert);
            end_vertex = n_vert;
        }
        //sanity check: end_vertex - start_vertex should be smaller or equal to n_maxrs
        if(end_vertex - start_vertex > n_maxrs){
            printf("Error: end_vertex - start_vertex %lld is larger than n_maxrs %lld, setting end_vertex to start_vertex + n_maxrs . Check inputs!\n", end_vertex - start_vertex, n_maxrs);
            end_vertex = start_vertex + n_maxrs;
        }
        //synch threads here, now everything is same for all threads
        __syncthreads();
        // Fill M
        int64_t fill_counter = 0;
        for(int64_t i_v = start_vertex + tid; i_v < end_vertex; i_v++ ){
            if(asso_idx[i_v] == uqidx){
                M[I2D(fill_counter, k, n_unique)] = i_v;
                fill_counter++;
                if(fill_counter > n_maxuq){
                    printf("Error: fill_counter %lld is larger than n_maxuq in first M loop %lld . Check inputs!\n", fill_counter, n_maxuq);
                    break;
                }
            }
        }
        //fill rest, might diverge but that's ok
        for(int64_t i_v = start_vertex; i_v < start_vertex + tid; i_v++ ){
            if(i_v < end_vertex && asso_idx[i_v] == uqidx){
                M[I2D(fill_counter, k, n_unique)] = i_v;
                fill_counter++;
                if(fill_counter > n_maxuq){
                    printf("Error: fill_counter %lld is larger than n_maxuq in second M loop %lld . Check inputs!\n", fill_counter, n_maxuq);
                    break;
                }
            }
        }

        // Pad with -1s
        for(; fill_counter < n_maxuq; fill_counter++){
            M[I2D(fill_counter, k, n_unique)] = -1;
        }
        // Fill M_not
        if(calc_m_not){
            //synch threads here, now everything is same for all threads
            __syncthreads();
            fill_counter = 0;
            for(int64_t i_v = start_vertex + tid; i_v < end_vertex; i_v++ ){
                if (asso_idx[i_v] != uqidx){
                    M_not[I2D(fill_counter, k, n_unique)] = i_v;
                    fill_counter++;
                    if(fill_counter > n_maxrs){
                        printf("Error: fill_counter %lld is larger than n_maxrs in first M_not loop %lld . Check inputs!\n", fill_counter, n_maxuq);
                        break;
                    }
                }
            }
            //fill rest, might diverge but that's ok
            for(int64_t i_v = start_vertex; i_v < start_vertex + tid; i_v++ ){
                if (i_v < end_vertex && asso_idx[i_v] != uqidx){
                    M_not[I2D(fill_counter, k, n_unique)] = i_v;
                    fill_counter++;
                    if(fill_counter > n_maxrs){
                        printf("Error: fill_counter %lld is larger than n_maxrs in first M_not loop %lld . Check inputs!\n", fill_counter, n_maxuq);
                        break;
                    }
                }
            }
            for(; fill_counter < n_maxrs; fill_counter++){
                M_not[I2D(fill_counter, k, n_unique)] = -1;
            }
        }
        __syncthreads(); //make sure all end at the same time
}

static void check_all_inputs(
    torch::Tensor asso_idx,
    torch::Tensor unique_idx,
    torch::Tensor unique_rs_asso,
    torch::Tensor rs,
    torch::Tensor max_n_unique_over_splits,
    torch::Tensor max_n_in_splits) {

    TORCH_CHECK(max_n_unique_over_splits.size(0) == 1, "max_n_unique_over_splits should have size 1");
    
    TORCH_CHECK(max_n_in_splits.size(0) == 1, "max_n_in_splits should have size 1");
    
    TORCH_CHECK(asso_idx.is_contiguous() && asso_idx.dtype() == torch::kInt64, 
        "asso_idx should be contiguous and of type int64");
    
    TORCH_CHECK(unique_idx.is_contiguous() && unique_idx.dtype() == torch::kInt64, 
        "unique_idx should be contiguous and of type int64");
    
    TORCH_CHECK(unique_rs_asso.is_contiguous() && unique_rs_asso.dtype() == torch::kInt64, 
        "unique_rs_asso should be contiguous and of type int64");
    
    TORCH_CHECK(rs.is_contiguous() && rs.dtype() == torch::kInt64, 
        "rs should be contiguous and of type int64");
    
    TORCH_CHECK(max_n_unique_over_splits.is_contiguous() && max_n_unique_over_splits.dtype() == torch::kInt64,
        "max_n_unique_over_splits should be contiguous and of type int64");
    
    TORCH_CHECK(max_n_in_splits.is_contiguous() && max_n_in_splits.dtype() == torch::kInt64, 
        "max_n_in_splits should be contiguous and of type int64");
    
    TORCH_CHECK(asso_idx.device() == unique_idx.device() && asso_idx.device() == unique_rs_asso.device() && 
        asso_idx.device() == rs.device() && asso_idx.device() == max_n_unique_over_splits.device() && 
        asso_idx.device() == max_n_in_splits.device(), "All inputs should be on the same device");
}

std::tuple<torch::Tensor, torch::Tensor> oc_helper_cuda_fn(
    torch::Tensor asso_idx,
    torch::Tensor unique_idx,
    torch::Tensor unique_rs_asso,
    torch::Tensor rs,
    torch::Tensor max_n_unique_over_splits,
    torch::Tensor max_n_in_splits,
    bool calc_m_not) {

    check_all_inputs(asso_idx, unique_idx, unique_rs_asso, rs, max_n_unique_over_splits, max_n_in_splits);

    const auto n_vert = asso_idx.size(0);
    const auto n_unique = unique_idx.size(0);

    auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(asso_idx.device());

    //move them to cpu first before accessing the elements
    auto n_maxuq = max_n_unique_over_splits.cpu().data_ptr<int64_t>()[0];
    auto n_maxrs = max_n_in_splits.cpu().data_ptr<int64_t>()[0];

    torch::Tensor M_transposed = torch::empty({n_maxuq, n_unique}, options_int);
    torch::Tensor M_not_transposed = torch::empty({n_maxrs, n_unique}, options_int);

    grid_and_block gb(n_unique, 512);

    calc_m<int64_t><<<gb.grid(), gb.block()>>>(
        asso_idx.data_ptr<int64_t>(),
        unique_idx.data_ptr<int64_t>(),
        unique_rs_asso.data_ptr<int64_t>(),
        rs.data_ptr<int64_t>(),
        M_transposed.data_ptr<int64_t>(),
        M_not_transposed.data_ptr<int64_t>(),
        n_vert,
        n_unique,
        n_maxuq,
        n_maxrs,
        calc_m_not
    );

    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA error in kernel execution");

    torch::Tensor M = M_transposed.transpose(0, 1).contiguous();//ensure contiguous
    torch::Tensor M_not = M_not_transposed.transpose(0, 1).contiguous();

    return std::make_tuple(M, M_not);
}