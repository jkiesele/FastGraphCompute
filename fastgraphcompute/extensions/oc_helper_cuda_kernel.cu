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

            int n_vert,
            int n_unique,
            int n_maxuq,
            int n_maxrs,
            bool calc_m_not) {

        int k = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x; //for coallesced memory access
        if(k >= n_unique)
            return;

        int uqidx = unique_idx[k];
        int uqidx_i_rs = unique_rs_asso[k]; // which row split it belongs to
        int start_vertex = rs[uqidx_i_rs];
        int end_vertex = rs[uqidx_i_rs+1];
        if(end_vertex > n_vert){
            printf("Error: end_vertex %d is larger than n_vert %d, setting end_vertex to n_vert . Check inputs!\n", end_vertex, n_vert);
            end_vertex = n_vert;
        }
        //sanity check: end_vertex - start_vertex should be smaller or equal to n_maxrs
        if(end_vertex - start_vertex > n_maxrs){
            printf("Error: end_vertex - start_vertex %d is larger than n_maxrs %d, setting end_vertex to start_vertex + n_maxrs . Check inputs!\n", end_vertex - start_vertex, n_maxrs);
            end_vertex = start_vertex + n_maxrs;
        }
        //synch threads here, now everything is same for all threads
        __syncthreads();
        // Fill M
        int fill_counter = 0;
        for(int i_v = start_vertex + tid; i_v < end_vertex; i_v++ ){
            if(asso_idx[i_v] == uqidx){
                M[I2D(fill_counter, k, n_unique)] = i_v;
                fill_counter++;
                if(fill_counter > n_maxuq){
                    printf("Error: fill_counter %d is larger than n_maxuq in first M loop %d . Check inputs!\n", fill_counter, n_maxuq);
                    break;
                }
            }
        }
        //fill rest, might diverge but that's ok
        for(int i_v = start_vertex; i_v < start_vertex + tid; i_v++ ){
            if(i_v < end_vertex && asso_idx[i_v] == uqidx){
                M[I2D(fill_counter, k, n_unique)] = i_v;
                fill_counter++;
                if(fill_counter > n_maxuq){
                    printf("Error: fill_counter %d is larger than n_maxuq in second M loop %d . Check inputs!\n", fill_counter, n_maxuq);
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
            for(int i_v = start_vertex + tid; i_v < end_vertex; i_v++ ){
                if (asso_idx[i_v] != uqidx){
                    M_not[I2D(fill_counter, k, n_unique)] = i_v;
                    fill_counter++;
                    if(fill_counter > n_maxrs){
                        printf("Error: fill_counter %d is larger than n_maxrs in first M_not loop %d . Check inputs!\n", fill_counter, n_maxuq);
                        break;
                    }
                }
            }
            //fill rest, might diverge but that's ok
            for(int i_v = start_vertex; i_v < start_vertex + tid; i_v++ ){
                if (i_v < end_vertex && asso_idx[i_v] != uqidx){
                    M_not[I2D(fill_counter, k, n_unique)] = i_v;
                    fill_counter++;
                    if(fill_counter > n_maxrs){
                        printf("Error: fill_counter %d is larger than n_maxrs in first M_not loop %d . Check inputs!\n", fill_counter, n_maxuq);
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

    if(max_n_unique_over_splits.size(0) != 1){
        throw std::invalid_argument("max_n_unique_over_splits should have size 1");
    }

    if(max_n_in_splits.size(0) != 1){
        throw std::invalid_argument("max_n_in_splits should have size 1");
    }

    if(!asso_idx.is_contiguous() || asso_idx.dtype() != torch::kInt32){
        throw std::invalid_argument("asso_idx should be contiguous and of type int32");
    }

    if(!unique_idx.is_contiguous() || unique_idx.dtype() != torch::kInt32){
        throw std::invalid_argument("unique_idx should be contiguous and of type int32");
    }

    if(!unique_rs_asso.is_contiguous() || unique_rs_asso.dtype() != torch::kInt32){
        throw std::invalid_argument("unique_rs_asso should be contiguous and of type int32");
    }

    if(!rs.is_contiguous() || rs.dtype() != torch::kInt32){
        throw std::invalid_argument("rs should be contiguous and of type int32");
    }

    if(!max_n_unique_over_splits.is_contiguous() || max_n_unique_over_splits.dtype() != torch::kInt32){
        throw std::invalid_argument("max_n_unique_over_splits should be contiguous and of type int32");
    }

    if(!max_n_in_splits.is_contiguous() || max_n_in_splits.dtype() != torch::kInt32){
        throw std::invalid_argument("max_n_in_splits should be contiguous and of type int32");
    }

    if(asso_idx.device() != unique_idx.device() || asso_idx.device() != unique_rs_asso.device() || 
       asso_idx.device() != rs.device() || asso_idx.device() != max_n_unique_over_splits.device() || 
       asso_idx.device() != max_n_in_splits.device()){
        throw std::invalid_argument("All inputs should be on the same device");
    }
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

    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(asso_idx.device());

    //move them to cpu first before accessing the elements
    auto n_maxuq = max_n_unique_over_splits.cpu().data_ptr<int32_t>()[0];
    auto n_maxrs = max_n_in_splits.cpu().data_ptr<int32_t>()[0];

    torch::Tensor M_transposed = torch::empty({n_maxuq, n_unique}, options_int);
    torch::Tensor M_not_transposed = torch::empty({n_maxrs, n_unique}, options_int);

    grid_and_block gb(n_unique, 512);

    calc_m<int32_t><<<gb.grid(), gb.block()>>>(
        asso_idx.data_ptr<int32_t>(),
        unique_idx.data_ptr<int32_t>(),
        unique_rs_asso.data_ptr<int32_t>(),
        rs.data_ptr<int32_t>(),
        M_transposed.data_ptr<int32_t>(),
        M_not_transposed.data_ptr<int32_t>(),
        n_vert,
        n_unique,
        n_maxuq,
        n_maxrs,
        calc_m_not
    );

    cudaDeviceSynchronize();

    torch::Tensor M = M_transposed.transpose(0, 1).contiguous();//ensure contiguous
    torch::Tensor M_not = M_not_transposed.transpose(0, 1).contiguous();

    return std::make_tuple(M, M_not);
}