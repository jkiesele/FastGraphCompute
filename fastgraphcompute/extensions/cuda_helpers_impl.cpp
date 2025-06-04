/*
 * cuda_helpers_impl.cpp
 *
 * Implementation file for cuda_helpers.h
 */

#include "cuda_helpers.h"

#ifndef __CUDACC__
// Define static variables for host compilation
dim3 __cuda_builtin_vars_dummy::gridDim;
dim3 __cuda_builtin_vars_dummy::blockDim;
uint3 __cuda_builtin_vars_dummy::blockIdx;
uint3 __cuda_builtin_vars_dummy::threadIdx;
#endif
