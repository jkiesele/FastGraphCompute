#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <typename index_t, typename scalar_t>
__global__ void select_with_default_cuda_kernel(
    const index_t* __restrict__ indices,     // Indices tensor (K x N)
    const scalar_t* __restrict__ tensor,     // Input tensor (V x F)
    scalar_t* __restrict__ output,           // Output tensor (K x N x F)
    const int K,                             // Number of rows in indices (K)
    const int N,                             // Number of columns in indices (N)
    const int V,                             // Number of rows in input tensor (V)
    const int F,                             // Number of features in input tensor (F)
    const scalar_t default_val               // Default value to set when index is -1
) {
    int k = blockIdx.x;  // Current row in indices (K dimension)
    int n = blockIdx.y;  // Current column in indices (N dimension)
    int f = threadIdx.x; // Current feature in the feature dimension (F)

    // Calculate the linear index for output (K x N x F)
    int output_index = k * (N * F) + n * F + f;

    // Get the index from indices
    int index = indices[k * N + n];

    if (index >= 0 && index < V) {
        // Gather the value from the input tensor (V x F)
        output[output_index] = tensor[index * F + f];
    } else {
        // If the index is -1 or out of bounds, set the default value
        output[output_index] = default_val;
    }
}

torch::Tensor select_with_default_cuda_fn(
    torch::Tensor indices,    // Input indices (K x N)
    torch::Tensor tensor,     // Input tensor (V x F)
    torch::Scalar default_val // Default value to use for invalid indices (-1)
) {
    const auto K = indices.size(0);  // Number of rows in indices
    const auto N = indices.size(1);  // Number of columns in indices
    const auto V = tensor.size(0);   // Number of rows in input tensor (V)
    const auto F = tensor.size(1);   // Number of features in input tensor (F)

    // Create output tensor (K, N, F)
    auto output = torch::zeros({K, N, F}, tensor.options());

    // Define the block and grid size
    dim3 blocks(K, N);        // Each block handles one entry in the K x N grid
    dim3 threads(F);          // Each thread in the block handles one feature

    AT_DISPATCH_INTEGRAL_TYPES(indices.scalar_type(), "select_with_default_cuda_kernel", ([&] {
    using index_t = scalar_t;  // The type of indices (int32 or int64)
    
      // Dispatch for both floating and integral types, including float32, float64, and int32
      AT_DISPATCH_FLOATING_TYPES_AND(torch::kInt32, tensor.scalar_type(), "select_with_default_cuda_kernel", ([&] {
        // You can add more types like torch::kInt64 if necessary
        using scalar_t = scalar_t;  // The type of tensor (float32, int32, etc.)

        // Cast default_val to the same type as `tensor`
        scalar_t casted_default_val = default_val.to<scalar_t>();

        // Call the kernel with the correct types
        select_with_default_cuda_kernel<index_t, scalar_t><<<blocks, threads>>>(
            indices.data_ptr<index_t>(),   // Indices tensor
            tensor.data_ptr<scalar_t>(),   // Input tensor
            output.data_ptr<scalar_t>(),   // Output tensor
            K, N, V, F,                    // Dimensions
            casted_default_val             // Casted default value
        );
      }));
    }));

    return output;
}
