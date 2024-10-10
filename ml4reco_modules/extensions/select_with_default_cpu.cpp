#include <torch/extension.h>

template <typename index_t, typename scalar_t>
void select_with_default_cpu_kernel(
    const index_t* __restrict__ indices,     // Indices tensor (K x N)
    const scalar_t* __restrict__ tensor,     // Input tensor (V x F)
    scalar_t* __restrict__ output,           // Output tensor (K x N x F)
    const int K,                             // Number of rows in indices (K)
    const int N,                             // Number of columns in indices (N)
    const int V,                             // Number of rows in input tensor (V)
    const int F,                             // Number of features in input tensor (F)
    const scalar_t default_val               // Default value to set when index is -1
) {
    // Iterate over K and N
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            int index = indices[k * N + n];  // Get the index from the indices array

            // Compute the linear index for the output array (K x N x F)
            int output_offset = k * (N * F) + n * F;

            if (index >= 0 && index < V) {
                // Valid index: Copy the feature vector from the input tensor
                for (int f = 0; f < F; ++f) {
                    output[output_offset + f] = tensor[index * F + f];
                }
            } else {
                // Invalid index (-1 or out of bounds): Set default value
                for (int f = 0; f < F; ++f) {
                    output[output_offset + f] = default_val;
                }
            }
        }
    }
}

torch::Tensor select_with_default_cpu(
    torch::Tensor indices,    // Input indices (K x N)
    torch::Tensor tensor,     // Input tensor (V x F)
    torch::Scalar default_val // Default value to use for invalid indices (-1)
) {
    const auto K = indices.size(0);  // Number of rows in indices
    const auto N = indices.size(1);  // Number of columns in indices
    const auto V = tensor.size(0);   // Number of rows in input tensor (V)
    const auto F = tensor.size(1);   // Number of features in input tensor (F)

    // Create output tensor (K x N x F)
    auto output = torch::zeros({K, N, F}, tensor.options());

    AT_DISPATCH_INTEGRAL_TYPES(indices.scalar_type(), "select_with_default_cpu_kernel", ([&] {
    using index_t = scalar_t;  // The type of indices (int32 or int64)

        // Manually dispatch for both floating and integral types
        AT_DISPATCH_FLOATING_TYPES_AND(torch::kInt32, tensor.scalar_type(), "select_with_default_cpu_kernel", ([&] {
            // You can add more types like torch::kInt64 if necessary
            using scalar_t = scalar_t;  // The type of tensor (float32, int32, etc.)
    
            // Cast default_val to the same type as `tensor`
            scalar_t casted_default_val = default_val.to<scalar_t>();
    
            // Call the kernel with the correct types
            select_with_default_cpu_kernel<index_t, scalar_t>(
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

// Register the function with the repository convention
TORCH_LIBRARY(select_with_default_cpu, m) {
    m.def("select_with_default_cpu", &select_with_default_cpu);
}
