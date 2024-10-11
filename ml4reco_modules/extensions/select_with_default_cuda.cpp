#include <torch/extension.h>

//forward declaration
torch::Tensor select_with_default_cuda_fn(
    torch::Tensor indices,    // Input indices (K x N)
    torch::Tensor tensor,     // Input tensor (V x F)
    torch::Scalar default_val // Default value to use for invalid indices (-1)
);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor select_with_default_cuda_interface(
    torch::Tensor indices,    // Input indices (K x N)
    torch::Tensor tensor,     // Input tensor (V x F)
    torch::Scalar default_val // Default value to use for invalid indices (-1)
)
{
    CHECK_INPUT(indices);
    CHECK_INPUT(tensor);

    //check if default_val is on cpu (should be)
    //TORCH_CHECK(default_val.device().is_cpu(), "default_val must be a CPU scalar");

    return select_with_default_cuda_fn(indices, tensor, default_val);
}

// Register the C++ interface with PyTorch
TORCH_LIBRARY(select_with_default_cuda, m) {
    m.def("select_with_default_cuda", &select_with_default_cuda_interface);
}