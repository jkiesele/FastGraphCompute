#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::tuple<torch::Tensor, torch::Tensor> index_replacer_cuda_fn(
    torch::Tensor to_be_replaced, 
    torch::Tensor replacements,
    torch::Tensor replaced,
    int64_t n_to_be_replaced, 
    int64_t n_replacements
    );

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> index_replacer_cuda_interface(
    torch::Tensor to_be_replaced, 
    torch::Tensor replacements,
    torch::Tensor replaced,
    int64_t n_to_be_replaced, 
    int64_t n_replacements
    ){
  CHECK_INPUT(to_be_replaced);
  CHECK_INPUT(replacements);
  CHECK_INPUT(replaced);
  return index_replacer_cuda_fn(
    to_be_replaced, replacements, replaced, n_to_be_replaced, n_replacements
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("index_replacer", &index_replacer_cuda_interface, "Index Replacer (CUDA)");
}

// TORCH_LIBRARY(index_replacer_cuda, m) {
//   m.def("index_replacer_cuda", index_replacer_cuda_interface);
// }