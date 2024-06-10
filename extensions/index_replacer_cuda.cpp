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

TORCH_LIBRARY(index_replacer_cuda, m) {
  m.def("index_replacer_cuda", index_replacer_cuda_interface);
}





// #include <torch/extension.h>
// #include <vector>

// // CUDA forward declarations
// torch::Tensor index_replacer_cuda_fn(
//     torch::Tensor input, 
//     torch::Tensor indices,
//     torch::Tensor replacements
//     );

// // C++ interface
// #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// torch::Tensor index_replacer_cuda_interface(
//     torch::Tensor input, 
//     torch::Tensor indices,
//     torch::Tensor replacements
//     ){
//   CHECK_INPUT(input);
//   CHECK_INPUT(indices);
//   CHECK_INPUT(replacements);
//   return index_replacer_cuda_fn(
//     input, indices, replacements
//     );
// }

// TORCH_LIBRARY(index_replacer_cuda, m) {
//   m.def("index_replacer_cuda", index_replacer_cuda_interface);
// }