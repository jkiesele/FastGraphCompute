#include <torch/extension.h>
#include <vector>
#include <iostream>

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")

void index_replacer_cpu_kernel(
    const int64_t* to_be_replaced,
    const int64_t* replacements,
    int64_t* replaced,
    const int64_t n_to_be_replaced,
    const int64_t n_replacements
) {
    for(int i=0;i<n_to_be_replaced;i++){
        const int64_t ridx = to_be_replaced[i];
        if(ridx<0){
            replaced[i] = ridx;
            continue;
        }
        if(ridx>=n_replacements){
            replaced[i] = replacements[i];
            continue;
        }
        replaced[i] = replacements[ridx];
    }
}

torch::Tensor index_replacer_cpu_fn(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements
) {
    CHECK_CPU(to_be_replaced);
    CHECK_CPU(replacements);

    auto replaced = torch::empty_like(to_be_replaced);

    auto n_to_be_replaced = to_be_replaced.numel();
    auto n_replacements = replacements.size(0);

    index_replacer_cpu_kernel(
        to_be_replaced.data_ptr<int64_t>(),
        replacements.data_ptr<int64_t>(),
        replaced.data_ptr<int64_t>(),
        n_to_be_replaced,
        n_replacements
    );

    return replaced;
}