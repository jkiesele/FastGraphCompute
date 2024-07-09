#include <torch/extension.h>
#include <vector>
#include <iostream>

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")

void index_replacer_cpu_kernel(
    const int* to_be_replaced,
    const int* replacements,
    int* replaced,
    const int n_to_be_replaced,
    const int n_replacements
) {
    for (int i = 0; i < n_to_be_replaced; ++i) {
        int ridx = to_be_replaced[i];
        if (ridx < 0 || ridx >= n_replacements) {
            replaced[i] = ridx;
            if (ridx >= n_replacements) {
                std::cout << "IndexReplacer: index out of range\n";
            }
            continue;
        }
        replaced[i] = replacements[ridx];
    }
}

torch::Tensor index_replacer_cpu(
    torch::Tensor to_be_replaced,
    torch::Tensor replacements
) {
    CHECK_CPU(to_be_replaced);
    CHECK_CPU(replacements);

    auto replaced = torch::empty_like(to_be_replaced);

    auto n_to_be_replaced = to_be_replaced.numel();
    auto n_replacements = replacements.size(0);

    auto to_be_replaced_ptr = to_be_replaced.data_ptr<int>();
    auto replacements_ptr = replacements.data_ptr<int>();
    auto replaced_ptr = replaced.data_ptr<int>();

    index_replacer_cpu_kernel(
        to_be_replaced_ptr,
        replacements_ptr,
        replaced_ptr,
        n_to_be_replaced,
        n_replacements
    );

    return replaced;
}

TORCH_LIBRARY(index_replacer_cpu, m) {
    m.def("index_replacer_cpu", index_replacer_cpu);
}