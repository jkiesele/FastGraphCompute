#include <torch/extension.h>

#define CHECK_INPUT(x) AT_ASSERTM(!x.is_cuda(), #x " must be a CPU tensor")



void compute(
            const int * to_be_replaced,
            const int * replacements,
            int * replaced,

            const int n_to_be_replaced,
            const int n_replacements)
{
    for(int i=0;i<n_to_be_replaced;i++){
        const int ridx = to_be_replaced[i];
        if(ridx<0){
            replaced[i] = ridx;
            continue;
        }
        if(ridx>=n_replacements){
            printf("IndexReplacerOpFunctor: index out of range\n");
            continue;
        }
        replaced[i] = replacements[ridx];
    }
}



void index_replacer_cpu(
    const at::Tensor& to_be_replaced,
    const at::Tensor& replacements,
    at::Tensor& replaced
) {
    CHECK_INPUT(to_be_replaced);
    CHECK_INPUT(replacements);
    CHECK_INPUT(replaced);

    auto to_be_replaced_a = to_be_replaced.accessor<int,1>();
    auto replacements_a = replacements.accessor<int,1>();
    auto replaced_a = replaced.accessor<int,1>();

    int n_to_be_replaced = to_be_replaced.size(0);
    int n_replacements = replacements.size(0);

    // for(int i=0;i<n_to_be_replaced;i++){
    //     for(int j=0;j<n_replacements;j++){
    //         if(to_be_replaced_a[i] == replacements_a[j]){
    //             replaced_a[i] = replacements_a[j];
    //             break;
    //         }
    //     }
    // }

    compute(
        to_be_replaced_a.data(),
        replacements_a.data(),
        replaced_a.data(),
        n_to_be_replaced,
        n_replacements
    );
}

#ifdef CUDA_AVAILABLE
void index_replacer_cuda(
    const at::Tensor& to_be_replaced,
    const at::Tensor& replacements,
    at::Tensor& replaced
);
#endif

void index_replacer(
    const at::Tensor& to_be_replaced,
    const at::Tensor& replacements,
    at::Tensor& replaced
) {
    if (replaced.is_cuda()) {
        #ifdef CUDA_AVAILABLE
        index_replacer_cuda(to_be_replaced, replacements, replaced);
        #else
        AT_ERROR("index_replacer is not available on CUDA");
        #endif
    } else {
        index_replacer_cpu(to_be_replaced, replacements, replaced);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("index_replacer", &index_replacer, "Index replacer");
    m.def("index_replacer_cpu", &index_replacer_cpu, "Index replacer CPU");
}