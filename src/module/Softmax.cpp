#include "Softmax.h"
#include <torch/torch.h>

torch::Tensor Softmax::operator()(const torch::Tensor& input) {
    s = torch::exp(input) / torch::sum(torch::exp(input), 0, true);
    return s;
}

torch::Tensor Softmax::backward(torch::Tensor& dY) {
    return s * (1 - s);
}
