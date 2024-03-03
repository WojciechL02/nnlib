#include "ReLU.h"
#include <torch/torch.h>

torch::Tensor ReLU::operator()(const torch::Tensor& input) {
    this->input = input.clone().detach();
    return torch::maximum(torch::zeros_like(input), input);
}

torch::Tensor ReLU::backward(torch::Tensor& dY) {
    torch::Tensor d_relu = torch::gt(input, torch::zeros_like(input));
    return torch::mul(d_relu, dY);
}

void ReLU::accept(Optimizer* optimizer) {
    optimizer->visit(this);
}
