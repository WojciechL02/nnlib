#include <torch/torch.h>
#include "Linear.h"

Linear::Linear(int in_features, int out_features) : in_features(in_features),
                                                    out_features(out_features)
{
    Linear::init_weight();
    dW = torch::zeros({out_features, in_features});
    db = torch::zeros({out_features, 1});
    vW = torch::zeros({out_features, in_features});
    vb = torch::zeros({out_features, 1});
}

void Linear::init_weight() {
    auto k = torch::sqrt(torch::tensor(1.0f / in_features)).item<float>();
    weight = torch::rand({out_features, in_features}).mul_(2 * k).sub_(k);
    bias = torch::rand({out_features, 1}).mul_(2 * k).sub_(k);
}

torch::Tensor Linear::operator()(const torch::Tensor& input) {
    this->input = input.clone().detach();  // Store a copy of input to avoid memory leak
    torch::Tensor output = torch::matmul(weight, input) + bias;
    return output;
}

torch::Tensor Linear::backward(torch::Tensor& dY) {
    dW = torch::matmul(dY, torch::transpose(input, 0, 1)) / dY.size(1);
    db = torch::sum(dY, 1, true) / dY.size(1);
    torch::Tensor dX = torch::matmul(torch::transpose(weight, 0, 1), dY);
    return dX;
}

void Linear::accept(Optimizer* optimizer) {
    optimizer->visit(this);
}
