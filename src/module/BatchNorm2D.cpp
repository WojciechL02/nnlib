#include <torch/torch.h>
#include "BatchNorm2D.h"

BatchNorm2D::BatchNorm2D(int num_features) : num_features(num_features), momentum(0.1f)
{
    gamma = torch::ones({1, num_features, 1, 1});
    beta = torch::zeros({1, num_features, 1, 1});
    moving_mean = torch::zeros({1, num_features, 1, 1});
    moving_var = torch::ones({1, num_features, 1, 1});
}

torch::Tensor BatchNorm2D::operator()(const torch::Tensor& input) {
    auto data = input.clone().detach();
    auto mean = data.mean(1, true);
    auto var = torch::pow((data - mean), torch::tensor({2})).mean({0, 2, 3}, true);
    auto x_hat = (data - mean) / torch::sqrt(var + 1e-5);

    moving_mean = (1.0 - momentum) * moving_mean + momentum * mean;
    moving_var = (1.0 - momentum) * moving_var + momentum * var;

    return gamma * x_hat + beta;
}

torch::Tensor BatchNorm2D::backward(torch::Tensor& dY) {
    return torch::ones({1, 1});
}

void BatchNorm2D::accept(Optimizer* optimizer) {

}
