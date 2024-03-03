#pragma once
#include <torch/torch.h>
#include "Layer.h"

class BatchNorm2D : public Layer {
public:
    BatchNorm2D(int num_features);
    torch::Tensor operator()(const torch::Tensor& input) override;
    torch::Tensor backward(torch::Tensor& dY) override;
    void accept(Optimizer* optimizer) override;

private:
    int num_features;
    float momentum;
    torch::Tensor gamma;
    torch::Tensor beta;
    torch::Tensor moving_mean;
    torch::Tensor moving_var;
};
