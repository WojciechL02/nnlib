#pragma once
#include <torch/torch.h>
#include "Layer.h"

class Linear : public Layer {
public:
    Linear(int in_features, int out_features);
    torch::Tensor operator()(const torch::Tensor& input) override;
    torch::Tensor backward(torch::Tensor& dY) override;
    void accept(Optimizer* optimizer) override;

    torch::Tensor weight;
    torch::Tensor bias;
    torch::Tensor dW;
    torch::Tensor db;
    torch::Tensor vW;
    torch::Tensor vb;

private:
    void init_weight();
    int in_features;
    int out_features;
};
