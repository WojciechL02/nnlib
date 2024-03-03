#pragma once
#include "Layer.h"
#include <torch/torch.h>

class Softmax : public Layer {
public:
    Softmax() = default;
    torch::Tensor operator()(const torch::Tensor& input) override;
    torch::Tensor backward(torch::Tensor& dY) override;

private:
    torch::Tensor s;
};
