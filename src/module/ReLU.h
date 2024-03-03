#pragma once
#include "Layer.h"

class ReLU : public Layer {
public:
    ReLU() = default;
    torch::Tensor operator()(const torch::Tensor& input) override;
    torch::Tensor backward(torch::Tensor& dY) override;
    void accept(Optimizer* optimizer) override;
};
