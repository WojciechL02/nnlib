#pragma once
#include <torch/torch.h>
#include "../optim/Optimizer.h"

class Layer {
public:
    virtual ~Layer() = default;
    virtual torch::Tensor operator()(const torch::Tensor& input) = 0;
    virtual torch::Tensor backward(torch::Tensor& dY) = 0;
    virtual void accept(Optimizer* optimizer) = 0;

protected:
    torch::Tensor input;
};
