#pragma once
#include <torch/torch.h>

class Loss {
public:
    virtual ~Loss() = default;
    virtual torch::Tensor operator()(torch::Tensor& logits, torch::Tensor& target) const = 0;
    virtual torch::Tensor derivative(torch::Tensor& logits, torch::Tensor& target) = 0;
};
