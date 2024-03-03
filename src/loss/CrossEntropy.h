#pragma once
#include "Loss.h"

class CrossEntropy : public Loss {
public:
    CrossEntropy() = default;
    torch::Tensor operator()(torch::Tensor& logits, torch::Tensor& target) const override;
    torch::Tensor derivative(torch::Tensor& logits, torch::Tensor& target) override;
};
