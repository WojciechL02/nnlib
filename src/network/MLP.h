#pragma once
#include <torch/torch.h>
#include <vector>
#include <memory>
#include "Network.h"
#include "../module/Layer.h"

class MLP : public Network {
public:
    MLP() = default;
    void add(std::unique_ptr<Layer> layer) override;
    void backward(torch::Tensor dloss) override;
    torch::Tensor operator()(torch::Tensor& data) override;

    std::vector<std::unique_ptr<Layer>> layers;
};
