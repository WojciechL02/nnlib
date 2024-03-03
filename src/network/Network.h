#pragma once
#include <torch/torch.h>
#include "../module/Layer.h"

class Network {
public:
    ~Network() = default;
    virtual void add(std::unique_ptr<Layer> layer) = 0;
    virtual void backward(torch::Tensor dloss) = 0;
    virtual torch::Tensor operator()(torch::Tensor& data) = 0;
};
