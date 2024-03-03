#include "MLP.h"
#include "../module/Layer.h"

void MLP::add(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

void MLP::backward(torch::Tensor dloss) {
    torch::Tensor grad = dloss;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
}

torch::Tensor MLP::operator()(torch::Tensor& data) {
    torch::Tensor output = data;
    for (auto& l : layers) {
        output = (*l)(output);
    }
    return output;
}
