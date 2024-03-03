#include <iostream>
#include "SGD.h"
#include "../module/Layer.h"
#include "../module/Linear.h"
#include "../module/ReLU.h"

SGD::SGD(std::vector<std::unique_ptr<Layer>>& layers, float lr, float momentum)
    : Optimizer(layers), lr(lr), momentum(momentum) {}

void SGD::step() {
    for (auto& layer : layers.get()) {
        layer->accept(this);
    }
}

void SGD::visit(Linear* layer) {
    layer->vW = momentum * layer->vW - layer->dW;
    layer->vb = momentum * layer->vb - layer->db;
    layer->weight = layer->weight + layer->vW;
    layer->bias = layer->bias + layer->vb;
}
