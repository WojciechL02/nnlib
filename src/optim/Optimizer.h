#pragma once
#include <vector>
#include <memory>

class Layer;
class Linear;
class ReLU;

class Optimizer {
public:
    explicit Optimizer(std::vector<std::unique_ptr<Layer>>& layers) : layers(layers) {}
    ~Optimizer() = default;
    virtual void step() = 0;
    virtual void visit(Linear* layer) = 0;
    virtual void visit(ReLU* layer) {};

protected:
    std::reference_wrapper<std::vector<std::unique_ptr<Layer>>> layers;
};
