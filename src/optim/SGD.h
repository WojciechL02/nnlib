#pragma once
#include "Optimizer.h"

class SGD : public Optimizer {
public:
    SGD(std::vector<std::unique_ptr<Layer>>& layers, float lr, float momentum);
    void step() override;
    void visit(Linear* layer) override;

private:
    float lr;
    float momentum;
};
