#include "CrossEntropy.h"
#include <torch/torch.h>

torch::Tensor CrossEntropy::operator()(torch::Tensor& logits, torch::Tensor& target) const {
    logits = torch::log_softmax(logits, 0);
    auto loss = torch::sum(-target * logits, 0).mean();
    return loss;
}

torch::Tensor CrossEntropy::derivative(torch::Tensor& logits, torch::Tensor& target) {
    logits = torch::softmax(logits, 0);
    auto derivative = logits - target;
    return derivative / logits.size(1);
}
