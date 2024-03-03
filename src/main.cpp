#include <iostream>
#include <memory>
#include <torch/torch.h>
#include "module/Linear.h"
#include "module/Softmax.h"
#include "module/ReLU.h"
#include "loss/CrossEntropy.h"
#include "network/MLP.h"
#include "optim/SGD.h"

int main() {
    torch::AutoGradMode enable_grad(false);

    const torch::Device device = torch::kCPU;
    const std::string data_root = "../data";
    const int num_classes = 10;
    const int64_t batch_size = 64;
    const int epochs = 10;

    auto train_dataset = torch::data::datasets::MNIST(data_root)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto train_loader =
            torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), batch_size);

    auto test_dataset = torch::data::datasets::MNIST(
            data_root, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), batch_size);


    MLP net;
    net.add(std::make_unique<Linear>(784, 128));
    net.add(std::make_unique<ReLU>());
    net.add(std::make_unique<Linear>(128, 10));


    CrossEntropy criterion;
    SGD sgd(net.layers, 0.01, 0.9);

    for (int epoch = 0; epoch < epochs; epoch++) {
        //    TRAINING
        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device).flatten(2).squeeze(1).permute({1, 0});
            auto targets = batch.target.to(device);
            targets = torch::nn::functional::one_hot(targets, num_classes).permute({1, 0}).to(torch::kFloat);

            auto output = net(data);
            auto dloss = criterion.derivative(output, targets);
            auto loss = criterion(output, targets);

            net.backward(dloss);
            sgd.step();
        }

        // VALIDATION
        float total_loss = 0.;
        int num_batch = 0;
        float correct = 0.;
        float num_samples = 0.;
        for (auto& batch : *test_loader) {
            auto data = batch.data.to(device).flatten(2).squeeze(1).permute({1, 0});
            auto targets = batch.target.to(device);
            targets = torch::nn::functional::one_hot(targets, num_classes).permute({1, 0}).to(torch::kFloat);

            auto output = net(data);
            auto arg = torch::nn::functional::one_hot(argmax(output, 0), num_classes).permute({1, 0}).to(torch::kFloat);
            correct += torch::mul(arg, targets).sum().item().to<float>();
            num_samples += data.size(1);

            auto loss = criterion(output, targets);
            total_loss += loss.item().to<float>();
            num_batch += 1;
        }
        std::cout << "Epoch " << epoch + 1 << " | Loss: " << total_loss / num_batch << " | Acc: " << correct / num_samples << std::endl;
    }

    return 0;
}
