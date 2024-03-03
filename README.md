# NN Library

## Description

This is a library for creating neural networks implemented from scratch in C++ using Libtorch (withouth AutoGrad, just using Tensors and mathematical operations). Backpropagation is also implemented from scratch (for now only SGD optimizer is available). 

## Functionality

* Create custom Multi-Layer Perceptron
* Linear layers
* Softmax, ReLU
* SGD optimizer
* Cross Entropy loss 

## Usage

Firstly, you need to download the dataset:
```bash
cd data
./download.sh
```
Then in the main directory:
```bash
cmake .
make
./nnlib
```

## Tests

The program has been briefly tested (only to check the correct operation of all modules) on the MNIST set. <br><br>
Setup:
Architecture: Linear(784, 128), ReLU, Linear(128, 10) <br>
Optimizer: SGD(lr=0.01, momentum=0.9) <br><br>
Results (on validation set):
```
Epoch 1 | Loss: 0.165423 | Acc: 0.9454
Epoch 2 | Loss: 0.106779 | Acc: 0.9669
Epoch 3 | Loss: 0.101036 | Acc: 0.9695
Epoch 4 | Loss: 0.0876465 | Acc: 0.9745
Epoch 5 | Loss: 0.081219 | Acc: 0.977
Epoch 6 | Loss: 0.0766412 | Acc: 0.9792
Epoch 7 | Loss: 0.0775358 | Acc: 0.9802
Epoch 8 | Loss: 0.0776486 | Acc: 0.979
Epoch 9 | Loss: 0.0761722 | Acc: 0.9801
Epoch 10 | Loss: 0.0751737 | Acc: 0.9802
```
