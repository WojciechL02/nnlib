cmake_minimum_required(VERSION 3.27)
set(CMAKE_CXX_STANDARD 17)
project(nnlib)

set(CMAKE_PREFIX_PATH /home/wojciech/Downloads/libtorch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(nnlib src/main.cpp
        src/module/Layer.h
        src/module/Linear.cpp
        src/module/Linear.h
        src/module/ReLU.cpp
        src/module/ReLU.h
        src/module/Softmax.cpp
        src/module/Softmax.h
        src/loss/Loss.h
        src/loss/CrossEntropy.cpp
        src/loss/CrossEntropy.h
        src/network/Network.h
        src/network/MLP.cpp
        src/network/MLP.h
        src/optim/Optimizer.h
        src/optim/SGD.cpp
        src/optim/SGD.h
        src/module/BatchNorm2D.cpp
        src/module/BatchNorm2D.h
)

target_link_libraries(nnlib "${TORCH_LIBRARIES}")

set_property(TARGET nnlib PROPERTY CXX_STANDARD 17)
