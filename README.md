# Memory Efficient Neural Tanget Kernel
This code implements Neural Tanget Kernel algorithm based on the following publications:
* [Harnessing the Power of Infinitely Wide Deep Nets on Small-data Tasks](https://arxiv.org/abs/1910.01663) and its [Github rep](https://github.com/LeoYu/neural-tangent-kernel-UCI)
* [On the Inductive Bias of Neural Tangent Kernels](https://arxiv.org/abs/1905.12173)
* [On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956)
## Motivations
Neural Tanget Kernel (NTK) has attracted much attention recently because research showed that NTK can approximate behavior of neural networks with infinite width learned with L2 loss. Because NTK provides exact solutions which does not depend on hyperparameters like weight initialization methods, learning rate, batch size, etc., its property could be desirable for small and noisy datasets, as proved in paper [Harnessing the Power of Infinitely Wide Deep Nets on Small-data Tasks](https://arxiv.org/abs/1910.01663). However, the memory requirement for NTK scales at the rate of N<sup>2</sup>, where N is the number of data points. Just keeping the NTK of 100,000 data points in memory using float32 requires 100k * 100k * 4 = 40G of memory. Although NTK algorithm is fundamentally not scalable, it is important to make the computation of NTK as memory efficient as possible so as to make it scale to larger data size given the same memory and compute constraints.
## Implementation
NTK computation is implemented in C++ with a easy to use wrapper in Python. The memory consumption for computing the NTK is 2N<sup>2</sup> because it requires one auxilary matrix of the same size as NTK. The auxiliary matrix can be discarded after NTK computation is finished. Memory efficient computation of [Kernel Ridge Regression](https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf) based on the NTK has not been implemented yet. If we use [Arpack-Eigen](https://github.com/yixuan/arpack-eigen) to compute the exact or approximate inverse of the NTK matrix, we could probably expect that the total memory consumption of a Kernel Ridge Regression based on NTK would consume a few times of the memory of the NTK matrix.
## Setup
### Prerequisites
g++(version > 7), Python3, numpy, sklearn
### Build
Before running the python code and Jupyter notebook example, you need to compile the C++ to a shared library using the following command:
```
g++-7 -O3 -shared -std=gnu++17 -I ./pybind11/include `python3-config --cflags --ldflags --libs` NTK.cpp -o NtkInternal.so -fPIC
```
### Running Tests
To make sure that the python package works as intended, use the following command to run tests:
```
python NTK_test.py
```
## Example
Please find a working example in Boston_dataset.ipynb. This example compared the performance of NTK based Kernel Ridge Regression and linear Ridge Regression using the [Boston Housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
