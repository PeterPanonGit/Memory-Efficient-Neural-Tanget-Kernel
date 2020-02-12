# Memory Efficient Neural Tanget Kernel
This code implements a memory efficient Neural Tanget Kernel algorithm based on the following publications:
* [Harnessing the Power of Infinitely Wide Deep Nets on Small-data Tasks](https://arxiv.org/abs/1910.01663) and its [Github rep](https://github.com/LeoYu/neural-tangent-kernel-UCI)
* [On the Inductive Bias of Neural Tangent Kernels](https://arxiv.org/abs/1905.12173)
* [On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956)
## Motivations
### Background
Neural Tanget Kernel (NTK) has attracted much attention recently because research showed that NTK can approximate behavior of neural networks with infinite width learned with L2 loss. This paper provided a proof that Kernel Ridge Regression using NTK is equivalent to a fully trained infinitely wide neural network: [On Exact Computation with an Infinitely Wide
Neural Net](https://arxiv.org/abs/1904.11955). Because NTK provides exact solutions which does not depend on hyperparameters like weight initialization methods, learning rate, batch size, etc., its property could be desirable for small and noisy datasets, as proved in paper [Harnessing the Power of Infinitely Wide Deep Nets on Small-data Tasks](https://arxiv.org/abs/1910.01663). The size of the NTK matrix is N<sup>2</sup>, where N is the number of data points. Just keeping the NTK of 100,000 data points in memory using float32 requires 100k * 100k * 4 = 40G of memory. Most existing implementations need multiple auxilary matrices of the same size as the NTK matrix, making it difficult for NTK algorithm to scale large datasets. It is important to use as few auxilary matrices as possible to make the NTK algorithm memory efficent
### Contributions
* This implementation uses in place computation and uses only one auxilary matrix to compute the NTK matrix, making the memory complexity of the NTK algorithm to be O(2N<sup>2</sup>).
* It provides a clean and easy to use API to compute NTK matrix with different number of total layers and number of fixed layers.
## How to Use
### API
The main API to compute NTK matrix is the NtkIterator class defined in NTK.py. Constructor of the NtkIterator takes train data, eval data, and max number of layers as input. Then set_fix_dep() method should be called to define the number of layers that are fixed (not trained). When next() method is called, the NKT matrix H in the NtkIterator class instance will be updated to the values of the next trainable layer, and the H matrix can be used in a Kernel Ridge Regression to approximate the performance of a infinitely wide neural network.
### Examples
Please find a working example in Boston_dataset.ipynb. This example compared the performance of NTK based Kernel Ridge Regression and linear Ridge Regression using the [Boston Housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) with Root of Mean Squared Error (RMSE) and R<sup>2</sup> as metrics. Cross validation based model selection is used for both algorithms to select the optimum hyperparameters.
### Code Structure
* NTK.hpp / NTK.cpp: internal implementation of the NTK algorithm using C++ for better memory efficency and speed.
* NTK.py: python wrapper around the C++ implementation to provide easy to use API.
* NTK_test.py: test cases to ensure the correctness of the NTK algorithm implementation.
## Performance
### Memory Consumption
The memory consumption for this implementation scale at the speed of 2N<sup>2</sup>. In place computation are used as much as possible to minize memory consumption, and 2N<sup>2</sup> is the minimum possible memory consumption for NTK algorithm because the algorithm requires one auxilary matrix which is of the same size as the N<sup>2</sup> sized NTK matrix. The auxiliary matrix can be discarded after NTK computation is finished. [Kernel Ridge Regression](https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf) in Sklearn is implemented based on [Cholesky Decomposition](https://github.com/scikit-learn/scikit-learn/blob/c79a5b4194de6fe4b7b64396999352e38170cf57/sklearn/linear_model/_ridge.py#L159) and thus should consume approximately 2N<sup>2</sup> at its peak. So the total memory consumption for computing a Kernel Ridge Regression using NTK can be approximated as 2 * N * N * 4 bytes (float32 is used instead of double). Solving a problem with 100,000 data points would take about 80G memory.
### Computation Speed
Computation of the NTK is O(N<sup>2</sup>). So the bottleneck of the speed is the Kernel Ridge Regression, which requires the computation of the inverse of the kernel matrix and its complexity is O(N<sup>3</sup>). Sklearn has benchmarked the [performance of its Kernel Ridge Regression](https://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html), which solves 10,000 datapoints at around 10 seconds. If we believe the complexity scales at the speed of N<sup>3</sup>, we may assume that solving 100,000 data points takes 10 * 10<sup>3</sup> ~ 3 hours. It is possible to speed up the Kernel Ridge Regression by approximating the kernal matrix using random features, as suggested in [Random Fourier Features for Kernel Ridge Regression: Approximation Bounds and Statistical Guarantees](https://arxiv.org/abs/1804.09893). This can speed up the computation speed of Kernel Ridge Regression but does not change the memory consumption.
### Limits
Even with this memory efficient implementation, it would be hard to scale NTK based Kernel Ridge Regression to very large dataset. If we do 50% split on train and test data, we may expect the largest dataset this method can handle is around 500k data points if you want the computation to be able to finish in a day. It is possible to do tune hyperparameters and do model selection on a small randomly sampled dataset, and then evaluate the model on the large dataset.
## Setup
### Prerequisites
g++(version > 7), pybind11, Python3, numpy, sklearn
### Build
Before running the python code and Jupyter notebook example, you need to download C++ package pybind11, and then compile the C++ to a shared library using the following command:
```
git clone https://github.com/pybind/pybind11.git
g++-7 -O3 -shared -std=gnu++17 -I ./pybind11/include `python3-config --cflags --ldflags --libs` NTK.cpp -o NtkInternal.so -fPIC
```
### Running Tests
To make sure that the python package works as intended, use the following command to run tests:
```
python NTK_test.py
```
