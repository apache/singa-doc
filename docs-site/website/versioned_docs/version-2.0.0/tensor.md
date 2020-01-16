---
id: version-2.0.0-tensor
title: Tensor
original_id: tensor
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Each Tensor instance is a multi-dimensional array allocated on a specific Device instance. Tensor instances store variables and provide linear algebra operations over different types of hardware devices without user awareness. Note that users need to make sure the tensor operands are allocated on the same device except copy functions.

## Tensor Usage

### Create Tensor

```python
>>> import numpy as np
>>> from singa import tensor
>>> tensor.from_numpy( np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32) )
[[1. 0. 0.]
 [0. 1. 0.]]
```

### Convert to numpy

```python
>>> a = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
>>> tensor.from_numpy(a)
[[1. 0. 0.]
 [0. 1. 0.]]
>>> tensor.to_numpy(tensor.from_numpy(a))
array([[1., 0., 0.],
       [0., 1., 0.]], dtype=float32)
```

### Tensor Methods

```python
>>> t = tensor.from_numpy(a)
>>> t.transpose([1,0])
[[1. 0.]
 [0. 1.]
 [0. 0.]]
```

### Tensor Arithmetic Methods

`tensor` is evaluated in real time.

```python
>>> t + 1
[[2. 1. 1.]
 [1. 2. 1.]]
>>> t / 5
[[0.2 0.  0. ]
 [0.  0.2 0. ]]
```

### Tensor Functions

Functions in module `singa.tensor` return new `tensor` object after applying defined transformation in the function.

```python
>>> tensor.log(t+1)
[[0.6931472 0.        0.       ]
 [0.        0.6931472 0.       ]]
```

### Tensor on Different Devices

`tensor` is created on host(CPU) by default, and can also be created on different backends by specifiying the `device`. Existing `tensor` could also be moved between `device` by `to_device()`.

```python
>>> from singa import device
>>> x = tensor.Tensor((2, 3), device.create_cuda_gpu())
>>> x.gaussian(1,1)
>>> x
[[1.531889   1.0128608  0.12691343]
 [2.1674204  3.083676   2.7421203 ]]
>>> # move to host
>>> x.to_device(device.get_default_device())
```

### Simple Neural Network Example

```python
from singa import device
from singa import tensor
from singa import opt
from singa import autograd
class MLP:
    def __init__(self):
        self.linear1 = autograd.Linear(3, 4)
        self.linear2 = autograd.Linear(4, 5)
    def forward(self, x):
        y=self.linear1(x)
        return self.linear2(y)
def train(model, x, t, dev, epochs=10):
    for i in range(epochs):
        y = model.forward(x)
        loss = autograd.mse_loss(y, t)
        print("loss: ", loss)
        sgd = opt.SGD()
        for p, gp in autograd.backward(loss):
            sgd.update(p, gp)
        sgd.step()
    print("training completed")
if __name__ == "__main__":
    autograd.training = True
    model = MLP()
    dev = device.get_default_device()
    x = tensor.Tensor((2, 3), dev)
    t = tensor.Tensor((2, 5), dev)
    x.gaussian(1,1)
    t.gaussian(1,1)
    train(model, x, t, dev)
```

Output:

```
loss:  [4.917431]
loss:  [2.5147934]
loss:  [2.0670078]
loss:  [1.9179827]
loss:  [1.8192691]
loss:  [1.7269677]
loss:  [1.6308627]
loss:  [1.52674]
loss:  [1.4122975]
loss:  [1.2866782]
training completed
```

## Tensor implementation

SINGA has three different sets of implmentations of Tensor functions, one for each type of Device.

- 'tensor_math_cpp.h' implements operations using Cpp (with CBLAS) for CppGPU devices.
- 'tensor_math_cuda.h' implements operations using Cuda (with cuBLAS) for CudaGPU devices.
- 'tensor_math_opencl.h' implements operations using OpenCL for OpenclGPU devices.

## Python API

_work in progress_

## CPP API

_work in progress_
