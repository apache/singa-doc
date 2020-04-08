---
id: version-3.0.0.rc1-tensor
title: Tensor
original_id: tensor
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Each Tensor instance is a multi-dimensional array allocated on a specific Device
instance. Tensor instances store variables and provide linear algebra operations
over different types of hardware devices without user awareness. Note that users
need to make sure the tensor operands are allocated on the same device except
copy functions.

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

Functions in module `singa.tensor` return new `tensor` object after applying the
transformation defined in the function.

```python
>>> tensor.log(t+1)
[[0.6931472 0.        0.       ]
 [0.        0.6931472 0.       ]]
```

### Tensor on Different Devices

`tensor` is created on host (CPU) by default; it can also be created on
different hardware devices by specifying the `device`. A `tensor` could be moved
between `device`s via `to_device()` function.

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

## Tensor Implementation

The previous section shows the general usage of `Tensor`, the implementation
under the hood will be covered below. First, the design of Python and C++
tensors will be introduced. Later part will talk about how the frontend (Python)
and backend (C++) are connected and how to extend them.

### Python Tensor

Python class `Tensor`, defined in `python/singa/tensor.py`, provides high level
tensor manipulations for implementing deep learning operations (via
[autograd](./autograd)), as well as data management by end users.

It primarily works by simply wrapping around C++ tensor methods, both arithmetic
(e.g. `sum`) and non arithmetic methods (e.g. `reshape`). Some advanced
arithmetic operations are later introduced and implemented using pure Python
tensor API, e.g. `tensordot`. Python Tensor APIs could be used to implement
complex neural network operations easily with the flexible methods available.

### C++ Tensor

C++ class `Tensor`, defined in `include/singa/core/tensor.h`, primarily manages
the memory that holds the data, and provides low level APIs for tensor
manipulation. Also, it provides various arithmetic methods (e.g. `matmul`) by
wrapping different backends (CUDA, BLAS, cuBLAS, etc.).

#### Execution Context and Memory Block

Two important concepts or data structures for `Tensor` are the execution context
`device`, and the memory block `Block`.

Each `Tensor` is physically stored on and managed by a hardware device,
representing the execution context (CPU, GPU). Tensor math calculations are
executed on the device.

Tensor data in a `Block` instance, defined in `include/singa/core/common.h`.
`Block` owns the underlying data, while tensors take ownership on the metadata
describing the tensor, like `shape`, `strides`.

#### Tensor Math Backends

To leverage on the efficient math libraries provided by different backend
hardware devices, SINGA has one set of implementations of Tensor functions for
each supported backend.

- 'tensor_math_cpp.h' implements operations using Cpp (with CBLAS) for CppCPU
  devices.
- 'tensor_math_cuda.h' implements operations using Cuda (with cuBLAS) for
  CudaGPU devices.
- 'tensor_math_opencl.h' implements operations using OpenCL for OpenclGPU
  devices.

### Exposing C++ APIs to Python

SWIG(http://www.swig.org/) is a tool that can automatically convert C++ APIs
into Python APIs. SINGA uses SWIG to expose the C++ APIs to Python. Several
files are generated by SWIG, including `python/singa/singa_wrap.py`. The Python
modules (e.g., `tensor`, `device` and `autograd`) imports this module to call
the C++ APIs for implementing the Python classes and functions.

```python
import tensor

t = tensor.Tensor(shape=(2, 3))
```

For example, when a Python `Tensor` instance is created as above, the `Tensor`
class implementation creates an instance of the `Tensor` class defined in
`singa_wrap.py`, which corresponds to the C++ `Tensor` class. For clarity, the
`Tensor` class in `singa_wrap.py` is referred as `CTensor` in `tensor.py`.

```python
# in tensor.py
from . import singa_wrap as singa

CTensor = singa.Tensor
```

### Create New Tensor Functions

With the groundwork set by the previous description, extending tensor functions
could be done easily in a bottom up manner. For math operations, the steps are:

- Declare the new API to `tensor.h`
- Generate code using the predefined macro in `tensor.cc`, refer to
  `GenUnaryTensorFn(Abs);` as an example.
- Declare the template method/function in `tensor_math.h`
- Do the real implementation at least for CPU (`tensor_math_cpp.h`) and
  GPU(`tensor_math_cuda.h`)
- Expose the API via SWIG by adding it into `src/api/core_tensor.i`
- Define the Python Tensor API in `tensor.py` by calling the automatically
  generated function in `singa_wrap.py`
- Write unit tests where appropriate

## Python API

_work in progress_

## CPP API

_work in progress_
