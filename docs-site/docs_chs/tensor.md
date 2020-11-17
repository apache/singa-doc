---
id: tensor
title: Tensor
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

每个Tensor实例都是分配在特定设备实例上的多维数组。Tensor实例可以存储变量，并在不同类型的硬件设备上提供线性代数运算，而无需用户察觉。需要注意的是，除了复制函数外，用户需要确保张量操作数分配在同一个设备上。

## Tensor用法

### 创建Tensor

```python
>>> import numpy as np
>>> from singa import tensor
>>> tensor.from_numpy( np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32) )
[[1. 0. 0.]
 [0. 1. 0.]]
```

### 转换到numpy

```python
>>> a = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
>>> tensor.from_numpy(a)
[[1. 0. 0.]
 [0. 1. 0.]]
>>> tensor.to_numpy(tensor.from_numpy(a))
array([[1., 0., 0.],
       [0., 1., 0.]], dtype=float32)
```

### Tensor方法

```python
>>> t = tensor.from_numpy(a)
>>> t.transpose([1,0])
[[1. 0.]
 [0. 1.]
 [0. 0.]]
```

`Tensor`变换，最多支持6维。

```python
>>> a = tensor.random((2,3,4,5,6,7))
>>> a.shape
(2, 3, 4, 5, 6, 7)
>>> a.reshape((2,3,4,5,7,6)).transpose((3,2,1,0,4,5)).shape
(5, 4, 3, 2, 7, 6)
```

### Tensor算术方法

`tensor`是实时计算的：

```python
>>> t + 1
[[2. 1. 1.]
 [1. 2. 1.]]
>>> t / 5
[[0.2 0.  0. ]
 [0.  0.2 0. ]]
```

`tensor` broadcasting运算:

```python
>>> a
[[1. 2. 3.]
 [4. 5. 6.]]
>>> b
[[1. 2. 3.]]
>>> a + b
[[2. 4. 6.]
 [5. 7. 9.]]
>>> a * b
[[ 1.  4.  9.]
 [ 4. 10. 18.]]
>>> a / b
[[1.  1.  1. ]
 [4.  2.5 2. ]]
>>> a/=b # inplace operation
>>> a
[[1.  1.  1. ]
 [4.  2.5 2. ]]
```

`tensor` broadcasting矩阵乘法：

```python
>>> from singa import tensor
>>> a = tensor.random((2,2,2,3))
>>> b = tensor.random((2,3,4))
>>> tensor.mult(a,b).shape
(2, 2, 2, 4)
```

### Tensor函数

`singa.tensor`模块中的函数在应用函数中定义的变换后返回新的`Tensor`对象。

```python
>>> tensor.log(t+1)
[[0.6931472 0.        0.       ]
 [0.        0.6931472 0.       ]]
```

### Tensor在不同Devices上

`tensor`默认在主机(CPU)上创建；也可以通过指定设备在不同的硬件`device`上创建。`tensor`可以通过`to_device()`函数在`device`之间移动。

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

### 使用Tensor训练MLP

```python

"""
  code snipet from examples/mlp/module.py
"""

label = get_label()
data = get_data()

dev = device.create_cuda_gpu_on(0)
sgd = opt.SGD(0.05)

# define tensor for input data and label
tx = tensor.Tensor((400, 2), dev, tensor.float32)
ty = tensor.Tensor((400,), dev, tensor.int32)
model = MLP(data_size=2, perceptron_size=3, num_classes=2)

# attached model to graph
model.set_optimizer(sgd)
model.compile([tx], is_train=True, use_graph=True, sequential=False)
model.train()

for i in range(1001):
    tx.copy_from_numpy(data)
    ty.copy_from_numpy(label)
    out, loss = model(tx, ty, 'fp32', spars=None)

    if i % 100 == 0:
        print("training loss = ", tensor.to_numpy(loss)[0])
```

输出:

```bash
$ python3 examples/mlp/module.py
training loss =  0.6158037
training loss =  0.52852553
training loss =  0.4571422
training loss =  0.37274635
training loss =  0.30146334
training loss =  0.24906921
training loss =  0.21128304
training loss =  0.18390492
training loss =  0.16362564
training loss =  0.148164
training loss =  0.13589878
```

## Tensor实现

上一节介绍了`Tensor`的一般用法，下面将介绍其底层的实现。首先，将介绍Python和C++ tensors的设计。后面会讲到前端（Python）和后端（C++）如何连接，如何扩展。

### Python Tensor

Python类`Tensor`，定义在`python/singa/tensor.py`中，提供了高层的张量操作，用于实现深度学习操作（通过[autograd](./autograd)），以及终端用户的数据管理。

它主要是通过简单地封装C++张量方法来工作，包括算术方法（如`sum`）和非算术方法（如`reshape`）。一些高级的算术运算以后会引入，并使用纯Python的张量API来实现，如`tensordot`。Python Tensor API可以利用灵活的方法轻松实现复杂的神经网络操作。

### C++ Tensor

C++类`Tensor`，定义在`include/singa/core/tensor.h`中，主要是管理存放数据的内存，并提供低级的API用于张量操作。同时，它还通过封装不同的后端（CUDA、BLAS、cuBLAS等）提供各种算术方法（如`matmul`）。

#### 执行背景和内存块

Tensor的两个重要概念或者说数据结构是执行背景`device`，和内存块`Block`。

每个`Tensor`物理上存储在一个硬件设备上，并由硬件设备管理，代表执行背景（CPU、GPU），Tensor的数学计算是在设备上执行的。

Tensor数据在`Block`实例中，定义在`include/singa/core/common.h`中。`Block`拥有底层数据，而tensors则在描述tensor的元数据上拥有所有权，比如`shape`、`stride`。

#### Tensor数学后端

为了利用不同后端硬件设备提供的高效数学库，SINGA为每个支持的后端设备提供了一套Tensor函数的实现。

- 'tensor_math_cpp.h'为CppCPU设备使用Cpp（带CBLAS）实现操作。
- 'tensor_math_cuda.h'为CudaGPU设备实现了使用Cuda(带cuBLAS)的操作。
- 'tensor_math_opencl.h'为OpenclGPU设备实现了使用OpenCL的操作。

### 将C++ APIs暴露给Python


SWIG(http://www.swig.org/)是一个可以自动将C++ API转换为Python API的工具。SINGA使用SWIG将C++ APIs公开到Python中。SWIG会生成几个文件，包括`python/singa/singa_wrap.py`。Python模块(如`tensor`、`device`和`autograd`)导入这个模块来调用C++ API来实现Python类和函数。

```python
import tensor

t = tensor.Tensor(shape=(2, 3))
```

例如，当按上面的方法创建Python `Tensor`实例时，`Tensor`类的实现会创建一个在`singa_wrap.py`中定义的Tensor类的实例，它对应于C++ `Tensor`类。为了清楚起见，`singa_wrap.py`中的`Tensor`类在`tensor.py`中被称为`CTensor`。

```python
# in tensor.py
from . import singa_wrap as singa

CTensor = singa.Tensor
```

### 创建新的Tensor函数


有了前面的描述所奠定的基础，扩展张量函数可以用自下而上的方式轻松完成。对于数学运算，其步骤是：

- 在`tensor.h`中声明新的API。
- 使用 `tensor.cc` 中预定义的宏生成代码，参考 `GenUnaryTensorFn(Abs);` 。
- 在`tensor_math.h`中声明template 方法/函数。
- 至少在CPU(`tensor_math_cpp.h`)和GPU(`tensor_math_cuda.h`)中进行真正的实现。
- 将 API 加入到 `src/api/core_tensor.i` 中，通过 SWIG 公开 API。
- 通过调用 `singa_wrap.py` 中自动生成的函数，在 `tensor.py` 中定义 Python Tensor API。
- 在适当的地方编写单元测试

## Python API

_进行中_

## CPP API

_进行中_
