---
id: version-3.1.0_Viet-tensor
title: Tensor
original_id: tensor
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Mỗi thực thể Tensor instance là một array đa chiều được đặt trong một thực thể
Device. Thực thể Tensor lưu các biến và cung cấp phép tính đại số tuyến tính cho
các loại thiết bị phần cứng khác nhau mà không cần người dùng để ý. Lưu ý rằng
người dùng cần đảm bảo các toán hạng tensor được đặt ở cùng một thiết bị ngoại
trừ hàm copy.

## Cách sử dụng Tensor

### Tạo Tensor

```python
>>> import numpy as np
>>> from singa import tensor
>>> tensor.from_numpy( np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32) )
[[1. 0. 0.]
 [0. 1. 0.]]
```

### Chuyển sang numpy

```python
>>> a = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
>>> tensor.from_numpy(a)
[[1. 0. 0.]
 [0. 1. 0.]]
>>> tensor.to_numpy(tensor.from_numpy(a))
array([[1., 0., 0.],
       [0., 1., 0.]], dtype=float32)
```

### Phương pháp Tensor

```python
>>> t = tensor.from_numpy(a)
>>> t.transpose([1,0])
[[1. 0.]
 [0. 1.]
 [0. 0.]]
```

biến đổi `tensor` tới 6 chữ số

```python
>>> a = tensor.random((2,3,4,5,6,7))
>>> a.shape
(2, 3, 4, 5, 6, 7)
>>> a.reshape((2,3,4,5,7,6)).transpose((3,2,1,0,4,5)).shape
(5, 4, 3, 2, 7, 6)
```

### Phương pháp số học Tensor

`tensor` được đánh giá trong thời gian thực.

```python
>>> t + 1
[[2. 1. 1.]
 [1. 2. 1.]]
>>> t / 5
[[0.2 0.  0. ]
 [0.  0.2 0. ]]
```

`tensor` tạo số học:

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

`tensor` broadcasting on matrix multiplication (GEMM)

```python
>>> from singa import tensor
>>> a = tensor.random((2,2,2,3))
>>> b = tensor.random((2,3,4))
>>> tensor.mult(a,b).shape
(2, 2, 2, 4)
```

### Hàm lập trình Tensor Functions

Hàm Functions trong `singa.tensor` tạo ra đối tượng `tensor` mới sau khi áp dụng
phép tính trong hàm function.

```python
>>> tensor.log(t+1)
[[0.6931472 0.        0.       ]
 [0.        0.6931472 0.       ]]
```

### Tensor ở các thiết bị khác nhau

`tensor` được tạo ra trên máy chủ (CPU) từ ban đầu; nó cũng được tạo ra trên các
thiết bị phần cứng khác nhau bằng cách cụ thể hoá `device`. Một `tensor` có thể
chuyển giữa `device`s qua hàm `to_device()` function.

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

### Dùng Tensor để train MLP

```python

"""
  Đoạn mã trích từ examples/mlp/module.py
"""

label = get_label()
data = get_data()

dev = device.create_cuda_gpu_on(0)
sgd = opt.SGD(0.05)

# định nghĩa tensor cho dữ liệu và nhãn đầu vào
tx = tensor.Tensor((400, 2), dev, tensor.float32)
ty = tensor.Tensor((400,), dev, tensor.int32)
model = MLP(data_size=2, perceptron_size=3, num_classes=2)

# đính model vào graph
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

Đầu ra:

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

## Áp dụng Tensor

Mục trước chỉ ra cách sử dụng chung của `Tensor`, việc áp dụng cụ thể được đưa
ra sau đây. Đầu tiên, sẽ giới thiệu việc thiết lập tensors Python và C++. Phần
sau sẽ nói về cách frontend (Python) và backend (C++) kết nối với nhau và cách
để mở rộng chúng.

### Python Tensor

`Tensor` của lớp Python, được định nghĩa trong `python/singa/tensor.py`, cung
cấp cách dùng tensor ở tầng cao, để thực hiện việc vận hành deep learning (qua
[autograd](./autograd)), cũng như là quản lý dữ liệu bởi người dùng cuối.

Hoạt động cơ bản của nó là gói xung quanh các phương pháp C++ tensor, cả phương
pháp số học (như `sum`) và không số học (như `reshape`). Một vài phép số học cao
cấp về sau được giới thiệu và áp dụng sử dụng thuần Python tensor API, như
`tensordot`. Python Tensor APIs có thể sử dụng để thực hiện dễ dàng các phép
tính neural network phức tạp với các phương pháp methods linh hoạt có sẵn.

### C++ Tensor

`Tensor` lớp C++, được định nghĩa trong `include/singa/core/tensor.h`, về cơ bản
quản lý bộ nhớ nắm giữ dữ liệu, và cung cấp APIs tầm thấp cho các hàm thực hiện
tensor. Đồng thời nó cũng cung cấp các phương pháp số học đa dạng (như `matmul`)
bằng cách gói các chương trình backends khác nhau (CUDA, BLAS, cuBLAS, v.v.).

#### Văn bản thực hiện và Khoá Bộ nhớ

Hai khái niệm hay cấu trúc dữ liệu quan trọng của `Tensor` là việc áp dụng
`device`, và khoá bộ nhớ `Block`.

Mỗi `Tensor` được lưu theo nghiã đen và quản lý bởi một thiết bị phần cứng, thể
hiện theo nghĩa thực hành (CPU, GPU). Tính toán Tensor được thực hiện trên thiết
bị.

Dữ liệu Tensor trong hàm `Block`, được định nghĩa trong
`include/singa/core/common.h`. `Block` chứa dữ liệu cơ sở, trong khi tensors
chịu trách nhiệm về lý lịch dữ liệu metadata mô tả tensor, như `shape`,
`strides`.

#### Tensor Math Backends

Để tận dụng các thư viện chương trình toán hiệu quả cung cấp bởi backend từ các
thiết bị phần cứng khác nhau, SINGA cung cấp một bộ Tensor functions cho mỗi
backend được hỗ trợ.

- 'tensor_math_cpp.h' áp dụng vận hành sử dụng Cpp (với CBLAS) cho thiết bị
  CppCPU.
- 'tensor_math_cuda.h' áp dụng vận hành sử dụng Cuda (với cuBLAS) cho thiết bị
  CudaGPU.
- 'tensor_math_opencl.h' áp dụng vận hành sử dụng OpenCL cho thiết bị OpenclGPU.

### Trình bày C++ APIs qua Python

SWIG(http://www.swig.org/) là công cụ có thể tự động qui đổi C++ APIs sang
Python APIs. SINGA sử dụng SWIG để trình bày C++ APIs sang Python. Một vài tệp
tin được tạo bởi SWIG, bao gồm `python/singa/singa_wrap.py`. Các Python mô-đun
(như, `tensor`, `device` và `autograd`) nhập mô-đun để gọi C++ APIs để áp dụng
hàm và lớp Python.

```python
import tensor

t = tensor.Tensor(shape=(2, 3))
```

Ví dụ, khi một Python `Tensor` instance được tạo ra ở trên, việc áp dụng
`Tensor` class tạo ra một instance của `Tensor` class định nghĩa trong
`singa_wrap.py`, tương ứng với C++ `Tensor` class. Rõ ràng hơn, `Tensor` class
trong `singa_wrap.py` để chỉ `CTensor` trong `tensor.py`.

```python
# trong tensor.py
from . import singa_wrap as singa

CTensor = singa.Tensor
```

### Tạo Hàm Tensor Functions mới

Với nền tảng được mô tả phía trên, mở rộng hàm tensor functions có thể dễ dàng
thực hiện từ dưới lên, Với các phép toán, các bước làm như sau:

- Khai báo API mới cho `tensor.h`
- Tạo mã code sử dụng tiền tố xác định trước trong `tensor.cc`, lấy
  `GenUnaryTensorFn(Abs);` làm ví dụ.
- Khai báo theo mẫu method/function trong `tensor_math.h`
- Thực hiện áp dụng ít nhất cho CPU (`tensor_math_cpp.h`) và
  GPU(`tensor_math_cuda.h`)
- Trình API qua SWIG bằng cách thêm nó vào `src/api/core_tensor.i`
- Định nghĩa Python Tensor API trong `tensor.py` bằng cách tự động gọi hàm
  function được tạo trong `singa_wrap.py`
- Viết unit tests khi phù hợp

## Python API

_đang cập nhật_

## CPP API

_đang cập nhật_
