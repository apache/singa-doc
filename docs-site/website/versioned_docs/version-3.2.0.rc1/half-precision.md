---
id: version-3.2.0.rc1-half-precision
title: Half Precision
original_id: half-precision
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Half precision training could bring benefits:
- using less GPU memory, supporting larger network. 
- training faster. 

## Half data type

### Half data type definition
The IEEE 754 standard specifies a binary16 as having the following
 [format](https://en.wikipedia.org/wiki/Half-precision_floating-point_format):
Sign bit: 1 bit
Exponent width: 5 bits
Significand precision: 11 bits (10 explicitly stored)

### Half data type operation
Load data in fp32 and easily convert to fp16 by casting.
```python
>>> from singa import tensor, device
>>> dev = device.create_cuda_gpu()
>>> x = tensor.random((2,3),dev)
>>> x
[[0.7703407  0.42764223 0.5872884 ]
 [0.78362167 0.70469785 0.64975065]], float32
>>> y = x.as_type(tensor.float16)
>>> y
[[0.7705 0.4277 0.5874]
 [0.7837 0.7046 0.65  ]], float16
```

Primary operations are supported in fp16.
```python
>>> y+y
[[1.541  0.8555 1.175 ]
 [1.567  1.409  1.3   ]], float16
```

## Training in Half

### Training in Half three step
Training in half precision could be done easily in three steps:
1. Load data and convert to half
2. Set data type of optimizer
3. Train model as usual
``` python
# cast input data to fp16
x = load_data()
x = x.astype(np.float16)
tx = tensor.from_numpy(x)

# load model
model = build_model()
# set optimizer dtype to fp16
sgd = opt.SGD(lr=0.1, dtype=tensor.float16)

# train as usual
out, loss = model(tx, ty)
```

### Example
An example script is `train_cnn.py`, run below command to train in half.
```python
python examples/cnn/train_cnn.py cnn mnist -pfloat16
```

## Implementation

### Half Type Dependency
This half implementation is integrated in C++ backend as general half type 
support.

To run on GPU, `__half` is available in Cuda math API. To support `__half` 
math operation, it is required to compile against Nvidia compute arch > 6.0
 (Pascal).

### Nvidia Hardware Acceleration: Tensor Core
Tensor Core released by Nvidia further accelerates half precision and multiples 
throughput for operations like GEMM(CuBlas) and convolution(CuDNN). To enable 
Tensor core operation, there are a few restriction on GEMM dimensions, 
convolution channel size, Cuda version, and GPU version(Turing or later) and etc.

### Implement Operations
Half operations are primarily implemented in `tensor_math_cuda.h`, by specializing
operation template with half type and implementation the low level computation.

For example, GEMM operation is implemented as:
```c++
template <>
void GEMM<half_float::half, lang::Cuda>(const half_float::half alpha,
                                        const Tensor& A, const Tensor& B,
                                        const half_float::half beta, Tensor* C,
                                        Context* ctx) {
  // ...
  CUBLAS_CHECK(cublasGemmEx(handle, transb, transa, ncolB, nrowA, ncolA,
                           alphaPtr, BPtr, Btype, ldb, APtr, Atype, lda,
                           betaPtr, CPtr, Ctype, ldc, computeType, algo));
  // ...
}
```