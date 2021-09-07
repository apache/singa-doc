---
id: version-3.2.0_Chinese-half-precision
title: Half Precision
original_id: half-precision
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Half precision training 优点:
- CPU内存使用低, 网络支持大。
- 训练速度快。

## Half data type

### Half data type 定义
在 IEEE 754 标准中明确binary16有如下格式：
 [format](https://en.wikipedia.org/wiki/Half-precision_floating-point_format):
Sign bit: 1 bit
Exponent width: 5 bits
Significand precision: 11 bits (10 explicitly stored)

### Half data type 运算
以fp32形式加载数据，快速转换成fp16。
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

初级运算支持fp16格式。 
```python
>>> y+y
[[1.541  0.8555 1.175 ]
 [1.567  1.409  1.3   ]], float16
```

## Training in Half

### Training in Half 三个步骤
半精度训练只需要如下三个步骤:
1. 加载数据并且转换成半精度数据
2. 设置数据优化类型
3. 启动训练模型
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

### 示例
提供示例脚本`train_cnn.py`，可执行下面的命令语句开始半精度模型训练。
```python
python examples/cnn/train_cnn.py cnn mnist -pfloat16
```

## 实现

### Half Type 依赖性
该半精度实现方式就像一半半精度模型支持的一样，是被整合在C++后端来实现的。

在GPU上跑的时候，`__half`可用在uda math API中，为了支持`__half`数学运算，需要编译Nvidia compute arch > 6.0(Pascal)


### Nvidia Hardware Acceleration: Tensor Core
Nvidia发布Tensor Core后进一步加速了半精度和倍数吞吐量的运算，如GEMM(CuBlas) and convolution(CuDNN)。要启用Tensor core的运算，在GEMM方面有一些要求，比如：卷积通道大小，Cuda版本和GPU版本（图灵或更高版本）等等。

### Implement Operations
半精度运算起初被整合在`tensor_math_cuda.h`中，专门提供半精度类型运算模版和实现方式，用来实现低数据量的计算。

示例, GEMM 运算实现如下:
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