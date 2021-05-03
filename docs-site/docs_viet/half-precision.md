---
id: half-precision
title: Half Precision
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Half precision training mang lại các lợi ích:
- sử dụng ít bộ nhớ GPU, hỗ trợ network lớn hơn.
- training nhanh hơn. 

## Loại dữ liệu Half

### Định nghĩa dữ liệu Half
Theo tiêu chuẩn IEEE 754 chỉ rõ binary16 có các
 [định dạng](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) sau:
Sign bit: 1 bit
Exponent width: 5 bits
Significand precision: 11 bits (lưu trữ chính xác 10)

### Xử lý dữ liệu Half
Tải dữ liệu ở dạng fp32 và dễ dàng đổi sang fp16 bằng casting.
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

Các chương trình cơ bản được hỗ trợ với fp16.
```python
>>> y+y
[[1.541  0.8555 1.175 ]
 [1.567  1.409  1.3   ]], float16
```

## Training Half

### Training Half qua 3 bước
Training theo half precision được thực hiện đơn giản qua ba bước:
1. Tải dữ liệu và chuyển đổi sang half
2. Chọn loại dữ liệu cho optimizer
3. Train model như bình thường
``` python
# cast dữ liệu đầu vào sang fp16
x = load_data()
x = x.astype(np.float16)
tx = tensor.from_numpy(x)

# tải model
model = build_model()
# chuyển optimizer dtype sang fp16
sgd = opt.SGD(lr=0.1, dtype=tensor.float16)

# train như bình thường
out, loss = model(tx, ty)
```

### Ví Dụ
Tập tin ví dụ `train_cnn.py`, chạy lệnh dưới đây để train theo half.
```python
python examples/cnn/train_cnn.py cnn mnist -pfloat16
```

## Áp Dụng

### Dependency Dạng Half
Thực hiện theo dạng half được tích hợp ở C++ backend như hỗ trợ dạng half nói chung.

Để chạy trên GPU, `__half` có trên API của Cuda math. Để hỗ trợ chạy `__half` 
math, cần compile với Nvidia compute arch > 6.0 (Pascal).

### Gia tốc Nvidia Hardware: Tensor Core
Tensor Core phát hành bởi Nvidia gia tốc half precision và các 
throughput cho các hàm như GEMM(CuBlas) và convolution(CuDNN). Khi kích hoạt hàm
Tensor core, có một vài hạn chế về quy mô GEMM, 
kích thước kênh convolution, phiên bản Cuda, và phiên bản GPU (Turing hoặc mới hơn) v.v.

### Áp dụng Hàm
Hàm Half cơ bản được thực hiện trong `tensor_math_cuda.h`, bằng cách chuyên môn hoá mô hình thực hiện với half type và áp dụng low level computation.

Ví dụ, hàm GEMM được thực hiện như sau: 
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