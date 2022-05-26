---
id: version-3.3.0_Viet-device
title: Device
original_id: device
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Device dùng ở đây nghĩa là thiết bị phần cứng với bộ nhớ và các bộ phận máy
tính. Tất cả [Tensor operations](./tensor) được sắp xếp bởi các thiết bị
resident device khi chạy. Bộ nhớ của Tensor luôn luôn được quản lý bởi memory
manager của thiết bị đó. Bởi vậy việc tận dụng tối đa bộ nhớ và thực hiện được
tiến hành tại Device class.

## Các thiết bị cụ thể

Hiện tại, SINGA được chạy trên ba Device,

1.  CudaGPU cho cạc Nvidia GPU card chạy code Cuda
2.  CppCPU cho CPU chạy Cpp code
3.  OpenclGPU cho cạc GPU chạy OpenCL code

## Ví Dụ Sử Dụng

Code dưới đây là ví dụ về việc tạo device:

```python
from singa import device
cuda = device.create_cuda_gpu_on(0)  # sử dụng cạc GPU với ID 0
host = device.get_default_device()  # tạo host mặc định cho device (CppCPU)
ary1 = device.create_cuda_gpus(2)  # tạo 2 devices, bắt đầu từ ID 0
ary2 = device.create_cuda_gpus([0,2])  # tạo 2 devices với ID 0 và 2
```
