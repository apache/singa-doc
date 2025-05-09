---
id: version-5.0.0_Chinese-device
title: Device
original_id: device
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Device 代表一个有内存和计算单元的硬件设备。所有的[Tensor 操作](./tensor)都由常驻
设备安排执行的，Tensor 内存也由设备的内存管理器管理。因此，需要在 Device 类中实
现内存和执行的优化。

## 特定设备

目前，SINGA 支持三种设备：

1.  CudaGPU：用于运行 Cuda 代码的 Nvidia GPU。
2.  CppCPU：用于运行 Cpp 代码的 CPU。
3.  OpenclGPU：用于运行 OpenCL 代码的 GPU 卡。

## 用法示例

下面的代码提供了创建设备的例子：

```python
from singa import device
cuda = device.create_cuda_gpu_on(0)  # use GPU card of ID 0
host = device.get_default_device()  # get the default host device (a CppCPU)
ary1 = device.create_cuda_gpus(2)  # create 2 devices, starting from ID 0
ary2 = device.create_cuda_gpus([0,2])  # create 2 devices on ID 0 and 2
```
