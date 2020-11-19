---
id: version-3.1.0_Chinese-benchmark-train
title: Benchmark for Distributed Training
original_id: benchmark-train
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->


项目：我们使用深度卷积神经网络[ResNet-50](https://github.com/apache/singa/blob/master/examples/cnn/model/resnet.py)。它有50个卷积层，用于图像分类。它需要3.8个GFLOPs来通过网络处理一张图像（尺寸为224x224）。输入的图像大小为224x224。


硬件方面：我们使用的是AWS的p2.8xlarge实例，每个实例有8个Nvidia Tesla K80 GPU，共96GB GPU内存，32个vCPU，488GB主内存，10Gbps网络带宽。

衡量标准：我们衡量不同数量worker的每次迭代时间，以评估SINGA的可扩展性。Batch-size固定为每个GPU32个。采用同步训练方案。因此，有效的batch-size是`32N`，其中N是GPU的数量。我们与一个流行的开源系统进行比较，该系统采用参数服务器拓扑结构。选择第一个GPU作为服务器。

![Benchmark Experiments](assets/benchmark.png) <br/> **可扩展性测试。条形为吞吐量，折线形为通信成本。**
