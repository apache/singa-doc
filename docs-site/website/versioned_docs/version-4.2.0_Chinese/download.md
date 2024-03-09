---
id: version-4.2.0_Chinese-downloads
title: Download SINGA
original_id: downloads
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## Verify

要验证下载的 tar.gz 文件，下载[KEYS](https://www.apache.org/dist/singa/KEYS)和
ASC 文件，然后执行以下命令:

```shell
% gpg --import KEYS
% gpg --verify downloaded_file.asc downloaded_file
```

你也可以检查 SHA512 或 MD5 值判断下载是否完成。

## V3.0.0 (18 April 2020):

- [Apache SINGA 3.0.0](http://www.apache.org/dyn/closer.cgi/singa/3.0.0/apache-singa-3.0.0.tar.gz)
  [\[SHA512\]](https://www.apache.org/dist/singa/3.0.0/apache-singa-3.0.0.tar.gz.sha512)
  [\[ASC\]](https://www.apache.org/dist/singa/3.0.0/apache-singa-3.0.0.tar.gz.asc)
- [Release Notes 3.0.0](releases/RELEASE_NOTES_3.0.0)
- 新特性及重要更新：
  - 增强了 ONNX。在 SINGA 中测试了多种 ONNX 模型。
  - 使用 MPI 和 NCCL Communication 进行分布式训练，通过梯度稀疏化和压缩以及分块
    传输进行了优化。
  - 计算图的构建，利用图优化了速度和内存。
  - 新的文档网站（singa.apache.org）和 API 参考网站（apache-singa.rtfd.io）。
  - 使用 CI 实现代码质量检查。
  - 将 MKLDNN 替换为 DNNL。
  - 更新 Tensor API 以支持广播操作。
  - 实现了支持 ONNX 模型的新的 autograd 操作符。

## 孵化版本（incubating） v2.0.0 (20 April 2019):

- [Apache SINGA 2.0.0 (incubating)](http://www.apache.org/dyn/closer.cgi/incubator/singa/2.0.0/apache-singa-incubating-2.0.0.tar.gz)
  [\[SHA512\]](https://www.apache.org/dist/incubator/singa/2.0.0/apache-singa-incubating-2.0.0.tar.gz.sha512)
  [\[ASC\]](https://www.apache.org/dist/incubator/singa/2.0.0/apache-singa-incubating-2.0.0.tar.gz.asc)
- [Release Notes 2.0.0 (incubating)](releases/RELEASE_NOTES_2.0.0.html)
- 新特性及重要更新：
  - 增强了 autograd 功能(适用于卷积网络和循环网络)。
  - 支持 ONNX。
  - 通过英特尔 MKL DNN lib 改进 CPP 操作。
  - 实现 tensor 广播。
  - 在 Apache 用户名下移动 Docker 镜像。
  - 更新 conda-build 配置中依赖的 lib 版本。

## 孵化版本（incubating） v1.2.0 (6 June 2018):

- [Apache SINGA 1.2.0 (incubating)](https://archive.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz.asc)
- [Release Notes 1.2.0 (incubating)](releases/RELEASE_NOTES_1.2.0.html)
- 新特性及重要更新：
  - 实现了 autograd（目前支持 MLP 模式）。
  - 升级 PySinga 以支持 Python 3
  - 改进 Tensor 类的 stride 范围。
  - 将 cuDNN 从 V5 升级到 V7。
  - 增加 VGG、Inception V4、ResNet 和 DenseNet 进行 ImageNet 分类。
  - 为 conda 包创建别名
  - 完整的中文文档
  - 添加在 Windows 上运行 Singa 的说明
  - 更新编译，CI
  - 修复一些错误

## 孵化版本（incubating） v1.1.0 (12 February 2017):

- [Apache SINGA 1.1.0 (incubating)](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz.asc)
- [Release Notes 1.1.0 (incubating)](releases/RELEASE_NOTES_1.1.0.html)
- 新特性和重要更新：
  - 创建 Docker 镜像(CPU 和 GPU 版本)
  - 为 SINGA 创建 Amazon AMI（CPU 版）。
  - 集成 Jenkins，自动生成 Wheel 和 Debian 包（用于安装），并更新网站。
  - 增强 FeedFowardNet，例如，多输入和用于调试的 verbose 模式。
  - 添加 Concat 和 Slice 层。
  - 优化 CrossEntropyLoss 以接受带有多个标签的实例。
  - 添加包含图像增强方法的 image_tool.py。
  - 支持通过快照 API 加载和保存模型。
  - 在 Windows 上编译 SINGA 源代码。
  - 将必要依赖库与 SINGA 代码一起编译。
  - 为 SINGA 启用 Java binding（基本）。
  - 在 checkpoint 文件中添加版本 ID。
  - 增加 Rafiki 工具包以提供 RESTFul APIs。
  - 添加了从 Caffe 预训练的例子，包括 GoogleNet。

## 孵化版本（incubating） v1.0.0 (8 September 2016):

- [Apache SINGA 1.0.0 (incubating)](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz.asc)
- [Release Notes 1.0.0 (incubating)](releases/RELEASE_NOTES_1.0.0.html)
- 新特性和重要更新：
  - 创建 Tensor 概念，用于支持更多的机器学习模型。
  - 创建 Device 概念，用于运行在不同的硬件设备上，包括 CPU，(Nvidia/AMD) GPU 和
    FPGA (将在以后的版本中测试)。
  - 用 cmake 取代 GNU autotool 进行编译。
  - 支持 Mac OS。
  - 改进 Python binding，包括安装和编程。
  - 更多的深度学习模型，包括 VGG 和 ResNet。
  - 更多的 IO 类用于读取/写入文件和编码/解码数据。
  - 直接基于 Socket 的新网络通信组件。
  - Cudnn V5，包含 Dropout 和 RNN 层。
  - 将网页构建工具从 maven 更换为 Sphinx。
  - 整合 Travis-CI。

## 孵化版本（incubating） v0.3.0 (20 April 2016):

- [Apache SINGA 0.3.0 (incubating)](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz.asc)
- [Release Notes 0.3.0 (incubating)](releases/RELEASE_NOTES_0.3.0.html)
- 新特性和重要更新：
  - 在 GPU 集群上进行训练，可以在 GPU 集群上进行深度学习模型的训练。
  - 改进 Python wrapper 简化配置工作，包括神经网络和 SGD 算法。
  - 新增 SGD 更新器，包括 Adam、AdaDelta 和 AdaMax。
  - 安装时减少了单节点训练的依赖库。
  - 使用 CPU 和 GPU 进行异构训练。
  - 支持 cuDNN V4。
  - 数据预取。
  - 修复一些 bug。

## 孵化版本（incubating） v0.2.0 (14 January 2016):

- [Apache SINGA 0.2.0 (incubating)](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz.asc)
- [Release Notes 0.2.0 (incubating)](releases/RELEASE_NOTES_0.2.0.html)
- 新特性和重要更新：
  - 在 GPU 上进行训练，可以在一个节点上用多个 GPU 卡训练复杂的模型。
  - 混合神经网分区支持数据和模型同时并行。
  - Python wrapper 简化配置，包括神经网络和 SGD 算法。
  - 实现了 RNN 模型和 BPTT 算法，支持基于 RNN 模型的应用，如 GRU。
  - 云软件集成，包括 Mesos、Docker 和 HDFS。
  - 可视化神经网结构和层信息，以便优化调试。
  - 针对 Blobs 和原始数据指针的线性代数函数和随机函数。
  - 添加新的层，包括 SoftmaxLayer、ArgSortLayer、DummyLayer、RNN 层和 cuDNN 层。
  - 更新 Layer 类以携带多个数据/梯度 Blobs。
  - 通过加载之前训练的模型参数，提取特征并测试新数据的性能。
  - 为 IO 操作添加 Store 类。

## Incubating v0.1.0 (8 October 2015):

- [Apache SINGA 0.1.0 (incubating)](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz.asc)
- [Amazon EC2 image](https://console.aws.amazon.com/ec2/v2/home?region=ap-southeast-1#LaunchInstanceWizard:ami=ami-b41001e6)
- [Release Notes 0.1.0 (incubating)](releases/RELEASE_NOTES_0.1.0.html)
- 新特性和重要更新：
  - 允许使用 GNU 构建工具进行安装。
  - 实现用 zookeeper 进行工作管理的脚本。
  - 实现基于 NeuralNet 和 Layer 的编程模型。
  - 实现基于 Worker、Server 和 Stub 的系统架构。
  - 训练模型来自三种不同的模型类别，即前馈模型、能量模型和 RNN 模型。
  - 使用 CPU 的同步和异步分布式训练框架。
  - checkpoint 文件生成和恢复。
  - 使用 gtest 进行单元测试。
