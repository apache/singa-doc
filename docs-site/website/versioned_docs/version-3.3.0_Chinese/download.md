---
id: version-3.3.0_Chinese-downloads
title: Download SINGA
original_id: downloads
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## Verify

要验证下载的tar.gz文件，下载[KEYS](https://www.apache.org/dist/singa/KEYS)和ASC文件，然后执行以下命令:

```shell
% gpg --import KEYS
% gpg --verify downloaded_file.asc downloaded_file
```

你也可以检查SHA512或MD5值判断下载是否完成。

## V3.0.0 (18 April 2020):

- [Apache SINGA 3.0.0](http://www.apache.org/dyn/closer.cgi/singa/3.0.0/apache-singa-3.0.0.tar.gz)
  [\[SHA512\]](https://www.apache.org/dist/singa/3.0.0/apache-singa-3.0.0.tar.gz.sha512)
  [\[ASC\]](https://www.apache.org/dist/singa/3.0.0/apache-singa-3.0.0.tar.gz.asc)
- [Release Notes 3.0.0](releases/RELEASE_NOTES_3.0.0)
- 新特性及重要更新：
  - 增强了ONNX。在SINGA中测试了多种ONNX模型。
  - 使用 MPI 和 NCCL Communication进行分布式训练，通过梯度稀疏化和压缩以及分块传输进行了优化。
  - 计算图的构建，利用图优化了速度和内存。
  - 新的文档网站（singa.apache.org）和API参考网站（apache-singa.rtfd.io）。
  - 使用CI实现代码质量检查。
  - 将MKLDNN替换为DNNL。
  - 更新Tensor API以支持广播操作。
  - 实现了支持ONNX模型的新的autograd操作符。

## 孵化版本（incubating） v2.0.0 (20 April 2019):

- [Apache SINGA 2.0.0 (incubating)](http://www.apache.org/dyn/closer.cgi/incubator/singa/2.0.0/apache-singa-incubating-2.0.0.tar.gz)
  [\[SHA512\]](https://www.apache.org/dist/incubator/singa/2.0.0/apache-singa-incubating-2.0.0.tar.gz.sha512)
  [\[ASC\]](https://www.apache.org/dist/incubator/singa/2.0.0/apache-singa-incubating-2.0.0.tar.gz.asc)
- [Release Notes 2.0.0 (incubating)](releases/RELEASE_NOTES_2.0.0.html)
- 新特性及重要更新：
  - 增强了autograd功能(适用于卷积网络和循环网络)。
  - 支持ONNX。
  - 通过英特尔MKL DNN lib改进CPP操作。
  - 实现tensor广播。
  - 在Apache用户名下移动Docker镜像。
  - 更新conda-build配置中依赖的lib版本。

## 孵化版本（incubating） v1.2.0 (6 June 2018):

- [Apache SINGA 1.2.0 (incubating)](https://archive.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz.asc)
- [Release Notes 1.2.0 (incubating)](releases/RELEASE_NOTES_1.2.0.html)
- 新特性及重要更新：
  - 实现了autograd（目前支持MLP模式）。
  - 升级PySinga以支持Python 3
  - 改进Tensor类的stride范围。
  - 将cuDNN从V5升级到V7。
  - 增加VGG、Inception V4、ResNet和DenseNet进行ImageNet分类。
  - 为conda包创建别名
  - 完整的中文文档
  - 添加在Windows上运行Singa的说明
  - 更新编译，CI
  - 修复一些错误

## 孵化版本（incubating） v1.1.0 (12 February 2017):

- [Apache SINGA 1.1.0 (incubating)](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz.asc)
- [Release Notes 1.1.0 (incubating)](releases/RELEASE_NOTES_1.1.0.html)
- 新特性和重要更新：
  - 创建Docker镜像(CPU和GPU版本)
  - 为SINGA创建Amazon AMI（CPU版）。
  - 集成Jenkins，自动生成Wheel和Debian包（用于安装），并更新网站。
  - 增强FeedFowardNet，例如，多输入和用于调试的verbose模式。
  - 添加Concat和Slice层。
  - 优化CrossEntropyLoss以接受带有多个标签的实例。
  - 添加包含图像增强方法的image_tool.py。
  - 支持通过快照API加载和保存模型。
  - 在Windows上编译SINGA源代码。
  - 将必要依赖库与SINGA代码一起编译。
  - 为SINGA启用Java binding（基本）。
  - 在checkpoint文件中添加版本ID。
  - 增加Rafiki工具包以提供RESTFul APIs。
  - 添加了从Caffe预训练的例子，包括GoogleNet。

## 孵化版本（incubating） v1.0.0 (8 September 2016):

- [Apache SINGA 1.0.0 (incubating)](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz.asc)
- [Release Notes 1.0.0 (incubating)](releases/RELEASE_NOTES_1.0.0.html)
- 新特性和重要更新：
  - 创建Tensor概念，用于支持更多的机器学习模型。
  - 创建Device概念，用于运行在不同的硬件设备上，包括CPU，(Nvidia/AMD) GPU 和 FPGA (将在以后的版本中测试)。
  - 用 cmake 取代 GNU autotool 进行编译。
  - 支持 Mac OS。
  - 改进Python binding，包括安装和编程。
  - 更多的深度学习模型，包括VGG和ResNet。
  - 更多的IO类用于读取/写入文件和编码/解码数据。
  - 直接基于Socket的新网络通信组件。
  - Cudnn V5，包含Dropout和RNN层。
  - 将网页构建工具从maven更换为Sphinx。
  - 整合Travis-CI。

## 孵化版本（incubating） v0.3.0 (20 April 2016):

- [Apache SINGA 0.3.0 (incubating)](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz.asc)
- [Release Notes 0.3.0 (incubating)](releases/RELEASE_NOTES_0.3.0.html)
- 新特性和重要更新：
  - 在GPU集群上进行训练，可以在GPU集群上进行深度学习模型的训练。
  - 改进Python wrapper简化配置工作，包括神经网络和SGD算法。
  - 新增SGD更新器，包括Adam、AdaDelta和AdaMax。
  - 安装时减少了单节点训练的依赖库。
  - 使用CPU和GPU进行异构训练。
  - 支持 cuDNN V4。
  - 数据预取。
  - 修复一些bug。

## 孵化版本（incubating） v0.2.0 (14 January 2016):

- [Apache SINGA 0.2.0 (incubating)](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz.asc)
- [Release Notes 0.2.0 (incubating)](releases/RELEASE_NOTES_0.2.0.html)
- 新特性和重要更新：
  - 在GPU上进行训练，可以在一个节点上用多个GPU卡训练复杂的模型。
  - 混合神经网分区支持数据和模型同时并行。
  - Python wrapper简化配置，包括神经网络和SGD算法。
  - 实现了RNN模型和BPTT算法，支持基于RNN模型的应用，如GRU。
  - 云软件集成，包括Mesos、Docker和HDFS。
  - 可视化神经网结构和层信息，以便优化调试。
  - 针对Blobs和原始数据指针的线性代数函数和随机函数。
  - 添加新的层，包括SoftmaxLayer、ArgSortLayer、DummyLayer、RNN层和cuDNN层。
  - 更新Layer类以携带多个数据/梯度Blobs。
  - 通过加载之前训练的模型参数，提取特征并测试新数据的性能。
  - 为IO操作添加Store类。

## Incubating v0.1.0 (8 October 2015):

- [Apache SINGA 0.1.0 (incubating)](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz.asc)
- [Amazon EC2 image](https://console.aws.amazon.com/ec2/v2/home?region=ap-southeast-1#LaunchInstanceWizard:ami=ami-b41001e6)
- [Release Notes 0.1.0 (incubating)](releases/RELEASE_NOTES_0.1.0.html)
- 新特性和重要更新：
  - 允许使用GNU构建工具进行安装。
  - 实现用zookeeper进行工作管理的脚本。
  - 实现基于NeuralNet和Layer的编程模型。
  - 实现基于Worker、Server和Stub的系统架构。
  - 训练模型来自三种不同的模型类别，即前馈模型、能量模型和RNN模型。
  - 使用CPU的同步和异步分布式训练框架。
  - checkpoint文件生成和恢复。
  - 使用gtest进行单元测试。
