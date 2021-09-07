---
id: version-3.2.0_Chinese-examples
title: Examples
original_id: examples
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

本页列出了一些使用SINGA的深度学习任务示例。源代码维护在 [Github](https://github.com/apache/singa/tree/master/examples) 上的 SINGA repo 内。对于使用SINGA Python API在CPU或单GPU上运行的例子，它们也可以在[Google Colab](https://colab.research.google.com/)上获得。你可以直接在谷歌云上运行它们，而无需在本地设置环境。下面给出了每个例子的链接。

## 图像分类

| 网络模型       | 数据集                          | 链接                                                                                                   |
| ----------- | --------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Simple CNN  | MNIST, CIFAR10, CIFAR100          | [Colab](https://colab.research.google.com/drive/1fbGUs1AsoX6bU5F745RwQpohP4bHTktq)                      |
| AlexNet     | ImageNet                          | [Cpp]()                                                                                                 |
| VGG         | ImageNet                          | [Cpp](), [Python](), [Colab](https://colab.research.google.com/drive/14kxgRKtbjPCKKsDJVNi3AvTev81Gp_Ds) |
| XceptionNet | MNIST, CIFAR10, CIFAR100          | [Python]()                                                                                              |
| ResNet      | MNIST, CIFAR10, CIFAR100, CIFAR10 | [Python](), [Colab](https://colab.research.google.com/drive/1u1RYefSsVbiP4I-5wiBKHjsT9L0FxLm9)          |
| MobileNet   | ImageNet                          | [Colab](https://colab.research.google.com/drive/1HsixqJMIpKyEPhkbB8jy7NwNEFEAUWAf)                      |

## 目标检测

| 网络模型       | 数据集    | 链接                                                                             |
| ----------- | ---------- | ---------------------------------------------------------------------------------- |
| Tiny YOLOv2 | Pascal VOC | [Colab](https://colab.research.google.com/drive/11V4I6cRjIJNUv5ZGsEGwqHuoQEie6b1T) |

## 面部及表情识别

| 模型           | 数据集                                                                                                                                                | 链接                                                                              |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| ArcFace         | Refined MS-Celeb-1M                                                                                                                                    | [Colab](https://colab.research.google.com/drive/1qanaqUKGIDtifdzEzJOHjEj4kYzA9uJC) |
| Emotion FerPlus | [Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) | [Colab](https://colab.research.google.com/drive/1XHtBQGRhe58PDi4LGYJzYueWBeWbO23r) |

## 图像生成

| 模型 | 数据集 | 链接                                                                             |
| ----- | ------- | ---------------------------------------------------------------------------------- |
| GAN   | MNIST   | [Colab](https://colab.research.google.com/drive/1f86MNDW47DJqHoIqWD1tOxcyx2MWys8L) |
| LSGAN | MNIST   | [Colab](https://colab.research.google.com/drive/1C6jNRf28vnFOI9JVM4lpkJPqxsnhxdol) |

## 机器理解

| 模型     | 数据集                                                                  | 链接                                                                             |
| ---------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Bert-Squad | [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/) | [Colab](https://colab.research.google.com/drive/1kud-lUPjS_u-TkDAzihBTw0Vqr0FjCE-) |

## 文字识别

| 模型       | 数据集 | 链接      |
| ----------- | ------- | ---------- |
| Simple LSTM | IMDB    | [python]() |

## 文本排序

| 模型  | 数据集    | 链接      |
| ------ | ----------- | ---------- |
| BiLSTM | InsuranceQA | [python]() |

## 其他

- MNIST数据集的有限玻尔兹曼机, [source](),
  [Colab](https://colab.research.google.com/drive/19996noGu9JyHHkVmp4edBGu7PJSRQKsd).
