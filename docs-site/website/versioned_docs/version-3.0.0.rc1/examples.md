---
id: version-3.0.0.rc1-examples
title: Examples
original_id: examples
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

This page lists some example deep learning tasks using SINGA. The source code is
maintained inside SINGA repo on
[Github](https://github.com/apache/singa/tree/master/examples). For examples
running on CPU or single GPU using SINGA Python APIs, they are also available on
[Google Colab](https://colab.research.google.com/). You can run them directly on
Google Cloud without setting up the environment locally. The link to each
example is given below.

## Image Classification

| Model       | Dataset                           | Links                                                                                                   |
| ----------- | --------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Simple CNN  | MNIST, CIFAR10, CIFAR100          | [Colab](https://colab.research.google.com/drive/1fbGUs1AsoX6bU5F745RwQpohP4bHTktq)                      |
| AlexNet     | ImageNet                          | [Cpp]()                                                                                                 |
| VGG         | ImageNet                          | [Cpp](), [Python](), [Colab](https://colab.research.google.com/drive/14kxgRKtbjPCKKsDJVNi3AvTev81Gp_Ds) |
| XceptionNet | MNIST, CIFAR10, CIFAR100          | [Python]()                                                                                              |
| ResNet      | MNIST, CIFAR10, CIFAR100, CIFAR10 | [Python](), [Colab](https://colab.research.google.com/drive/1u1RYefSsVbiP4I-5wiBKHjsT9L0FxLm9)          |
| MobileNet   | ImageNet                          | [Colab](https://colab.research.google.com/drive/1HsixqJMIpKyEPhkbB8jy7NwNEFEAUWAf)                      |

## Object Detection

| Model       | Dataset    | Links                                                                              |
| ----------- | ---------- | ---------------------------------------------------------------------------------- |
| Tiny YOLOv2 | Pascal VOC | [Colab](https://colab.research.google.com/drive/11V4I6cRjIJNUv5ZGsEGwqHuoQEie6b1T) |

## Face and Emotion Recognition

| Model           | Dataset                                                                                                                                                | Links                                                                              |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| ArcFace         | Refined MS-Celeb-1M                                                                                                                                    | [Colab](https://colab.research.google.com/drive/1qanaqUKGIDtifdzEzJOHjEj4kYzA9uJC) |
| Emotion FerPlus | [Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) | [Colab](https://colab.research.google.com/drive/1XHtBQGRhe58PDi4LGYJzYueWBeWbO23r) |

## Image Generation

| Model | Dataset | Links                                                                              |
| ----- | ------- | ---------------------------------------------------------------------------------- |
| GAN   | MNIST   | [Colab](https://colab.research.google.com/drive/1f86MNDW47DJqHoIqWD1tOxcyx2MWys8L) |
| LSGAN | MNIST   | [Colab](https://colab.research.google.com/drive/1C6jNRf28vnFOI9JVM4lpkJPqxsnhxdol) |

## Machine Comprehension

| Model      | Dataset                                                                   | Links                                                                              |
| ---------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Bert-Squad | [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/) | [Colab](https://colab.research.google.com/drive/1kud-lUPjS_u-TkDAzihBTw0Vqr0FjCE-) |

## Misc.

- Restricted Boltzmann Machine over the MNIST dataset, [source](),
  [Colab](https://colab.research.google.com/drive/19996noGu9JyHHkVmp4edBGu7PJSRQKsd).
