---
id: examples
title: Ví Dụ
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Phần này đưa ra một vài ví dụ về việc thực hiện deep learning sử dụng SINGA. Mã nguồn (source code) được cung cấp trong SINGA repo trên
[Github](https://github.com/apache/singa/tree/master/examples). Có thể tham khảo các ví dụ sử dụng SINGA Python APIs trên CPU hoặc một GPU trên
[Google Colab](https://colab.research.google.com/). Bạn có thể trực tiếp chạy thử trên 
Google Cloud mà không cần tạo local environment. Đường dẫn tới mỗi ví dụ được cung cấp dưới đây.

## Image Classification

| Model       | Dataset                           | Đường dẫn                                                                                                   |
| ----------- | --------------------------------- | ------------------------------------------------------------------------------------------------------- |
| CNN cơ bản  | MNIST, CIFAR10, CIFAR100          | [Colab](https://colab.research.google.com/drive/1fbGUs1AsoX6bU5F745RwQpohP4bHTktq)                      |
| AlexNet     | ImageNet                          | [Cpp]()                                                                                                 |
| VGG         | ImageNet                          | [Cpp](), [Python](), [Colab](https://colab.research.google.com/drive/14kxgRKtbjPCKKsDJVNi3AvTev81Gp_Ds) |
| XceptionNet | MNIST, CIFAR10, CIFAR100          | [Python]()                                                                                              |
| ResNet      | MNIST, CIFAR10, CIFAR100, CIFAR10 | [Python](), [Colab](https://colab.research.google.com/drive/1u1RYefSsVbiP4I-5wiBKHjsT9L0FxLm9)          |
| MobileNet   | ImageNet                          | [Colab](https://colab.research.google.com/drive/1HsixqJMIpKyEPhkbB8jy7NwNEFEAUWAf)                      |

## Object Detection

| Model       | Dataset    | Đường dẫn                                                                                |
| ----------- | ---------- | ---------------------------------------------------------------------------------- |
| Tiny YOLOv2 | Pascal VOC | [Colab](https://colab.research.google.com/drive/11V4I6cRjIJNUv5ZGsEGwqHuoQEie6b1T) |

## Nhận diện Khuôn mặt và Cảm xúc 

| Model           | Dataset                                                                                                                                                | Đường dẫn                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| ArcFace         | Refined MS-Celeb-1M                                                                                                                                    | [Colab](https://colab.research.google.com/drive/1qanaqUKGIDtifdzEzJOHjEj4kYzA9uJC) |
| Emotion FerPlus | [Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) | [Colab](https://colab.research.google.com/drive/1XHtBQGRhe58PDi4LGYJzYueWBeWbO23r) |

## Image Generation

| Model | Dataset | Đường dẫn                                                                                |
| ----- | ------- | ---------------------------------------------------------------------------------- |
| GAN   | MNIST   | [Colab](https://colab.research.google.com/drive/1f86MNDW47DJqHoIqWD1tOxcyx2MWys8L) |
| LSGAN | MNIST   | [Colab](https://colab.research.google.com/drive/1C6jNRf28vnFOI9JVM4lpkJPqxsnhxdol) |

## Machine Comprehension

| Model      | Dataset                                                                   | Đường dẫn                                                                                |
| ---------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Bert-Squad | [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/) | [Colab](https://colab.research.google.com/drive/1kud-lUPjS_u-TkDAzihBTw0Vqr0FjCE-) |

## Text Classification

| Model       | Dataset | Đường dẫn       |
| ----------- | ------- | ---------- |
| Simple LSTM | IMDB    | [python]() |

## Text Ranking

| Model  | Dataset     | Đường dẫn      |
| ------ | ----------- | ---------- |
| BiLSTM | InsuranceQA | [python]() |

## Misc.

- Restricted Boltzmann Machine sử dụng dữ liệu MNIST, [nguồn](),
  [Colab](https://colab.research.google.com/drive/19996noGu9JyHHkVmp4edBGu7PJSRQKsd).
