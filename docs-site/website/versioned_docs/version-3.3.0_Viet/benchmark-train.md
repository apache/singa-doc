---
id: version-3.3.0_Viet-benchmark-train
title: Benchmark cho Distributed Training
original_id: benchmark-train
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Tải công việc: chúng tôi sử dụng Mạng nơ-ron tích chập sâu (deep convolutional
neural network),
[ResNet-50](https://github.com/apache/singa/blob/master/examples/cnn/model/resnet.py)
làm ứng dụng. ResNet-50 có 50 lớp tích chập (convolution layers) để phân loại
hình ảnh. Nó đòi hỏi 3.8 GFLOPs để đưa vào một hình ảnh (kích thước ảnh 224x224)
qua mạng lưới. Kích thước ảnh đầu vào là 224x224.

Phần cứng: chúng tôi sử dụng máy p2.8xlarge từ AWS, mỗi máy gồm 8 Nvidia Tesla
K80 GPUs, bộ nhớ tổng cộng 96 GB GPU, 32 vCPU, 488 GB main memory, 10 Gbps
network bandwidth.

Metric: chúng tôi tính thời gian mỗi bước cho mỗi workers để đánh giá khả năng
mở rộng của SINGA. Kích thước của mỗi nhóm được cố định ở 32 mỗi GPU. Phương
thức training đồng bộ (Synchronous training scheme) được áp dụng. Vì thế, kích
thước nhóm hiệu quả là $32N$, trong đó N là số máy GPUs. Chúng tôi so sánh với
một hệ thống mở được dùng phổ biến có sử dụng tham số server cấu trúc liên kết.
Máy GPU đầu tiên được chọn làm server.

![Thí nghiệm Benchmark](assets/benchmark.png) <br/> **Kiểm tra khả năng mở rộng.
Bars được dùng cho thông lượng (throughput); lines dùng cho lượng kết nối
(communication cost).**
