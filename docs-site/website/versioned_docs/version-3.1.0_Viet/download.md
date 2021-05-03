---
id: version-3.1.0_Viet-downloads
title: Tải SINGA
original_id: downloads
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## Kiểm Chứng

Để kiểm chứng tập tin tar.gz đã tải, tải
[KEYS](https://www.apache.org/dist/singa/KEYS) và tập tin ASC sau đó thực hiện
các lệnh sau

```shell
% gpg --import KEYS
% gpg --verify downloaded_file.asc downloaded_file
```

Bạn có thể kiểm tra giá trị của SHA512 hoặc MD5 để xem liệu việc tải về đã hoàn
thành chưa.

## V3.1.0 (30 tháng 10 năm 2020):

- [Apache SINGA 3.1.0](http://www.apache.org/dyn/closer.cgi/singa/3.1.0/apache-singa-3.1.0.tar.gz)
  [\[SHA512\]](https://www.apache.org/dist/singa/3.1.0/apache-singa-3.1.0.tar.gz.sha512)
  [\[ASC\]](https://www.apache.org/dist/singa/3.1.0/apache-singa-3.1.0.tar.gz.asc)
- [Release Notes 3.1.0](http://singa.apache.org/docs/releases/RELEASE_NOTES_3.1.0)
- Thay đổi chung:
  - Cập nhật Tensor core:
    - Hỗ trợ tensor transformation (reshape, transpose) cho tensors có tới 6
      chiều (dimensions).
    - Áp dụn traverse_unary_transform ở Cuda backend, tương tự như CPP backend
      one.
  - Thêm hàm tensor operators vào autograd module.
  - Cải tạo lại sonnx để
    - Hỗ trợ việc tạo hàm operators từ cả layer và autograd.
    - Viết lại SingaRep để SINGA representation mạnh và nhanh hơn.
    - Thêm SONNXModel áp dụng từ Model để API và các tính năng đồng bộ với nhau.
  * Thay thế Travis CI với trình tự Github. Thêm quản lý chất lượng và độ bao
    phủ.
  * Thêm Tập lệnh compiling và packaging nhằm tạo gói wheel packages cho
    distribution.
  * Fix bugs
    - Hoàn thiện Tập lệnh training cho ví dụ về IMDB LSTM model.
    - Ổn định lại hàm Tensor operation Mult khi sử dụng Broadcasting.
    - Hàm Gaussian trong Tensor giờ có thể chạy trên Tensor với kích thước lẻ.
    - Cập nhật hàm hỗ trợ chạy thử gradients() trong autograd để tìm tham số
      gradient qua tham số python object id khi chạy thử.

## V3.0.0 (18 April 2020):

- [Apache SINGA 3.0.0](https://archive.apache.org/dist/singa/3.0.0/apache-singa-3.0.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/singa/3.0.0/apache-singa-3.0.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/singa/3.0.0/apache-singa-3.0.0.tar.gz.asc)
- [Ghi Chú Phát Hành 3.0.0](http://singa.apache.org/docs/releases/RELEASE_NOTES_3.0.0)
- Các tính năng mới và thay đổi chính,
  - Nâng cấp ONNX. Thử nghiệm nhiều ONNX models trên SINGA.
  - Thực hiện Distributed training với MPI và tối ưu hoá NCCL Communication qua
    phân bổ và nén độ dốc, và truyền tải phân khúc.
  - Xây dựng và tối ưu hoá tốc độ và bộ nhớ sử dụng graph của Computational
    graph.
  - Lập trang Tài Liệu sử dụng mới (singa.apache.org) và website tham khảo API
    (apache-singa.rtfd.io).
  - CI cho việc kiểm tra chất lượng mã code.
  - Thay thế MKLDNN bằng DNNL
  - Cập nhật APIs cho tensor để hỗ trợ hàm broadcasting.
  - Tạo autograd operators mới để hỗ trợ các ONNX models.

## Incubating v2.0.0 (20 tháng 4 năm 2019):

- [Apache SINGA 2.0.0 (incubating)](https://archive.apache.org/dist/incubator/singa/2.0.0/apache-singa-incubating-2.0.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/incubator/singa/2.0.0/apache-singa-incubating-2.0.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/2.0.0/apache-singa-incubating-2.0.0.tar.gz.asc)
- [Ghi Chú Phát Hành 2.0.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_2.0.0.html)
- Các tính năng mới và thay đổi chính,
  - Nâng cấp autograd (cho Convolution networks và recurrent networks)
  - Hỗ trợ ONNX
  - Cải thiện hàm CPP operations qua Intel MKL DNN lib
  - Thực hiện tensor broadcasting
  - Chuyển Docker images dưới tên sử dụng trong Apache
  - Cập nhật các phiên bản dependent lib trong conda-build config

## Incubating v1.2.0 (6 June 2018):

- [Apache SINGA 1.2.0 (incubating)](https://archive.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz.asc)
- [Release Notes 1.2.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_1.2.0.html)
- Các tính năng mới và thay đổi chính,
  - Thực hiện autograd (đang hỗ trợ MLP model)
  - Nâng cấp PySinga để hỗ trợ Python 3
  - Cải thiện Tensor class với mục stride
  - Nâng cấp cuDNN từ V5 sang V7
  - Thêm VGG, Inception V4, ResNet, và DenseNet cho ImageNet classification
  - Tạo alias cho gói conda packages
  - Hoàn thiện Tài liệu sử dụng bằng tiếng Trung
  - Thêm hướng dẫn chạy Singa trên Windows
  - Cập nhật compilation, CI
  - Sửa lỗi nếu có

## Incubating v1.1.0 (12 February 2017):

- [Apache SINGA 1.1.0 (incubating)](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz.asc)
- [Release Notes 1.1.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_1.1.0.html)
- Các tính năng mới và thay đổi chính,
  - Tạo Docker images (phiên bản CPU và GPU)
  - Tạo Amazon AMI cho SINGA (phiên bản CPU)
  - Tích hợp với Jenkins để tự động tạo gói Wheel và Debian (cho cài đặt), và
    cập nhật website.
  - Nâng cấp FeedFowardNet, vd., nhiều mode cho inputs và verbose để sửa lỗi
  - Thêm Concat và Slice layers
  - Mở rộng CrossEntropyLoss nhằm chấp nhật instance với nhiều labels
  - Thêm image_tool.py với phương thức image augmentation
  - Hỗ trợ tải và lưu model qua Snapshot API
  - Compile SINGA source trên Windows
  - Compile những dependent libraries bắt buộc cùng với SINGA code
  - Kích hoạt Java binding (cơ bản) cho SINGA
  - Thêm phiên bản ID trong kiểm soát tập tin
  - Thêm gói sử dụng Rafiki cung cấp RESTFul APIs
  - Thêm ví dụ pretrained từ Caffe, bao gồm GoogleNet

## Incubating v1.0.0 (8 September 2016):

- [Apache SINGA 1.0.0 (incubating)](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz.asc)
- [Release Notes 1.0.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_1.0.0.html)
- Các tính năng mới và thay đổi chính,
  - Tạo Tensor nhằm hỗ trợ nhiều model khác nhau.
  - Tạo Device để chạy trên các thiết bị phần cứng khác nhau, bao gồm CPU,
    (Nvidia/AMD) GPU và FPGA (sẽ thử nghiệm ở các phiên bản mới).
  - Thay thế GNU autotool với cmake khi compilation.
  - Hỗ trợ Mac OS
  - Cải thiện Python binding, bao gồm cài đặt và lập trình.
  - Tạo thêm nhiều deep learning models, bao gồm VGG và ResNet
  - Thêm IO classes để đọc/viết tập tin và mã hoá/giải mã dữ liệu
  - Các thành phần network communication mới trực tiếp từ Socket.
  - Cudnn V5 với Dropout và RNN layers.
  - Thay thế công cụ xây dựng website từ maven sang Sphinx
  - Tích hợp Travis-CI

## Incubating v0.3.0 (20 April 2016):

- [Apache SINGA 0.3.0 (incubating)](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz.asc)
- [Release Notes 0.3.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_0.3.0.html)
- Các tính năng mới và thay đổi chính,
  - Training trên nhóm máy GPU: cho phép training các deep learning models trên
    một nhóm máy GPU
  - Cải thiện Python wrapper khiến cho job configuration trở nên dễ dàng, bao
    gồm neural net và thuật toán SGD.
  - Thêm cập nhật SGD updaters mới, bao gồm Adam, AdaDelta và AdaMax.
  - Cài đặt cần ít dependent libraries hơn cho mỗi node training.
  - Đa dạng training với CPU và GPU.
  - Hỗ trợ cuDNN V4.
  - Tìm nạp trước dữ liệu.
  - Sửa lỗi nếu có.

## Incubating v0.2.0 (14 January 2016):

- [Apache SINGA 0.2.0 (incubating)](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz.asc)
- [Release Notes 0.2.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_0.2.0.html)
- Các tính năng mới và thay đổi chính,
  - Training trên GPU cho phép training các models phức tạp trên một node với
    nhiều card GPU.
  - Chia nhỏ Hybrid neural net hỗ trợ dữ liệu và model song song cùng lúc.
  - Cải thiện Python wrapper khiến cho job configuration trở nên dễ dàng, bao
    gồm neural net và thuật toán SGD.
  - Áp dụng RNN model và thuật toán BPTT để hỗ trợ các ứng dụng dựa trên RNN
    models, e.g., GRU.
  - Tích hợp các phần mêm đám mây bao gồm Mesos, Docker và HDFS.
  - Cung cấp hình ảnh cấu trúc neural net và thông tin layer, hỗ trợ việc sửa
    lỗi.
  - Hàm Linear algebra và các hàm ngẫu nhiên không dùng Blobs và chỉ điểm dữ
    liệu thô.
  - Tạo layers mới, bao gồm SoftmaxLayer, ArgSortLayer, DummyLayer, RNN layers
    và cuDNN layers.
  - Cập nhật Layer class để chứa nhiều data/grad Blobs.
  - Trích xuất các features và thử nghiệm hiệu quả cho dữ liệu mới bằng cách tải
    các tham số model đã được train từ trước.
  - Thêm Store class cho hàm IO operations.

## Incubating v0.1.0 (8 October 2015):

- [Apache SINGA 0.1.0 (incubating)](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz.asc)
- [Amazon EC2 image](https://console.aws.amazon.com/ec2/v2/home?region=ap-southeast-1#LaunchInstanceWizard:ami=ami-b41001e6)
- [Release Notes 0.1.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_0.1.0.html)
- Các thay đổi chính gồm có,
  - Cài đặt sử dụng tiện ích GNU build
  - Tập lệnh cho job management với zookeeper
  - Lập trình model dựa trên NeuralNet và trích xuất Layer.
  - Kết cấu hệ thống dựa trên Worker, Server và Stub.
  - Training models từ ba model khác nhau, là feed-forward models, energy models
    và RNN models.
  - Đồng bộ và không đồng bộ và không đồng bộ distributed training frameworks sử
    dụng CPU
  - Điểm kiểm tra (Checkpoint) và khôi phục
  - Kiểm tra đơn vị sử dụng gtest
