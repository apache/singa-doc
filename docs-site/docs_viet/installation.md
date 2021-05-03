---
id: installation
title: Cài đặt
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## Sử dụng Conda

Conda là gói quản lý chương trình cho Python, CPP và các phần mềm khác. 

Hiện nay SINGA có gói conda packages dùng cho Linux và MacOSX.
[Miniconda3](https://conda.io/miniconda.html) được khuyến khích dùng với SINGA. 
Sau khi cài đặt miniconda, thực hiện các lệnh sau để cài đặt
SINGA.

1. Cho CPU
   [![Mở trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ntkhi-Z6XTR8WYPXiLwujHd2dOm0772V?usp=sharing)

```shell
$ conda install -c nusdbsystem -c conda-forge singa-cpu
```

2. GPU với CUDA và cuDNN (yêu cầu CUDA driver >=384.81)
   [![Mở trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1do_TLJe18IthLOnBOsHCEe-FFPGk1sPJ?usp=sharing)

```shell
$ conda install -c nusdbsystem -c conda-forge singa-gpu
```

3. Cài đặt SINGA với phiên bản cụ thể. Câu lệnh sau cho danh sách tất cả gói chương trình SINGA đang hoạt động.

```shell
$ conda search -c nusdbsystem singa

Loading channels: done
# Name                       Version           Build  Channel
singa                      3.1.0.rc2        cpu_py36  nusdbsystem
singa                      3.1.0.rc2 cudnn7.6.5_cuda10.2_py36  nusdbsystem
singa                      3.1.0.rc2 cudnn7.6.5_cuda10.2_py37  nusdbsystem
```

<!--- > Lưu ý rằng việc sử dụng nightly built images không được khuyến khích ngoại trừ trong quá trình phát triển và kiểm định. Khuyến khích việc sử dụng phiên bản phát hành ổn định. -->

Dùng câu lệnh sau để cài đặt một phiên bản SINGA cụ thể,

```shell
$ conda install -c nusdbsystem -c conda-forge singa=X.Y.Z=cpu_py36
```

Nếu không có lỗi khi chạy

```shell
$ python -c "from singa import tensor"
```

thì bạn đã cài đặt SINGA thành công. 

## Sử dụng Pip

1. Cho CPU
   [![Mở trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17RA056Brwk0vBQTFaZ-l9EbqwADO0NA9?usp=sharing)

```bash
pip install singa -f http://singa.apache.org/docs/next/wheel-cpu.html --trusted-host singa.apache.org
```

Bạn có thể cài đặt một phiên bản SINGA cụ thể sử dụng `singa==<version>`, thay thông tin
`<version>`, v.d, `3.1.0`. Xem danh sách các phiên bản SINGA đang hoạt động ở đường dẫn. 

Để cài đặt phiên bản phát triển mới nhất, thay đường dẫn bằng 
http://singa.apache.org/docs/next/wheel-cpu-dev.html

2. GPU với CUDA và cuDNN
   [![Mở trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1W30IPCqj5fG8ADAQsFqclaCLyIclVcJL?usp=sharing)

```bash
pip install singa -f http://singa.apache.org/docs/next/wheel-gpu.html --trusted-host singa.apache.org
```

Bạn có thể thiết lập phiên bản SINGA và CUDA, như
`singa==3.1.0+cuda10.2`. Danh sách tổ hợp phiên bản SINGA với CUDA được cung cấp trong đường dẫn.

Để cài đặt phiên bản phát triển mới nhất, thay đường dẫn bằng 
http://singa.apache.org/docs/next/wheel-gpu-dev.html

Lưu ý: phiên bản Python của Python environment trong máy của bạn sẽ được sử dụng để tìm gói wheel tương ứng. Ví dụ, nếu bạn sử dụng Python 3.6, thì gói wheel kết cấu trong Python 3.6 sẽ được pip chọn để cài đặt. 
Thực tế, tên của tệp tin wheel bao gồm phiên bản SINGA, phiên bản CUDA và Python. Vì thế, `pip` biết tệp tin wheel nào để tải và cài đặt. 

Tham khảo chú thích ở phần đầu của tệp tin `setup.py` về cách tạo các gói
wheel packages.

## Sử dụng Docker

Cài đặt Docker vào máy chủ local theo
[hướng dẫn](https://docs.docker.com/install/). Thêm người dùng vào
[nhóm docker](https://docs.docker.com/install/linux/linux-postinstall/) để chạy câu lệnh docker mà không cần dùng `sudo`.

1. Cho CPU.

```shell
$ docker run -it apache/singa:X.Y.Z-cpu-ubuntu16.04 /bin/bash
```

2. Với GPU. Cài đặt 
   [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) sau khi cài
   Docker.

```shell
$ nvidia-docker run -it apache/singa:X.Y.Z-cuda9.0-cudnn7.4.2-ubuntu16.04 /bin/bash
```

3. Xem danh sách toàn bộ SINGA Docker images (tags), tại
   [trang web docker hub](https://hub.docker.com/r/apache/singa/). Với mỗi docker
   image, tag được đặt tên theo 

```shell
version-(cpu|gpu)[-devel]
```

| Tag       | Mô tả                      | Ví dụ giá trị                                                                                                                                                             |
| --------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `phiên bản` | phiên bản SINGA                    | '2.0.0-rc0', '2.0.0', '1.2.0'                                                                                                                                             |
| `cpu`     | image không thể sử dụng cho GPUs     | 'cpu'                                                                                                                                                                     |
| `gpu`     | image có thể sử dụng cho Nvidia GPUs | 'gpu', or 'cudax.x-cudnnx.x' e.g., 'cuda10.0-cudnn7.3'                                                                                                                    |
| `devel`   | chỉ số phát triển       | nếu không có, gói SINGA Python package chỉ được cài đặt cho runtime; nếu có, environment cũng được tạo ra, bạn có thể kết cấu lại SINGA từ nguồn tại '/root/singa' |
| `OS`      | cho biết phiên bản OS       | 'ubuntu16.04', 'ubuntu18.04'                                                                                                                                              |

## Từ nguồn

Bạn có thể [tạo và cài đặt SINGA](build.md) từ mã code nguồn sử dụng các công cụ tạo chương trình hoặc conda-build, trên hệ điều hành máy chủ cục bộ (local host os) hay trong Docker container.

## Câu Hỏi Thường Gặp

- Q: Lỗi khi chạy `from singa import tensor`

  A: Kiểm tra chi tiết lỗi từ

  ```shell
  python -c  "from singa import _singa_wrap"
  # tới thưu mục chứa _singa_wrap.so
  ldd path to _singa_wrap.so
  python
  >> import importlib
  >> importlib.import_module('_singa_wrap')
  ```

  Thư mục chứa `_singa_wrap.so` thường ở
  `~/miniconda3/lib/python3.7/site-packages/singa`. Thông thường, lỗi này được gây ra bởi sự không tương thích hoặc thiếu các thư viện dependent libraries, v.d cuDNN hay
  protobuf. Cách giải quyết là tạo một virtual environment mới và cài đặt SINGA trong environment đó, v.d,

  ```shell
  conda create -n singa
  conda activate singa
  conda install -c nusdbsystem -c conda-forge singa-cpu
  ```

- Q: Khi sử dụng virtual environment, mỗi khi tôi cài SINGA, numpy cũng tự động bị cài lại. Tuy nhiên, numpy không được sử dụng khi chạy `import numpy`

  A: Lỗi này có thể do biến `PYTHONPATH` environment lẽ ra phải để trống trong khi bạn sử dụng virtual environment để tránh mâu thuẫn với đường dẫn của virtual environment.

- Q: Khi chạy SINGA trên Mac OS X, tôi gặp lỗi "Fatal Python error:
  PyThreadState_Get: no current thread Abort trap: 6"

  A: Lỗi này thường xảy ra khi bạn có nhiều phiên bản Python trong hệ thống, v.d, bản của OS và bản được cài bởi Homebrew.
  Bản Python dùng cho SINGA phải giống với bản Python interpreter. Bạn có thể kiểm tra interpreter của mình bằng  `which python` và kiểm tra bản Python dùng cho SINGA
  qua `otool -L <path to _singa_wrap.so>`. Vấn đề này được giải quyết nếu
  SINGA được cài qua conda.
