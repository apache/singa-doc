---
id: version-3.3.0_Viet-install-win
title: Cách cài SINGA trên Windows
original_id: install-win
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Quá trình cài đặt SINGA từ nguồn sử dụng Microsoft Windows bao gồm bốn bước: cài
đặt thư viện dependencies, cài SINGA từ nguồn, (không bắt buộc) cài đặt python
module và (không bắt buộc) chạy thử unit tests.

## Cài đặt thư viện dependencies

Bạn có thể tạo một thư mục để cài đặt thư viện dependencies.

Các thư viện dependencies bao gồm:

- Compiler và IDE
  - Visual Studio. Công cụ biên tập mã này miễn phí và có thể được dùng trong
    việc cài đặt SINGA. https://www.visualstudio.com/
- CMake
  - Có thể tải về qua http://cmake.org/
  - Đảm bảo đường dẫn khả thi của cmake nằm trong đường dẫn chương trình system
    path, hoặc sử dụng đường dẫn đầy đủ khi gọi hàm cmake.
- SWIG

  - Có thể tải từ http://swig.org/
  - Đảm bảo đường dẫn khả thi của swig nằm trong đường dẫn chương trình system
    path, hoặc sử dụng đường dẫn đầy đủ khi gọi hàm swig. Sử dụng các phiên bản
    cập nhật như 3.0.12.

- Protocol Buffers
  - Tải các phiên bản phù hợp như 2.6.1:
    https://github.com/google/protobuf/releases/tag/v2.6.1 .
  - Tải cả hai tệp protobuf-2.6.1.zip và protoc-2.6.1-win32.zip .
  - Giải nén cả hai tệp trên trong thư mục thư viện dependencies. Thêm đường dẫn
    khả thi cho protoc vào system path, hoặc sử dụng đường dẫn đầy đủ khi gọi
    hàm này.
  - Mở Visual Studio solution có thể tìm trong thư mục vsproject.
  - Thay đổi cài đặt thiết lập Settings tới Release and x64.
  - Cài đặt libprotobuf project.
- Openblas

  - Tải phiên bản nguồn phù hợp như 0.2.20 từ http://www.openblas.net
  - Giải nén nguồn trong thư mục thư viện dependencies.
  - Nếu bạn không có chương trình Perl, tải perl environment như Strawberry Perl
    (http://strawberryperl.com/)
  - Cài đặt Visual Studio solution bằng lệnh sau từ thư mục nguồn:

  ```bash
  cmake -G "Visual Studio 15 2017 Win64"
  ```

  - Mở Visual Studio solution và thay đổi cấu hình cài đặt cho Release and x64.
  - Cài libopenblas project

- Google glog
  - Tải phiên bản phù hợp như 0.3.5 từ https://github.com/google/glog/releases
  - Giải nén nguồn trong thư mục thư viện dependencies.
  - Mở Visual Studio solution.
  - Thay đổi cài đặt thiết lập Settings tới Release and x64.
  - Cài đặt libglog project

## Cài SINGA từ nguồn

- Tải code nguồn của SINGA
- Cấu tạo các tệp protobuf:

  - Tới thư mục src/proto

  ```shell
  mkdir python_out
  protoc.exe *.proto --python_out python_out
  ```

- Tạo swig interfaces cho C++ và Python: Tới mục src/api

  ```shell
  swig -python -c++ singa.i
  ```

- Tạo Visual Studio solution cho SINGA: Đi tới thư mục nguồn SINGA

  ```shell
  mkdir build
  cd build
  ```

- Gọi hàm cmake và thêm đường dẫn vào trong system của bạn, tương tự như ví dụ
  sau:

  ```shell
  cmake -G "Visual Studio 15 2017 Win64" ^
    -DGLOG_INCLUDE_DIR="D:/WinSinga/dependencies/glog-0.3.5/src/windows" ^
    -DGLOG_LIBRARIES="D:/WinSinga/dependencies/glog-0.3.5/x64/Release" ^
    -DCBLAS_INCLUDE_DIR="D:/WinSinga/dependencies/openblas-0.2.20/lapack-netlib/CBLAS/include" ^
    -DCBLAS_LIBRARIES="D:/WinSinga/dependencies/openblas-0.2.20/lib/RELEASE" ^
    -DProtobuf_INCLUDE_DIR="D:/WinSinga/dependencies/protobuf-2.6.1/src" ^
    -DProtobuf_LIBRARIES="D:/WinSinga/dependencies/protobuf-2.6.1/vsprojects/x64/Release" ^
    -DProtobuf_PROTOC_EXECUTABLE="D:/WinSinga/dependencies/protoc-2.6.1-win32/protoc.exe" ^
    ..
  ```

- Mở generated solution trong Visual Studio
- Thay đổi cài đặt thiết lập Settings tới Release and x64
- Thêm tệp tin singa_wrap.cxx từ src/api tới singa_objects project
- Trong singa_objects project, mở Additional Include Directories.
- Thêm Python bao gồm đường dẫn
- Thêm numpy bao gồm đường dẫn
- Thêm protobuf bao gồm đường dẫn
- Trong định nghĩa preprocessor của singa_objects project, thêm USE_GLOG
- Sử dụng singa_objects project

- Trong singa project:

  - thêm singa_wrap.obj vào Thư viện Object
  - đổi tên mục target thành \_singa_wrap
  - đổi định dạng tệp target thành .pyd
  - đổi định dạng cấu hình sang Dynamic Library (.dll)
  - đi tới Additional Library Directories và thêm đường dẫn vào các thư viện
    python, openblas, protobuf và glog
  - đi tới các thư viện Dependencies bổ sung để thêm libopenblas.lib,
    libglog.lib và libprotobuf.lib

- tạo singa project

## Cài đặt Python module

- Đổi `_singa_wrap.so` thành `_singa_wrap.pyd` trong build/python/setup.py
- Copy các tệp tin trong `src/proto/python_out` sang `build/python/singa/proto`

- Không bắt buộc, tạo và kích hoạt virtual environment:

  ```shell
  mkdir SingaEnv
  virtualenv SingaEnv
  SingaEnv\Scripts\activate
  ```

- tới thư mục build/python và chạy:

  ```shell
  python setup.py install
  ```

- Sử dụng \_singa_wrap.pyd, libglog.dll và libopenblas.dll bằng cách thêm chúng
  vào đường dẫn hoặc copy chúng vào thư mục gói chương trình singa trong gói
  python site-packages

- Xác nhận SINGA đã được cài đặt bằng cách chạy:

  ```shell
  python -c "from singa import tensor"
  ```

Tham khảo video quá trình cài đặt tại đây:

[![youtube video](https://img.youtube.com/vi/cteER7WeiGk/0.jpg)](https://www.youtube.com/watch?v=cteER7WeiGk)

## Chạy Unit Tests

- Trong thư mục test, tạo Visual Studio solution:

  ```shell
  cmake -G "Visual Studio 15 2017 Win64"
  ```

- Mở generated solution trong Visual Studio.

- Thay đổi cài đặt thiết lập Settings tới Release and x64.

- Tạo glog project.

- Trong mục test_singa project:

  - Thêm USE_GLOG vào Định nghĩa Preprocessor.
  - Trong Additional Include Directories, thêm đường dẫn của GLOG_INCLUDE_DIR,
    CBLAS_INCLUDE_DIR và Protobuf_INCLUDE_DIR đã được dùng ở bước 2 bên trên.
    Đồng thời tạo và tạo/bao gồm các thư mục.
  - Đi tới Additional Library Directories và thêm đường dẫn vào thư viện
    openblas, protobuf và glog. Thêm build/src/singa_objects.dir/Release.
  - Tới Thư viện Dependencies bổ sung và thêm libopenblas.lib, libglog.lib và
    libprotobuf.lib. Sửa tên của hai thư viện: gtest.lib và singa_objects.lib.

- Cài test_singa project.

- Sử dụng libglog.dll và libopenblas.dll bằng cách thêm chúng vào đường dẫn hoặc
  copy chúng vào thư mục test/release.

- Tiến hành unit tests bằng cách

  - Từ dòng lệnh:

  ```shell
  test_singa.exe
  ```

  - Từ Visual Studio:
    - ấn chuột phải tại test_singa project và chọn 'Set as StartUp Project'.
    - Từ mục Debug menu, chọn 'Start Without Debugging'

Tham khảo video hướng dẫn chạy unit tests tại đây:

[![youtube video](https://img.youtube.com/vi/393gPtzMN1k/0.jpg)](https://www.youtube.com/watch?v=393gPtzMN1k)

## Cài đặt hỗ trợ GPU với CUDA

Trong mục này, chúng tôi sẽ mở rộng các bước phía trên để sử dụng GPU.

### Cài đặt thư viện Dependencies

Ngoài các thư viện dependencies ở mục 1 phía trên, chúng ta cần:

- CUDA

  Tải phiên bản phù hợp như 9.1 từ https://developer.nvidia.com/cuda-downloads .
  Đảm bảo bạn cài đặt Visual Studio integration module.

- cuDNN

  Tải phiên bản phù hợp như 7.1 từ https://developer.nvidia.com/cudnn

- cnmem:

  - Tải phiên bản mới nhất tại https://github.com/NVIDIA/cnmem
  - Tạo Visual Studio solution:

  ```shell
  cmake -G "Visual Studio 15 2017 Win64"
  ```

  - Mở generated solution trong Visual Studio.
  - Thay đổi cài đặt thiết lập Settings tới Release and x64.
  - Tạo cnmem project.

### Cài SINGA từ nguồn

- Gọi hàm cmake và thêm đường dẫn vào trong system của bạn, tương tự như ví dụ
  sau:
  ```shell
  cmake -G "Visual Studio 15 2017 Win64" ^
    -DGLOG_INCLUDE_DIR="D:/WinSinga/dependencies/glog-0.3.5/src/windows" ^
    -DGLOG_LIBRARIES="D:/WinSinga/dependencies/glog-0.3.5/x64/Release" ^
    -DCBLAS_INCLUDE_DIR="D:/WinSinga/dependencies/openblas-0.2.20/lapack-netlib/CBLAS/include" ^
    -DCBLAS_LIBRARIES="D:/WinSinga/dependencies/openblas-0.2.20/lib/RELEASE" ^
    -DProtobuf_INCLUDE_DIR="D:/WinSinga/dependencies/protobuf-2.6.1/src" ^
    -DProtobuf_LIBRARIES="D:\WinSinga/dependencies/protobuf-2.6.1/vsprojects/x64/Release" ^
    -DProtobuf_PROTOC_EXECUTABLE="D:/WinSinga/dependencies/protoc-2.6.1-win32/protoc.exe" ^
    -DCUDNN_INCLUDE_DIR=D:\WinSinga\dependencies\cudnn-9.1-windows10-x64-v7.1\cuda\include ^
    -DCUDNN_LIBRARIES=D:\WinSinga\dependencies\cudnn-9.1-windows10-x64-v7.1\cuda\lib\x64 ^
    -DSWIG_DIR=D:\WinSinga\dependencies\swigwin-3.0.12 ^
    -DSWIG_EXECUTABLE=D:\WinSinga\dependencies\swigwin-3.0.12\swig.exe ^
    -DUSE_CUDA=YES ^
    -DCUDNN_VERSION=7 ^
    ..
  ```

* Tạo swig interfaces cho C++ và Python: Tới mục src/api

  ```shell
  swig -python -c++ singa.i
  ```

* Mở generated solution trong Visual Studio

* Thay đổi cài đặt thiết lập Settings tới Release and x64.

#### Tạo singa_objects

- Thêm tệp tin singa_wrap.cxx từ src/api tới singa_objects project
- Trong singa_objects project, mở Additional Include Directories.
- Thêm Python bao gồm đường dẫn
- Thêm numpy bao gồm đường dẫn
- Thêm protobuf bao gồm đường dẫn
- Bổ sung bao gồm đường dẫn cho CUDA, cuDNN và cnmem
- Trong định nghĩa preprocessor của singa_objects project, thêm USE_GLOG,
  USE_CUDA và USE_CUDNN. Xoá DISABLE_WARNINGS.
- Tạo singa_objects project

#### Tạo singa-kernel

- Tạo một Visual Studio project mới dưới dạng "CUDA 9.1 Runtime". Đặt tên dạng
  như singa-kernel.
- project này chứa sẵn một tệp tin là kernel.cu. Xoá tệp tin này khỏi project.
- Thêm tệp tin này: src/core/tensor/math_kernel.cu
- Trong mục cài đặt project:

  - Đặt Platform Toolset sang dạng "Visual Studio 2015 (v140)"
  - Đổi Configuration Type sang " Static Library (.lib)"
  - Trong mục Include Directories, thêm vào build/include.

- Tạo singa-kernel project

#### Cài đặt singa

- Trong singa project:

  - thêm singa_wrap.obj vào Object Libraries
  - đổi tên target thành \_singa_wrap
  - đổi định dạng target sang .pyd
  - đổi định dạng cấu hình sang Dynamic Library (.dll)
  - đi tới Additional Library Directories và thêm đường dẫn vào các thư viện
    python, openblas, protobuf và glog
  - thêm các đường dẫn thư viện cho singa-kernel, cnmem, cuda và cudnn.
  - đi tới các thư viện Dependencies bổ sung để thêm libopenblas.lib,
    libglog.lib và libprotobuf.lib
  - Đồng thời thêm: singa-kernel.lib, cnmem.lib, cudnn.lib, cuda.lib ,
    cublas.lib, curand.lib và cudart.lib.

- tạo singa project

### Cài đặt Python module

- Đổi tên \_singa_wrap.so sang \_singa_wrap.pyd trong mục build/python/setup.py
- Copy các tệp tin trong src/proto/python_out sang build/python/singa/proto

- Không bắt buộc, tạo và kích hoạt virtual environment:

  ```shell
  mkdir SingaEnv
  virtualenv SingaEnv
  SingaEnv\Scripts\activate
  ```

- tới thư mục build/python và chạy:

  ```shell
  python setup.py install
  ```

- Sử dụng \_singa_wrap.pyd, libglog.dll, libopenblas.dll, cnmem.dll, CUDA
  Runtime (e.g. cudart64_91.dll) và cuDNN (e.g. cudnn64_7.dll) bằng cách thêm
  chúng vào đường dẫn hoặc copy chúng vào thư mục gói chương trình singa trong
  gói python site-packages

- Xác nhận SINGA đã được cài đặt bằng cách chạy:

  ```shell
  python -c "from singa import device; dev = device.create_cuda_gpu()"
  ```

Tham khảo video hướng dẫn cho mục này tại đây:

[![youtube video](https://img.youtube.com/vi/YasKVjRtuDs/0.jpg)](https://www.youtube.com/watch?v=YasKVjRtuDs)

### Run Unit Tests

- Trong thư mục tests, tạo Visual Studio solution:

  ```shell
  cmake -G "Visual Studio 15 2017 Win64"
  ```

- Mở solution được tạo trong Visual Studio, hoặc thêm project vào singa solution
  đã được tạo ở bước 5.2

- Thay đổi cài đặt thiết lập Settings tới Release and x64.

- Tạo glog project.

- Trong test_singa project:

  - Thêm USE_GLOG; USE_CUDA; USE_CUDNN vào Định Nghĩa Preprocessor.
  - Trong Thư viện Dependencies bổ sung, thêm đường dẫn của GLOG_INCLUDE_DIR,
    CBLAS_INCLUDE_DIR và Protobuf_INCLUDE_DIR đã được sử dụng tại bước 5.2 ở
    trên. Và thêm build, build/include, CUDA và cuDNN bao gồm thư mục.
  - Tới Thư viện Dependencies bổ sung và thêm đường dẫn của thư viện openblas,
    protobuf và glog. Và thêm đường dẫn thư viện của
    build/src/singa_objects.dir/Release, singa-kernel, cnmem, CUDA và cuDNN.
  - Tới Thư viện Dependencies bổ sung và thêm libopenblas.lib; libglog.lib;
    libprotobuf.lib; cnmem.lib; cudnn.lib; cuda.lib; cublas.lib; curand.lib;
    cudart.lib; singa-kernel.lib. Sửa tên của hai thư viện: gtest.lib và
    singa_objects.lib.

* Tạo test_singa project.

* Sử dụng libglog.dll, libopenblas.dll, cnmem.dll, cudart64_91.dll và
  cudnn64_7.dll bằng cách thêm chúng vào đường dẫn hoặc copy chúng vào thư mục
  test/release.

- Tiến hành unit tests bằng cách:

  - Từ dòng lệnh:

    ```shell
    test_singa.exe
    ```

  - Từ Visual Studio:
    - ấn chuột phải tại test_singa project và chọn 'Set as StartUp Project'.
    - Từ mục Debug menu, chọn 'Start Without Debugging'

Tham khảo video hướng dẫn chạy unit tests tại đây:

[![youtube video](https://img.youtube.com/vi/YOjwtrvTPn4/0.jpg)](https://www.youtube.com/watch?v=YOjwtrvTPn4)
