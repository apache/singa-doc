---
id: version-4.2.0_Chinese-install-win
title: Build SINGA on Windows
original_id: install-win
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

在 Microsoft Windows 上从源码构建 SINGA 的过程有四个部分：安装依赖关系、构建
SINGA 源码、（可选择）安装 python 模块和（可选择）运行单元测试。

## 安装依赖项

你可以创建一个文件夹来构建依赖关系。

使用到的依赖项有：

- 编译器和 IDE：
  - Visual Studio，社区版是免费的，可以用来构建 SINGA。
    https://www.visualstudio.com/
- CMake
  - 可以从 http://cmake.org/ 下载
  - 确保 cmake 可执行文件的路径在系统路径中，或者在调用 cmake 时使用完整路径。
- SWIG

  - 可以从 http://swig.org/ 下载
  - 确保 swig 可执行文件的路径在系统路径中，或者在调用 swig 时使用完整路径。请使
    用最新的版本，如 3.0.12。

- Protocol Buffers

  - 下载一个合适的版本，如 2.6.1:
    https://github.com/google/protobuf/releases/tag/v2.6.1 。
  - 下载 protobuf-2.6.1.zip 和 protoc-2.6.1-win32.zip。
  - 将这两个文件解压到 dependencies 文件夹中，将 protoc 可执行文件的路径添加到系
    统路径中，或者在调用它时使用完整路径。
  - 打开 Visual Studio solution，它可以在 vsproject 文件夹中找到。
  - 将 build settings 改为 Release 和 x64。
  - 构建 libprotobuf 项目。

- Openblas

  - 从 http://www.openblas.net 下载合适的源码，如 0.2.20。
  - 将源码解压到 dependencies 文件夹中。
  - 如果你没有安装 Perl，请下载一个 perl 环境，如 Strawberry Perl
    (http://strawberryperl.com/)。
  - 在源文件夹中运行此命令来构建 Visual Studio 解决方案：

  ```bash
  cmake -G "Visual Studio 15 2017 Win64"
  ```

  - 打开 Visual Studio 解决方案并将 build settings 更改为 Release 和 x64。
  - 构建 libopenblas 项目。

- Google glog
  - 下载一个合适的版本，如 0.3.5: https://github.com/google/glog/releases
  - 将源码解压到 dependencies 文件夹中。
  - 打开 Visual Studio solution.
  - 将 build settings 改为 Release and x64.
  - 构建 libglog 项目。

## 构建 SINGA 源代码

- 下载 SINGA 源代码
- 编译 protobuf 文件:

  - 在 src/proto 目录中：

  ```shell
  mkdir python_out
  protoc.exe *.proto --python_out python_out
  ```

- 为 C++和 Python 生成 swig 接口：在 src/api 目录中：

  ```shell
  swig -python -c++ singa.i
  ```

- 生成 SINGA 的 Visual Studio 解决方案：在 SINGA 源码根目录中：

  ```shell
  mkdir build
  cd build
  ```

- 调用 cmake 并添加系统路径，类似于下面的例子:

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

- 在 Visual Studio 中打开生成的解决方案。
- 将构建设置改为 Release 和 x64。
- 将 src/api 中的 singa_wrap.cxx 文件添加到 singa_objects 项目中。
- 在 singa_objects 项目中，打开 Additional Include Directories。
- 添加 Python 的 include 路径。
- 添加 numpy 的 include 路径。
- 添加 protobuf 的 include 路径。
- 在 singa_objects 项目的预处理程序定义中， 添加 USE_GLOG。
- 构建 singa_objects 项目。

- 在 singa 项目中:

  - 将 singa_wrap.obj 添加到对象库。
  - 将目标名称改为"\_singa_wrap"。
  - 将目标扩展名为.pyd。
  - 将配置类型改为动态库(.dll)。
  - 进入 Additional Library Directories，添加路径到 python、openblas、protobuf
    和 glog 库。
  - 在 Additional Dependencies 中添加 libopenblas.lib、libglog.lib 和
    libprotobuf.lib。

- 构建 singa 项目

## 安装 python 模块

- 将 build/python/setup.py 中的`_singa_wrap.so`改为`_singa_wrap.pyd`。
- 将`src/proto/python_out`中的文件复制到`build/python/singa/proto`中。

- （可选）创建并激活一个虚拟环境：

  ```shell
  mkdir SingaEnv
  virtualenv SingaEnv
  SingaEnv\Scripts\activate
  ```

- 进入 build/python 文件夹，运行:

  ```shell
  python setup.py install
  ```

- 将 \_singa_wrap.pyd、libglog.dll 和 libopenblas.dll 添加到路径中，或者将它们复
  制到 python site-packages 中的 singa package 文件夹中。

* 通过运行如下命令，来验证 SINGA 是否安装成功：

  ```shell
  python -c "from singa import tensor"
  ```

构建过程的视频教程可以在这里找到：

[![youtube video](https://img.youtube.com/vi/cteER7WeiGk/0.jpg)](https://www.youtube.com/watch?v=cteER7WeiGk)

## 运行单元测试

- 在测试文件夹中，生成 Visual Studio 解决方案：

  ```shell
  cmake -G "Visual Studio 15 2017 Win64"
  ```

- 在 Visual Studio 中打开生成的解决方案。

- 更改 build settings 为 Release 和 x64。

- 构建 glog 项目。

- 在 test_singa 项目中:

  - 将 USE_GLOG 添加到 Preprocessor Definitions 中。
  - 在 Additional Include Directories 中， 添加上面第 2 步中使用的
    GLOG_INCLUDE_DIR、 CBLAS_INCLUDE_DIR 和 Protobuf_INCLUDE_DIR 的路径。同时添
    加 build 和 build/include 文件夹。
  - 转到 Additional Library Directories，添加 openblas、protobuf 和 glog 库的路
    径。同时添加 build/src/singa_objects.dir/Release。
  - 转到 Additional Dependencies 并添加 libopenblas.lib、libglog.lib 和
    libprotobuf.lib。修改两个库的名字：gtest.lib 和 singa_objects.lib。

- 构建 test_singa 项目。

- 将 libglog.dll 和 libopenblas.dll 添加到路径中，或者将它们复制到 test/release
  文件夹中，使其可用。

- 单元测试可以通过如下方式执行：

  - 从命令行:

  ```shell
  test_singa.exe
  ```

  - 从 Visual Studio:
    - 右键点击 test_singa 项目，选择 "Set as StartUp Project"。
    - 在 Debug 菜单中，选择'Start Without Debugging'。

单元测试的视频教程可以在这里找到:

[![youtube video](https://img.youtube.com/vi/393gPtzMN1k/0.jpg)](https://www.youtube.com/watch?v=393gPtzMN1k)

## 构建包含 cuda 的 GPU 支持

在本节中，我们将扩展前面的步骤来启用 GPU。

### 安装依赖项

除了上面第 1 节的依赖关系外，我们还需要以下内容：

- CUDA

  从 https://developer.nvidia.com/cuda-downloads 下载一个合适的版本，如 9.1。确
  保已经安装了 Visual Studio 集成模块。

* cuDNN

  从 https://developer.nvidia.com/cudnn 下载一个合适的版本，如 7.1。

* cnmem:

  - 从 https://github.com/NVIDIA/cnmem 下载最新版本。
  - 构建 Visual Studio 解决方案：

  ```shell
  cmake -G "Visual Studio 15 2017 Win64"
  ```

  - 在 Visual Studio 中打开生成的解决方案。
  - 将 build settings 改为 Release 和 x64。
  - 构建 cnmem 项目。

### 构建 SINGA 源代码

- 调用 cmake 并添加系统路径，类似于下面的例子：
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

* 为 C++和 Python 生成 swig 接口。在 src/api 目录中：

  ```shell
  swig -python -c++ singa.i
  ```

* 在 Visual Studio 中打开生成的解决方案

* 将 build settings 改为 Release 和 x64

#### 构建 singa_objects

- 将 src/api 中的 singa_wrap.cxx 文件添加到 singa_objects 项目中。
- 在 singa_objects 项目中，打开 Additional Include Directories。
- 添加 Python 的 include 路径
- 添加 numpy include 路径
- 添加 protobuf 包括路径
- 增加 CUDA、cuDNN 和 cnmem 的包含路径。
- 在 singa_objects 项目的预处理程序定义中， 加入 USE_GLOG、 USE_CUDA 和
  USE_CUDNN。删除 DISABLE_WARNINGS。
- 建立 singa_objects 项目

#### 构建 singa-kernel

- 创建一个新的 Visual Studio 项目，类型为 "CUDA 9.1 Runtime"。给它起个名字，比如
  singa-kernel。
- 该项目自带一个名为 kernel.cu 的初始文件，从项目中删除这个文件。
- 添加这个文件：src/core/tensor/math_kernel.cu。
- 在项目设置中。

  - 将平台工具集设置为 "Visual Studio 2015 (v140)"
  - 将 "配置类型 "设置为 "静态库(.lib)"
  - 在 include 目录中，添加 build/include。

- 建立 singa-kernel 项目

#### 构建 singa

- 在 singa 项目中：

  - 将 singa_wrap.obj 添加到对象库中。
  - 将目标名称改为"\_singa_wrap"。
  - 将目标扩展名为.pyd。
  - 将配置类型改为动态库(.dll)。
  - 到 Additional Library Directories 中添加 python、openblas 的路径。protobuf
    和 glog 库。
  - 同时添加 singa-kernel、cnmem、cuda 和 cudnn 的 library path。
  - 到 Additional Dependencies，并添加 libopenblas.lib、libglog.lib 和
    libprotobuf.lib。
  - 另外还要添加
    ：singa-kernel.lib、cnmem.lib、cudnn.lib、cuda.lib、cublas.lib。curand.lib
    和 cudart.lib。

- 构建 singa 项目。

### Install Python module

- 将 build/python/setup.py 中的 \_singa_wrap.so 改为 \_singa_wrap.pyd。

- 将 src/proto/python_out 中的文件复制到 build/python/singa/proto 中。

- （可选） 创建并激活虚拟环境:

  ```shell
  mkdir SingaEnv
  virtualenv SingaEnv
  SingaEnv\Scripts\activate
  ```

- 进入 build/python 文件夹，运行:

  ```shell
  python setup.py install
  ```

- 将 \_singa_wrap.pyd, libglog.dll, libopenblas.dll, cnmem.dll, CUDA Runtime (例
  如 cudart64_91.dll) 和 cuDNN (例如 cudnn64_7.dll) 添加到路径中，或者将它们复制
  到 python site-packages 中的 singa package 文件夹中。

- 通过运行如下命令来验证 SINGA 是否已经安装：

  ```shell
  python -c "from singa import device; dev = device.create_cuda_gpu()"
  ```

这个部分的视频教程可以在这里找到：

[![youtube video](https://img.youtube.com/vi/YasKVjRtuDs/0.jpg)](https://www.youtube.com/watch?v=YasKVjRtuDs)

### 运行单元测试

- 在测试文件夹中，生成 Visual Studio 解决方案：

  ```shell
  cmake -G "Visual Studio 15 2017 Win64"
  ```

- 在 Visual Studio 中打开生成的解决方案，或者将项目添加到步骤 5.2 中创建的 singa
  解决方案中。

- 将 build settings 改为 Release 和 x64。

- 构建 glog 项目。

- 在 test_singa 项目中:

  - 将 USE_GLOG; USE_CUDA; USE_CUDNN 添加到 Preprocessor Definitions 中。
  - 在 Additional Include Directories 中， 添加上面 5.2 中使用的
    GLOG_INCLUDE_DIR、 CBLAS_INCLUDE_DIR 和 Protobuf_INCLUDE_DIR 的路径。同时添
    加 build、build/include、CUDA 和 cuDNN 的 include 文件夹。
  - 转到 Additional Library Directories，添加 openblas、protobuf 和 glog 库的路
    径。同时添加 build/src/singa_objects.dir/Release、singa-kernel、cnmem、CUDA
    和 cuDNN 库的路径。
  - 在 Additional Dependencies 中添加 libopenblas.lib; libglog.lib;
    libprotobuf.lib; cnmem.lib; cudnn.lib; cuda.lib; cublas.lib; curand.lib;
    cudart.lib; singa-kernel.lib。修正两个库的名字：gtest.lib 和
    singa_objects.lib。

* 构建.

* 将 libglog.dll、libopenblas.dll、cnmem.dll、cudart64_91.dll 和 cudnn64_7.dll
  添加到路径中，或将它们复制到 test/release 文件夹中，使其可用。

* 单元测试可以通过如下方式执行：

  - 从命令行:

    ```shell
    test_singa.exe
    ```

  - 从 Visual Studio:
    - 右键点击 test_singa 项目，选择 'Set as StartUp Project'.
    - 从 Debug 菜单，选择 'Start Without Debugging'

运行单元测试的视频教程可以在这里找到：

[![youtube video](https://img.youtube.com/vi/YOjwtrvTPn4/0.jpg)](https://www.youtube.com/watch?v=YOjwtrvTPn4)
