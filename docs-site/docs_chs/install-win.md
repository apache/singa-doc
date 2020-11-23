---
id: install-win
title: Build SINGA on Windows
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

在Microsoft Windows上从源码构建SINGA的过程有四个部分：安装依赖关系、构建SINGA源码、（可选择）安装python模块和（可选择）运行单元测试。

## 安装依赖项

你可以创建一个文件夹来构建依赖关系。

使用到的依赖项有：

- 编译器和IDE：
  - Visual Studio，社区版是免费的，可以用来构建SINGA。
    https://www.visualstudio.com/
- CMake
  - 可以从 http://cmake.org/ 下载
  - 确保 cmake 可执行文件的路径在系统路径中，或者在调用 cmake 时使用完整路径。
- SWIG
  - 可以从 http://swig.org/ 下载
  - 确保swig可执行文件的路径在系统路径中，或者在调用swig时使用完整路径。请使用最新的版本，如3.0.12。

- Protocol Buffers
  - 下载一个合适的版本，如2.6.1:
    https://github.com/google/protobuf/releases/tag/v2.6.1 。
  - 下载 protobuf-2.6.1.zip 和 protoc-2.6.1-win32.zip。
  - 将这两个文件解压到dependencies文件夹中，将protoc可执行文件的路径添加到系统路径中，或者在调用它时使用完整路径。
  - 打开Visual Studio solution，它可以在vsproject文件夹中找到。
  - 将build settings改为Release和x64。
  - 构建libprotobuf项目。

- Openblas
  - 从 http://www.openblas.net 下载合适的源码，如0.2.20。
  - 将源码解压到dependencies文件夹中。
  - 如果你没有安装Perl，请下载一个perl环境，如Strawberry Perl (http://strawberryperl.com/)。
  - 在源文件夹中运行此命令来构建Visual Studio解决方案：

  ```bash
  cmake -G "Visual Studio 15 2017 Win64"
  ```

  - 打开Visual Studio解决方案并将build settings更改为Release和x64。
  - 构建libopenblas项目。

- Google glog
  - 下载一个合适的版本，如0.3.5:
    https://github.com/google/glog/releases
  - 将源码解压到dependencies文件夹中。
  - 打开Visual Studio solution.
  - 将build settings改为Release and x64.
  - 构建libglog项目。

## 构建SINGA源代码

- 下载SINGA源代码
- 编译protobuf文件:

  - 在src/proto目录中：

  ```shell
  mkdir python_out
  protoc.exe *.proto --python_out python_out
  ```

- 为C++和Python生成swig接口：在src/api目录中：

  ```shell
  swig -python -c++ singa.i
  ```

- 生成SINGA的Visual Studio解决方案：在SINGA源码根目录中：

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

- 在Visual Studio中打开生成的解决方案。
- 将构建设置改为Release和x64。
- 将src/api中的singa_wrap.cxx文件添加到singa_objects项目中。
- 在 singa_objects 项目中，打开 Additional Include Directories。
- 添加Python的include路径。
- 添加numpy的include路径。
- 添加protobuf的include路径。
- 在 singa_objects 项目的预处理程序定义中， 添加 USE_GLOG。
- 构建singa_objects项目。

- 在singa项目中:
  - 将singa_wrap.obj添加到对象库。
  - 将目标名称改为"_singa_wrap"。
  - 将目标扩展名为.pyd。
  - 将配置类型改为动态库(.dll)。
  - 进入Additional Library Directories，添加路径到python、openblas、protobuf和glog库。
  - 在Additional Dependencies中添加libopenblas.lib、libglog.lib和libprotobuf.lib。

- 构建singa项目

## 安装python模块


- 将build/python/setup.py中的`_singa_wrap.so`改为`_singa_wrap.pyd`。
- 将`src/proto/python_out`中的文件复制到`build/python/singa/proto`中。

- （可选）创建并激活一个虚拟环境：
  ```shell
  mkdir SingaEnv
  virtualenv SingaEnv
  SingaEnv\Scripts\activate
  ```

- 进入build/python文件夹，运行:

  ```shell
  python setup.py install
  ```

- 将 _singa_wrap.pyd、libglog.dll 和 libopenblas.dll 添加到路径中，或者将它们复制到 python site-packages 中的 singa package 文件夹中。


- 通过运行如下命令，来验证SINGA是否安装成功：

  ```shell
  python -c "from singa import tensor"
  ```

构建过程的视频教程可以在这里找到：

[![youtube video](https://img.youtube.com/vi/cteER7WeiGk/0.jpg)](https://www.youtube.com/watch?v=cteER7WeiGk)

## 运行单元测试

- 在测试文件夹中，生成Visual Studio解决方案：

  ```shell
  cmake -G "Visual Studio 15 2017 Win64"
  ```

- 在Visual Studio中打开生成的解决方案。

- 更改build settings为Release和x64。

- 构建glog项目。

- 在test_singa项目中:
  - 将 USE_GLOG 添加到Preprocessor Definitions中。
  - 在 Additional Include Directories 中， 添加上面第 2 步中使用的 GLOG_INCLUDE_DIR、 CBLAS_INCLUDE_DIR 和 Protobuf_INCLUDE_DIR 的路径。同时添加build和build/include文件夹。
  - 转到Additional Library Directories，添加openblas、protobuf和glog库的路径。同时添加 build/src/singa_objects.dir/Release。
  - 转到 Additional Dependencies 并添加 libopenblas.lib、libglog.lib 和 libprotobuf.lib。修改两个库的名字：gtest.lib和singa_objects.lib。

- 构建test_singa项目。

- 将libglog.dll和libopenblas.dll添加到路径中，或者将它们复制到test/release文件夹中，使其可用。

- 单元测试可以通过如下方式执行：

  - 从命令行:

  ```shell
  test_singa.exe
  ```

  - 从Visual Studio:
    - 右键点击test_singa项目，选择 "Set as StartUp Project"。
    - 在Debug菜单中，选择'Start Without Debugging'。

单元测试的视频教程可以在这里找到:

[![youtube video](https://img.youtube.com/vi/393gPtzMN1k/0.jpg)](https://www.youtube.com/watch?v=393gPtzMN1k)

## 构建包含cuda的GPU支持

在本节中，我们将扩展前面的步骤来启用GPU。

### 安装依赖项

除了上面第1节的依赖关系外，我们还需要以下内容：

- CUDA

  从 https://developer.nvidia.com/cuda-downloads 下载一个合适的版本，如9.1。确保已经安装了Visual Studio集成模块。


- cuDNN

  从 https://developer.nvidia.com/cudnn 下载一个合适的版本，如7.1。

- cnmem:

  - 从 https://github.com/NVIDIA/cnmem 下载最新版本。
  - 构建Visual Studio解决方案：

  ```shell
  cmake -G "Visual Studio 15 2017 Win64"
  ```

  - 在Visual Studio中打开生成的解决方案。
  - 将build settings改为Release和x64。
  - 构建cnmem项目。

### 构建SINGA源代码

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

* 为C++和Python生成swig接口。在src/api目录中：

  ```shell
  swig -python -c++ singa.i
  ```

* 在Visual Studio中打开生成的解决方案

* 将build settings改为Release和x64

#### 构建singa_objects

- 将src/api中的singa_wrap.cxx文件添加到singa_objects项目中。
- 在 singa_objects 项目中，打开 Additional Include Directories。
- 添加Python的include路径
- 添加numpy include路径
- 添加protobuf包括路径
- 增加CUDA、cuDNN和cnmem的包含路径。
- 在 singa_objects 项目的预处理程序定义中， 加入 USE_GLOG、 USE_CUDA 和 USE_CUDNN。删除 DISABLE_WARNINGS。
- 建立 singa_objects 项目

#### 构建singa-kernel


- 创建一个新的Visual Studio项目，类型为 "CUDA 9.1 Runtime"。给它起个名字，比如singa-kernel。
- 该项目自带一个名为kernel.cu的初始文件，从项目中删除这个文件。
- 添加这个文件：src/core/tensor/math_kernel.cu。
- 在项目设置中。

  - 将平台工具集设置为 "Visual Studio 2015 (v140)"
  - 将 "配置类型 "设置为 "静态库(.lib)"
  - 在include目录中，添加build/include。

- 建立singa-kernel项目

#### 构建singa

- 在singa项目中：

  - 将singa_wrap.obj添加到对象库中。
  - 将目标名称改为"_singa_wrap"。
  - 将目标扩展名为.pyd。
  - 将配置类型改为动态库(.dll)。
  - 到Additional Library Directories中添加python、openblas的路径。protobuf和glog库。
  - 同时添加singa-kernel、cnmem、cuda和cudnn的library path。
  - 到Additional Dependencies，并添加libopenblas.lib、libglog.lib和 libprotobuf.lib。
  - 另外还要添加：singa-kernel.lib、cnmem.lib、cudnn.lib、cuda.lib、cublas.lib。curand.lib和cudart.lib。

- 构建singa项目。

### Install Python module

- 将 build/python/setup.py 中的 _singa_wrap.so 改为 _singa_wrap.pyd。

- 将 src/proto/python_out 中的文件复制到 build/python/singa/proto 中。

- （可选） 创建并激活虚拟环境:

  ```shell
  mkdir SingaEnv
  virtualenv SingaEnv
  SingaEnv\Scripts\activate
  ```

- 进入build/python文件夹，运行:

  ```shell
  python setup.py install
  ```

- 将 _singa_wrap.pyd, libglog.dll, libopenblas.dll, cnmem.dll, CUDA Runtime (例如 cudart64_91.dll) 和 cuDNN (例如 cudnn64_7.dll) 添加到路径中，或者将它们复制到 python site-packages 中的 singa package 文件夹中。

- 通过运行如下命令来验证SINGA是否已经安装：

  ```shell
  python -c "from singa import device; dev = device.create_cuda_gpu()"
  ```

这个部分的视频教程可以在这里找到：

[![youtube video](https://img.youtube.com/vi/YasKVjRtuDs/0.jpg)](https://www.youtube.com/watch?v=YasKVjRtuDs)

### 运行单元测试

- 在测试文件夹中，生成Visual Studio解决方案：

  ```shell
  cmake -G "Visual Studio 15 2017 Win64"
  ```

- 在Visual Studio中打开生成的解决方案，或者将项目添加到步骤5.2中创建的singa解决方案中。

- 将build settings改为Release和x64。

- 构建 glog 项目。

- 在test_singa项目中:

  - 将 USE_GLOG; USE_CUDA; USE_CUDNN 添加到Preprocessor Definitions中。
  - 在 Additional Include Directories 中， 添加上面 5.2 中使用的 GLOG_INCLUDE_DIR、 CBLAS_INCLUDE_DIR 和 Protobuf_INCLUDE_DIR 的路径。同时添加build、build/include、CUDA和cuDNN的include文件夹。
  - 转到Additional Library Directories，添加openblas、protobuf和glog库的路径。同时添加 build/src/singa_objects.dir/Release、singa-kernel、cnmem、CUDA 和 cuDNN 库的路径。
  - 在Additional Dependencies中添加libopenblas.lib; libglog.lib; libprotobuf.lib; cnmem.lib; cudnn.lib; cuda.lib; cublas.lib; curand.lib; cudart.lib; singa-kernel.lib。修正两个库的名字：gtest.lib和singa_objects.lib。

* 构建.

* 将libglog.dll、libopenblas.dll、cnmem.dll、cudart64_91.dll和cudnn64_7.dll添加到路径中，或将它们复制到test/release文件夹中，使其可用。

* 单元测试可以通过如下方式执行：

  - 从命令行:

    ```shell
    test_singa.exe
    ```

  - 从 Visual Studio:
    - 右键点击test_singa项目，选择 'Set as StartUp Project'.
    - 从Debug菜单，选择 'Start Without Debugging'

运行单元测试的视频教程可以在这里找到：

[![youtube video](https://img.youtube.com/vi/YOjwtrvTPn4/0.jpg)](https://www.youtube.com/watch?v=YOjwtrvTPn4)
