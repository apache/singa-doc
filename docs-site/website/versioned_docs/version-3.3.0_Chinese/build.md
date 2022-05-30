---
id: version-3.3.0_Chinese-build
title: Build SINGA from Source
original_id: build
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

源文件可以通过[tar.gz文件](https://dist.apache.org/repos/dist/dev/singa/)或git repo的形式下载：

```shell
$ git clone https://github.com/apache/singa.git
$ cd singa/
```

如果您想为SINGA贡献代码，请参考[贡献代码](contribute-code.md)页面的步骤和要求。

## 使用Conda构筑SINGA

Conda-build 是一个构建工具，它从anaconda cloud安装依赖的库并执行构建脚本。

安装conda-build(需要先安装conda)：

```shell
conda install conda-build
```

### 构建CPU版本

构建SINGA的CPU版本：

```shell
conda build tool/conda/singa/
```

以上命令已经在Ubuntu（14.04，16.04和18.04）和macOS 10.11上测试过。更多信息请参考[Travis-CI page](https://travis-ci.org/apache/singa)页面。

### 构建GPU版本

要构建GPU版的SINGA，计算机必须装备有Nvida GPU，而且需要安装CUDA driver(>=384.81)、CUDA toolkit(>=9)和cuDNN(>=7)。以下两个Docker镜像提供了构建环境：

1. apache/singa:conda-cuda9.0
2. apache/singa:conda-cuda10.0

构建环境准备好后，需要先导出CUDA版本，然后运行conda命令构建SINGA：

```shell
export CUDA=x.y (e.g. 9.0)
conda build tool/conda/singa/
```

### 后处理

生成的包文件的位置(`.tar.gz`)将打印在终端上，生成的包可以直接安装：

```shell
conda install -c conda-forge --use-local <path to the package file>
```

若要上传到anaconda云端供他人下载安装，需要在 anaconda 上注册一个账号，才能[上传包](https://docs.anaconda.com/anaconda-cloud/user-guide/getting-started/)：

```shell
conda install anaconda-client
anaconda login
anaconda upload -l main <path to the package file>
```

将包上传到云端后，您可以在[Anaconda Cloud](https://anaconda.org/)上看到，也可以通过以下命令查看：

```shell
conda search -c <anaconda username> singa
```

每个特定的SINGA软件包都由版本和构建字符串来标识。要安装一个特定的SINGA包，需要提供所有信息，例如：

```shell
conda install -c <anaconda username> -c conda-forge singa=2.1.0.dev=cpu_py36
```

为了使安装命令简单化，您可以创建以下附加包，这些包依赖于最新的CPU和GPU SINGA包：

```console
# for singa-cpu
conda build tool/conda/cpu/  --python=3.6
conda build tool/conda/cpu/  --python=3.7
# for singa-gpu
conda build tool/conda/gpu/  --python=3.6
conda build tool/conda/gpu/  --python=3.7
```

因此，当您运行：

```shell
conda install -c <anaconda username> -c conda-forge singa-xpu
```

时(`xpu`表示'cpu' or 'gpu'), 相应的真正的SINGA包将作为依赖库被安装。

## 使用本地工具在Ubuntu上构建SINGA

请参阅 SINGA [Dockerfiles](https://github.com/apache/singa/blob/master/tool/docker/devel/ubuntu/cuda9/Dockerfile#L30)，了解在 Ubuntu 16.04 上安装依赖库的说明。您也可以使用 devel 映像创建一个 Docker 容器，并在容器中构建 SINGA。要使用GPU、DNNL、Python和单元测试来构建SINGA，请运行以下命令：

```shell
mkdir build    # at the root of singa folder
cd build
cmake -DENABLE_TEST=ON -DUSE_CUDA=ON -DUSE_DNNL=ON -DUSE_PYTHON3=ON ..
make
cd python
pip install .
```
CMake选项的详细内容在本页最后一节解释，上面最后一条命令是安装Python包。你也可以运行`pip install -e .`，它可以创建符号链接，而不是将 Python 文件复制到 site-package 文件夹中。

如果SINGA在ENABLE_TEST=ON的情况下编译，您可以通过以下方式运行单元测试:

```shell
$ ./bin/test_singa
```

您可以看到所有的测试案例与测试结果。如果SINGA通过了所有测试，那么您就成功安装了SINGA。

## 使用本地工具在Centos7上构建SINGA

由于Centos7的软件包名称不同，因此从源码开始构建会有所不同。

### 安装依赖项

基础包和库文件：

```shell
sudo yum install freetype-devel libXft-devel ncurses-devel openblas-devel blas-devel lapack devel atlas-devel kernel-headers unzip wget pkgconfig zip zlib-devel libcurl-devel cmake curl unzip dh-autoreconf git python-devel glog-devel protobuf-devel
```

构建必需的包：

```shell
sudo yum group install "Development Tools"
```

若要安装swig：

```shell
sudo yum install pcre-devel
wget http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz
tar xvzf swig-3.0.10.tar.gz
cd swig-3.0.10.tar.gz
./configure --prefix=${RUN}
make
make install
```

安装gfortran：

```shell
sudo yum install centos-release-scl-rh
sudo yum --enablerepo=centos-sclo-rh-testing install devtoolset-7-gcc-gfortran
```

安装pip和其他包：

```shell
sudo yum install epel-release
sudo yum install python-pip
pip install matplotlib numpy pandas scikit-learn pydot
```

### 安装SINGA

按照《使用本地工具在Ubuntu上构建SINGA》的步骤1-5进行操作

### 测试

您可以通过如下方式进行

```shell
$ ./bin/test_singa
```

您可以看到所有的测试案例与测试结果。如果SINGA通过了所有测试，即表示安装成功。

## 在Windows中编译SINGA
在Windows上使用Python支持构建SINGA的说明可以在[install-win页面](install-win.md)找到。

## 关于编译选项的更多细节

### USE_MODULES (已过期废弃)

如果没有安装protobuf和openblas，你可以用它们一起编译SINGA

```shell
$ In SINGA ROOT folder
$ mkdir build
$ cd build
$ cmake -DUSE_MODULES=ON ..
$ make
```
cmake 会下载 OpenBlas 和 Protobuf (2.6.1) 并与 SINGA 一起编译。

您可以使用`ccmake ..`来配置编译选项。如果一些依赖的库不在系统默认路径中，则您需要手动导出以下环境变量：

```shell
export CMAKE_INCLUDE_PATH=<path to the header file folder>
export CMAKE_LIBRARY_PATH=<path to the lib file folder>
```

### USE_PYTHON

编译SINGA的Python封装器选项：

```shell
$ cmake -DUSE_PYTHON=ON ..
$ make
$ cd python
$ pip install .
```

### USE_CUDA
我们推荐用户安装CUDA和[cuDNN](https://developer.nvidia.com/cudnn)，以便在GPU上运行SINGA，以获得更好的性能。

SINGA已经在CUDA 9/10和cuDNN 7上进行了测试。如果cuDNN安装在非系统文件夹中，例如 /home/bob/local/cudnn/，则需要执行以下命令来让cmake在编译时能够找到它们：

```shell
$ export CMAKE_INCLUDE_PATH=/home/bob/local/cudnn/include:$CMAKE_INCLUDE_PATH
$ export CMAKE_LIBRARY_PATH=/home/bob/local/cudnn/lib64:$CMAKE_LIBRARY_PATH
$ export LD_LIBRARY_PATH=/home/bob/local/cudnn/lib64:$LD_LIBRARY_PATH
```

CUDA和cuDNN的cmake选项应该设置成“ON”：

```shell
# Dependent libs are install already
$ cmake -DUSE_CUDA=ON ..
$ make
```

### USE_DNNL

用户可以启用DNNL来提高CPU的计算性能，DNNL的安装指南可以在[这里](https://github.com/intel/mkl-dnn#installation)找到：


SINGA在DNNL v1.1环境下已经进行过测试并通过，

若要启用DNNL支持来编译SINGA:

```shell
# Dependent libs are installed already
$ cmake -DUSE_DNNL=ON ..
$ make
```

### USE_OPENCL

SINGA使用opencl-headers和viennacl（版本1.7.1及以上）来支持OpenCL，它们可以通过如下方式安装：

```shell
# On Ubuntu 16.04
$ sudo apt-get install opencl-headers, libviennacl-dev
# On Fedora
$ sudo yum install opencl-headers, viennacl
```

此外，你需要在你想运行OpenCL的平台安装OpenCL Installable Client Driver（ICD）。

- 对于AMD和Nvidia的GPU，驱动包也应该安装与之匹配的OpenCL ICD。
- 对于Intel的CPU和/或GPU，请从[Intel官方网站](https://software.intel.com/en-us/articles/opencl-drivers)上获取驱动程序。请注意，该网站上提供的驱动程序只支持最新的CPU和Iris GPU。
- 对于旧的Intel CPU，你可以使用beignet-opencl-icd包。

请注意，目前不建议在CPU上运行OpenCL，因为运行速度会很慢。内存传输是以整数秒为单位的（直观来说，CPU上是1000毫秒，而GPU上是1毫秒）。

更多关于建立OpenCL工作环境的信息可以在[这里](https://wiki.tiker.net/OpenCLHowTo)找到。

如果ViennaCL的软件包版本不是至少1.7.1，则需要从源码构建它：

从[这个git repo](https://github.com/viennacl/viennacl-dev)clone版本库，切换（checkout）到release-1.7.1分支，然后构建它，并把项目路径添加到PATH，再把构建的库文件添加到LD_LIBRARY_PATH。

构建支持OpenCL的SINGA（在SINGA 1.1上测试）：

```shell
$ cmake -DUSE_OPENCL=ON ..
$ make
```

### PACKAGE

这个设置是用来构建 Debian 软件包的。设置PACKAGE=ON，然后用make命令来编译软件包，如下所示：

```shell
$ cmake -DPACKAGE=ON
$ make package
```

## FAQ

- Q: 'import singa'阶段报错

  A: 请检查`python -c "from singa import _singa_wrap`中的详细错误。有时是由依赖库引起的，比如protobuf有多个版本，cudnn缺失，numpy版本不匹配等问题。下面展示了不同情况下的解决方案

  1. 检查cudnn和cuda。如果cudnn缺失或与wheel包的版本不一致，你可以下载正确的cudnn版本到~/local/cudnn/，然后：

     ```shell
     $ echo "export LD_LIBRARY_PATH=/home/<yourname>/local/cudnn/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
     ```

  2. 如果是protobuf的问题。你可以将protobuf (3.6.1)从源码安装到本地文件夹，比如 ~/local/，解压tar文件，然后：

     ```shell
     $ ./configure --prefix=/home/<yourname>local
     $ make && make install
     $ echo "export LD_LIBRARY_PATH=/home/<yourname>/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
     $ source ~/.bashrc
     ```

  3. 如果找不到包括python在内的其他类库，则使用`pip`或`conda`创建虚拟环境.

  4. 如果不是上述原因造成的，请到`_singa_wrap.so`文件夹中查看：
     ```shell
     $ python
     >> import importlib
     >> importlib.import_module('_singa_wrap')
     ```

     来检查错误信息。例如，如果是numpy的版本不匹配，错误信息会是：

     ```shell
     RuntimeError: module compiled against API version 0xb but this version of numpy is 0xa
     ```

     那么你就需要更新numpy到更高版本。

* Q: 运行`cmake ...`时出错，找不到依赖库。

  A: 如果你还没有安装这些依赖库，请先安装它们。如果你在系统文件夹之外的文件夹中安装了库，例如/usr/local，那您需要手动导出以下变量:

  ```shell
  $ export CMAKE_INCLUDE_PATH=<path to your header file folder>
  $ export CMAKE_LIBRARY_PATH=<path to your lib file folder>
  ```

- Q: 来自`make`的错误，例如linking阶段的错误.

  A: 如果您的库文件在系统默认路径以外的其他文件夹中，则需要手动导出以下变量。

  ```shell
  $ export LIBRARY_PATH=<path to your lib file folder>
  $ export LD_LIBRARY_PATH=<path to your lib file folder>
  ```

* Q: 来自头文件的错误，例如'cblas.h文件不存在'

  A: 您需要手动将cblas.h的文件夹包含在CPLUS_INCLUDE_PATH中，例如：

  ```shell
  $ export CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$CPLUS_INCLUDE_PATH
  ```

* Q: 在编译SINGA时，我收到错误信息`SSE2 instruction set not enabled`

  A:您可以尝试如下指令:

  ```shell
  $ make CFLAGS='-msse2' CXXFLAGS='-msse2'
  ```

* Q:当试图导入.py文件时，从google.protobuf.internal收到`ImportError: cannot import name enum_type_wrapper`。

  A: 您需要安装protobuf的Python绑定包，它可以通过如下方式安装：

  ```shell
  $ sudo apt-get install protobuf
  ```

  或从源文件编译：

  ```shell
  $ cd /PROTOBUF/SOURCE/FOLDER
  $ cd python
  $ python setup.py build
  $ python setup.py install
  ```

* Q: 当我从源码构建OpenBLAS时，被告知需要一个Fortran编译器。

  A: 您可以通过如下方式编译OpenBLAS：

  ```shell
  $ make ONLY_CBLAS=1
  ```
  或者通过如下方式安装：

  ```shell
  $ sudo apt-get install libopenblas-dev
  ```

* Q: 当我构建协议缓冲区时，它报告说在`/usr/lib64/libstdc++.so.6`中没有找到`GLIBC++_3.4.20`？

  A: 这意味着链接器找到了libstdc++.so.6，但该库属于旧的GCC（用于编译和链接程序的GCC）版本。此程序依赖于新的libstdc++中定义的代码，而该代码属于较新版本的GCC，所以必须告诉链接器如何找到较新的libstdc++共享库。最简单的解决方法是找到正确的libstdc++，并将其导出到LD_LIBRARY_PATH。例如，如果下面命令的输出中列出了GLIBC++_3.4.20：

        $ strings /usr/local/lib64/libstdc++.so.6|grep GLIBC++

  那么接下来需要设置环境变量为：

        $ export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

* Q: 当构建glog时，报告说 "src/logging_unittest.cc:83:20: error: 'gflags' is not a namespace-name"。

  A: 可能是由于安装的gflags用了不同的命名空间，比如 "google"，所以glog找不到'gflags'的命名空间。实际上建立glog并不需要gflags，所以你可以修改configure.ac文件来忽略gflags。

        1. cd to glog src directory
        2. change line 125 of configure.ac  to "AC_CHECK_LIB(gflags, main, ac_cv_have_libgflags=0, ac_cv_have_libgflags=0)"
        3. autoreconf

  执行上述命令后，就可以重新构建glog了。

* Q: 在使用虚拟环境时，每次运行pip install，都会重新安装numpy。但是，当运行`import numpy`时，numpy并没有被调用。

  A: 可能是由于`PYTHONPATH`造成的，在使用虚拟环境时，应将`PYTHONPATH`设置为空，以避免与虚拟环境本身的路径冲突。

* Q: 当从源代码编译PySINGA时，由于缺少<numpy/objectarray.h>，出现的编译错误。

  A: 请安装numpy并导出numpy头文件的路径为

        $ export CPLUS_INCLUDE_PATH=`python -c "import numpy; print numpy.get_include()"`:$CPLUS_INCLUDE_PATH

* Q: 当我在Mac OS X中运行SINGA时，报错 "Fatal Python error:
  PyThreadState_Get: no current thread Abort trap: 6"

  A: 这个错误通常发生在系统上有多个版本的Python，并且是通过pip安装的SINGA (通过conda安装时不会出现这个问题)，例如，操作系统自带的版本和Homebrew安装的版本。PySINGA所链接的Python必须与Python解释器（interpreter）相同。 您可以通过 `which python` 检查您的解释器路径并通过`otool -L <path to _singa_wrap.so>`检查PySINGA链接的Python路径。要解决这个问题，请用正确的 Python 版本编译 SINGA。需要注意的是，如果您从源码编译 PySINGA，您需要在调用[cmake](http://stackoverflow.com/questions/15291500/i-have-2-versions-of-python-installed-but-cmake-is-using-older-version-how-do)时指定路径：
  

        $ cmake -DPYTHON_LIBRARY=`python-config --prefix`/lib/libpython2.7.dylib -DPYTHON_INCLUDE_DIR=`python-config --prefix`/include/python2.7/ ..

  如果从二进制包中安装PySINGA，例如debian或wheel，那么你需要改变python解释器的路径，例如，重新设置$PATH，并把Python的正确路径放在前面的位置。
