---
id: installation
title: Installation
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## 使用Conda

Conda是一个Python、CPP等包的包管理器。

目前，SINGA有Linux和MacOSX的conda包。推荐使用[Miniconda3](https://conda.io/miniconda.html)来配合SINGA使用，安装miniconda后，执行以下命令之一安装SINGA。

1. 只使用CPU
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ntkhi-Z6XTR8WYPXiLwujHd2dOm0772V?usp=sharing)

```shell
$ conda install -c nusdbsystem -c conda-forge singa-cpu
```

2. 使用带CUDA和cuDNN的GPU（需要CUDA驱动>=384.81）
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1do_TLJe18IthLOnBOsHCEe-FFPGk1sPJ?usp=sharing)

```shell
$ conda install -c nusdbsystem -c conda-forge singa-gpu
```

3. 安装特定版本的SINGA，下面的命令列出了所有可用的SINGA软件包：

```shell
$ conda search -c nusdbsystem singa

Loading channels: done
# Name                       Version           Build  Channel
singa                      3.1.0.rc2        cpu_py36  nusdbsystem
singa                      3.1.0.rc2 cudnn7.6.5_cuda10.2_py36  nusdbsystem
singa                      3.1.0.rc2 cudnn7.6.5_cuda10.2_py37  nusdbsystem
```

<!--- > Please note that using the nightly built images is not recommended except for SINGA development and testing. Using stable releases is recommended. -->

下面的命令将安装SINGA的特定版本：

```shell
$ conda install -c nusdbsystem -c conda-forge singa=X.Y.Z=cpu_py36
```

如果运行以下命令没有报错：
```shell
$ python -c "from singa import tensor"
```

那么SINGA就安装成功了。

## 使用pip

1. 只使用CPU
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17RA056Brwk0vBQTFaZ-l9EbqwADO0NA9?usp=sharing)

```bash
pip install singa -f http://singa.apache.org/docs/next/wheel-cpu.html --trusted-host singa.apache.org
```

您可以通过`singa==<version>`安装特定版本的SINGA，其中`<version>`字段应被替换，例如`3.1.0`。可用的SINGA版本在链接中列出。

要安装最新的开发版本，请将链接替换为
http://singa.apache.org/docs/next/wheel-cpu-dev.html

2. 使用CUDA和cuDNN的GPU
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1W30IPCqj5fG8ADAQsFqclaCLyIclVcJL?usp=sharing)

```bash
pip install singa -f http://singa.apache.org/docs/next/wheel-gpu.html --trusted-host singa.apache.org
```

您也可以配置SINGA版本和CUDA版本，比如`singa==3.1.0+cuda10.2`，SINGA版本和CUDA版本的可用组合在链接中列出。

要安装最新的开发版本，请将链接替换为
http://singa.apache.org/docs/next/wheel-gpu-dev.html

注意：你本地Python环境的Python版本将被用来寻找相应的wheel包。例如，如果你本地的Python是3.6，那么就会通过pip选择在Python 3.6上编译的wheel包并安装。事实上，wheel文件的名称包括SINGA版本、CUDA版本和Python版本。因此，`pip`知道要下载和安装哪个wheel文件。

参考setup.py文件顶部的注释，了解如何构建wheel包。

## 使用Docker

按照[说明](https://docs.docker.com/install/)在你的本地主机上安装Docker。将您的用户添加到[docker组](https://docs.docker.com/install/linux/linux-postinstall/)中，以便在没有`sudo`的情况下运行docker命令。

1. 仅使用CPU

```shell
$ docker run -it apache/singa:X.Y.Z-cpu-ubuntu16.04 /bin/bash
```

2. 要使用GPU，在安装Docker后安装
   [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) 

```shell
$ nvidia-docker run -it apache/singa:X.Y.Z-cuda9.0-cudnn7.4.2-ubuntu16.04 /bin/bash
```

3. 关于SINGA Docker镜像（标签）的完整列表，请访问[docker hub site](https://hub.docker.com/r/apache/singa/)。对于每个docker镜像，标签的命名为：

```shell
version-(cpu|gpu)[-devel]
```

| Tag       | Description                      | Example value                                                                                                                                                             |
| --------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `version` | SINGA version                    | '2.0.0-rc0', '2.0.0', '1.2.0'                                                                                                                                             |
| `cpu`     | the image cannot run on GPUs     | 'cpu'                                                                                                                                                                     |
| `gpu`     | the image can run on Nvidia GPUs | 'gpu', or 'cudax.x-cudnnx.x' e.g., 'cuda10.0-cudnn7.3'                                                                                                                    |
| `devel`   | indicator for development        | if absent, SINGA Python package is installed for runtime only; if present, the building environment is also created, you can recompile SINGA from source at '/root/singa' |
| `OS`      | indicate OS version number       | 'ubuntu16.04', 'ubuntu18.04'                                                                                                                                              |

## 从源码编译
您可以使用本地构建工具或conda-build在本地主机操作系统上或在Docker容器中从源代码[构建和安装SINGA](build.md)。

## FAQ

- Q: `from singa import tensor`错误

  A: 执行下面的命令，检查详细的错误：

  ```shell
  python -c  "from singa import _singa_wrap"
  # go to the folder of _singa_wrap.so
  ldd path to _singa_wrap.so
  python
  >> import importlib
  >> importlib.import_module('_singa_wrap')
  ```
  `_singa_wrap.so` 的文件夹是 `~/miniconda3/lib/python3.7/site-packages/singa`。通常情况下，这个错误是由于依赖的库不匹配或缺失造成的，例如 cuDNN 或 protobuf。解决方法是创建一个新的虚拟环境，并在该环境中安装SINGA，例如：
  

  ```shell
  conda create -n singa
  conda activate singa
  conda install -c nusdbsystem -c conda-forge singa-cpu
  ```

- Q: 使用虚拟环境时，每次安装SINGA时，都会重新安装numpy。但是，当我运行`import numpy`时，numpy没有被使用。

  A: 
  这可能是由`PYTHONPATH`环境变量引起的，在使用虚拟环境时，应将其设置为空，以避免与虚拟环境的路径冲突。

- Q: 当我在Mac OS X中运行SINGA时，得到如下错误 "Fatal Python error:
  PyThreadState_Get: no current thread Abort trap: 6"

  A: 这个错误通常发生在系统中有多个 Python 版本的时候，例如，操作系统自带的版本和 Homebrew 安装的版本。SINGA链接的Python必须与Python解释器相同。您可以通过`which python`来检查解释器python版本并通过`otool -L <path to _singa_wrap.so>` 检查 SINGA 链接的 Python，如果通过conda安装SINGA，这个问题应该可以解决。
