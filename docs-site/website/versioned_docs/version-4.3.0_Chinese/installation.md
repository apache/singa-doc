---
id: version-4.3.0_Chinese-installation
title: Installation
original_id: installation
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## 使用 pip

1. 只使用 CPU
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17RA056Brwk0vBQTFaZ-l9EbqwADO0NA9?usp=sharing)

```bash
pip install singa -f http://singa.apache.org/docs/next/wheel-cpu.html --trusted-host singa.apache.org
```

您可以通过`singa==<version>`安装特定版本的 SINGA，其中`<version>`字段应被替换，
例如`2.1.0`。可用的 SINGA 版本在链接中列出。

要安装最新的开发版本，请将链接替换为
http://singa.apache.org/docs/next/wheel-cpu-dev.html

2. 使用 CUDA 和 cuDNN 的 GPU
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1W30IPCqj5fG8ADAQsFqclaCLyIclVcJL?usp=sharing)

```bash
pip install singa -f http://singa.apache.org/docs/next/wheel-gpu.html --trusted-host singa.apache.org
```

您也可以配置 SINGA 版本和 CUDA 版本，�`s4.3.0+cuda10.2`，SINGA 版本和 CUDA 版
本的可用组合在链接中列出。

要安装最新的开发版本，请将链接替换为
http://singa.apache.org/docs/next/wheel-gpu-dev.html

注意：你本地 Python 环境的 Python 版本将被用来寻找相应的 wheel 包。例如，如果你
本地的 Python 是 3.9，那么就会通过 pip 选择在 Python 3.9 上编译的 wheel 包并安装
。事实上，wheel 文件的名称包括 SINGA 版本、CUDA 版本和 Python 版本。因此
，`pip`知道要下载和安装哪个 wheel 文件。

参考 setup.py 文件顶部的注释，了解如何构建 wheel 包。

如果运行以下命令没有报错：

```shell
$ python -c "from singa import tensor"
```

那么 SINGA 就安装成功了。

## 使用 Docker

按照[说明](https://docs.docker.com/install/)在你的本地主机上安装 Docker。将您的
用户添加
到[docker 组](https://docs.docker.com/install/linux/linux-postinstall/)中，以便
在没有`sudo`的情况下运行 docker 命令。

1. 仅使用 CPU

```shell
$ docker run -it apache/singa:X.Y.Z-cpu-ubuntu16.04 /bin/bash
```

2. 要使用 GPU，在安装 Docker 后安装
   [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker)

```shell
$ nvidia-docker run -it apache/singa:X.Y.Z-cuda9.0-cudnn7.4.2-ubuntu16.04 /bin/bash
```

3. 关于 SINGA Docker 镜像（标签）的完整列表，请访
   问[docker hub site](https://hub.docker.com/r/apache/singa/)。对于每个 docker
   镜像，标签的命名为：

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

您可以使用本地构建工具或 conda-build 在本地主机操作系统上或在 Docker 容器中从源
代码[构建和安装 SINGA](build.md)。

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

  `_singa_wrap.so` 的文件夹是
  `~/miniconda3/lib/python3.10/site-packages/singa`。通常情况下，这个错误是由于
  依赖的库不匹配或缺失造成的，例如 cuDNN 或 protobuf。解决方法是创建一个新的虚拟
  环境，并在该环境中安装 SINGA，例如：

```shell
conda create -n singa
conda activate singa
conda install -c nusdbsystem -c conda-forge singa-cpu
```

- Q: 使用虚拟环境时，每次安装 SINGA 时，都会重新安装 numpy。但是，当我运
  行`import numpy`时，numpy 没有被使用。

  A: 这可能是由`PYTHONPATH`环境变量引起的，在使用虚拟环境时，应将其设置为空，以
  避免与虚拟环境的路径冲突。

- Q: 当我在 Mac OS X 中运行 SINGA 时，得到如下错误 "Fatal Python error:
  PyThreadState_Get: no current thread Abort trap: 6"

  A: 这个错误通常发生在系统中有多个 Python 版本的时候，例如，操作系统自带的版本
  和 Homebrew 安装的版本。SINGA 链接的 Python 必须与 Python 解释器相同。您可以通
  过`which python`来检查解释器 python 版本并通
  过`otool -L <path to _singa_wrap.so>` 检查 SINGA 链接的 Python，如果通过 conda
  安装 SINGA，这个问题应该可以解决。
