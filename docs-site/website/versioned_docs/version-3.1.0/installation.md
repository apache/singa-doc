---
id: version-3.1.0-installation
title: Installation
original_id: installation
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## Using Conda

Conda is a package manager for Python, CPP and other packages.

Currently, SINGA has conda packages for Linux and MacOSX.
[Miniconda3](https://conda.io/miniconda.html) is recommended to use with SINGA.
After installing miniconda, execute the one of the following commands to install
SINGA.

**It is preferable to use python 3.6 as the conda virtual environment.**

1. CPU only
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ntkhi-Z6XTR8WYPXiLwujHd2dOm0772V?usp=sharing)

```shell
$ conda install -c nusdbsystem -c conda-forge singa-cpu=3.1.0
```

Please note that we need to specify the version of singa-cpu during the installation.

2. GPU with CUDA and cuDNN (CUDA driver >=384.81 is required)
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1do_TLJe18IthLOnBOsHCEe-FFPGk1sPJ?usp=sharing)

```shell
$ conda install -c nusdbsystem -c conda-forge singa-gpu=3.1.0
```

Please note that we need to specify the version of singa-gpu during the installation.

3. GPU with CUDA and cuDNN (CUDA driver >=384.81 is required), and also with the distributed training module

```shell
$ conda install -c nusdbsystem -c conda-forge singa-dist=3.1.0
```

Please note that we need to specify the version of singa-dist during the installation.

4. Install a specific version of SINGA. The following command lists all the
   available SINGA packages.

```shell
$ conda search -c nusdbsystem singa

Loading channels: done
# Name                       Version           Build  Channel
singa                      3.1.0.rc2        cpu_py36  nusdbsystem
singa                      3.1.0.rc2 cudnn7.6.5_cuda10.2_py36  nusdbsystem
singa                      3.1.0.rc2 cudnn7.6.5_cuda10.2_py37  nusdbsystem
```

<!--- > Please note that using the nightly built images is not recommended except for SINGA development and testing. Using stable releases is recommended. -->

The following command installs a specific version of SINGA,

```shell
$ conda install -c nusdbsystem -c conda-forge singa=X.Y.Z=cpu_py36
```

If there is no error message from

```shell
$ python -c "from singa import tensor"
```

then SINGA is installed successfully.

## Using Pip

1. CPU only
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17RA056Brwk0vBQTFaZ-l9EbqwADO0NA9?usp=sharing)

```bash
pip install singa -f http://singa.apache.org/docs/next/wheel-cpu.html --trusted-host singa.apache.org
```

You can install a specific version of SINGA via `singa==<version>`, where the
`<version>` field should be replaced, e.g., `3.1.0`. The available SINGA
versions are listed at the link.

To install the latest develop version, replace the link with
http://singa.apache.org/docs/next/wheel-cpu-dev.html

2. GPU With CUDA and cuDNN
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1W30IPCqj5fG8ADAQsFqclaCLyIclVcJL?usp=sharing)

```bash
pip install singa -f http://singa.apache.org/docs/next/wheel-gpu.html --trusted-host singa.apache.org
```

You can also configure SINGA version and the CUDA version, like
`singa==3.1.0+cuda10.2`. The available combinations of SINGA version and CUDA
version are listed at the link.

To install the latest develop version, replace the link with
http://singa.apache.org/docs/next/wheel-gpu-dev.html

Note: the Python version of your local Python environment will be used to find
the corresponding wheel package. For example, if your local Python is 3.6, then
the wheel package compiled on Python 3.6 will be selected by pip and installed.
In fact, the wheel file's name include SINGA version, CUDA version and Python
version. Therefore, `pip` knows which wheel file to download and install.

Refer to the comments at the top of the `setup.py` file for how to build the
wheel packages.

## Using Docker

Install Docker on your local host machine following the
[instructions](https://docs.docker.com/install/). Add your user into the
[docker group](https://docs.docker.com/install/linux/linux-postinstall/) to run
docker commands without `sudo`.

1. CPU-only.

```shell
$ docker run -it apache/singa:X.Y.Z-cpu-ubuntu16.04 /bin/bash
```

2. With GPU enabled. Install
   [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) after install
   Docker.

```shell
$ nvidia-docker run -it apache/singa:X.Y.Z-cuda9.0-cudnn7.4.2-ubuntu16.04 /bin/bash
```

3. For the complete list of SINGA Docker images (tags), visit the
   [docker hub site](https://hub.docker.com/r/apache/singa/). For each docker
   image, the tag is named as

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

## From source

You can [build and install SINGA](build.md) from the source code using native
building tools or conda-build, on local host OS or in a Docker container.

## FAQ

- Q: Error from `from singa import tensor`

  A: Check the detailed error from

  ```shell
  python -c  "from singa import _singa_wrap"
  # go to the folder of _singa_wrap.so
  ldd path to _singa_wrap.so
  python
  >> import importlib
  >> importlib.import_module('_singa_wrap')
  ```

  The folder of `_singa_wrap.so` is like
  `~/miniconda3/lib/python3.7/site-packages/singa`. Normally, the error is
  caused by the mismatch or missing of dependent libraries, e.g. cuDNN or
  protobuf. The solution is to create a new virtual environment and install
  SINGA in that environment, e.g.,

  ```shell
  conda create -n singa
  conda activate singa
  conda install -c nusdbsystem -c conda-forge singa-cpu
  ```

- Q: When using virtual environment, every time I install SINGA, numpy would be
  reinstalled. However, the numpy is not used when I run `import numpy`

  A: It could be caused by the `PYTHONPATH` environment variable which should be
  set to empty when you are using virtual environment to avoid the conflicts
  with the path of the virtual environment.

- Q: When I run SINGA in Mac OS X, I got the error "Fatal Python error:
  PyThreadState_Get: no current thread Abort trap: 6"

  A: This error happens typically when you have multiple versions of Python in
  your system, e.g, the one comes with the OS and the one installed by Homebrew.
  The Python linked by SINGA must be the same as the Python interpreter. You can
  check your interpreter by `which python` and check the Python linked by SINGA
  via `otool -L <path to _singa_wrap.so>`. This problem should be resolved if
  SINGA is installed via conda.
