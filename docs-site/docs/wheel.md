---
id: wheel
title: Wheel
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## Installation Instructions

The wheel package of SINGA is installed via `pip`. Depending on the hardware
environment, here are two ways to install the wheel package.

- CPU only

```bash
pip install singa-<version> -f http://singa.apache.org/docs/next/wheel.html
```

The `<version>` field should be replaced, e.g., `3.0.0.dev1`. The available
SINGA versions are listed below.

- With CUDA and cuDNN

```bash
pip install singa-<version>+cuda<cu_version> -f http://singa.apache.org/docs/next/wheel.html
```

The `<version>` field should be replaced with specific SINGA version; The
`<cu_version>` field should be replaced with specific CUDA version; For example,
`singa-3.0.0.dev1+cuda10.2` is the full version specification for one package.
The available combinations of SINGA version and CUDA version are listed below.

Note: the Python version of your local Python environment will be used to find
the corresponding wheel package. For example, if your local Python is 3.6, then
the wheel package compiled on Python 3.6 will be selected by pip and installed.
In fact, the wheel file's name include SINGA version, CUDA version and Python
version. Therefore, `pip` knows which wheel file to download and install.

## Building Instructions

Refer to the comments at the top of the `setup.py` file.

## SINGA-3.0.0.dev1

- [CPU only, Python 3.6](https://singa-wheel.s3-ap-southeast-1.amazonaws.com/singa-3.0.0.dev1-cp36-cp36m-manylinux2014_x86_64.whl)
- [CPU only, Python 3.7](https://singa-wheel.s3-ap-southeast-1.amazonaws.com/singa-3.0.0.dev1-cp37-cp37m-manylinux2014_x86_64.whl)
- [CPU only, Python 3.8](https://singa-wheel.s3-ap-southeast-1.amazonaws.com/singa-3.0.0.dev1-cp38-cp38-manylinux2014_x86_64.whl)
- [CUDA10.2, cuDNN 7.6.5, Python 3.6](https://singa-wheel.s3-ap-southeast-1.amazonaws.com/singa-3.0.0.dev1%2Bcuda10.2-cp36-cp36m-manylinux2014_x86_64.whl)
- [CUDA10.2, cuDNN 7.6.5, Python 3.7](https://singa-wheel.s3-ap-southeast-1.amazonaws.com/singa-3.0.0.dev1-cp37-cp37m-manylinux2014_x86_64.whl)
- [CUDA10.2, cuDNN 7.6.5, Python 3.8](https://singa-wheel.s3-ap-southeast-1.amazonaws.com/singa-3.0.0.dev1-cp38-cp38-manylinux2014_x86_64.whl)
