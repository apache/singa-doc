---
id: build
title: Build SINGA from Source
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

The source files can be downloaded either as a
[tar.gz file](https://dist.apache.org/repos/dist/dev/singa/) or as a Git
repository:

```shell
$ git clone https://github.com/apache/singa.git
$ cd singa/
```

If you want to contribute code to SINGA, refer to
[contribute-code page](contribute-code.md) for the steps and requirements.

## Use Conda to build SINGA

Conda-build is a build tool that installs the dependent libraries from
Anaconda Cloud and executes the build scripts.

To install conda-build (after installing conda)

```shell
conda install conda-build
```

### Build CPU Version

To build the CPU version of SINGA

```shell
conda build tool/conda/singa/
```

SINGA builds are tested via GitHub Actions on Ubuntu 24.04 and macOS 26
(Arm64). Refer to the
[GitHub Actions workflows](https://github.com/apache/singa/tree/master/.github/workflows)
for more information.

### Build GPU Version

To build the GPU version of SINGA, the build machine must have an NVIDIA GPU,
and the CUDA driver (>= 384.81), CUDA toolkit (>=9) and cuDNN (>=7) must be
installed. The following two Docker images provide the build environment:

1. apache/singa:conda-cuda9.0
2. apache/singa:conda-cuda10.0

Once the build environment is ready, you need to export the CUDA version
first, and then run the conda command to build SINGA

```shell
export CUDA=x.y (e.g. 9.0)
conda build tool/conda/singa/
```

### Post Processing

The location of the generated package file (`.tar.gz`) is shown on the screen.
The generated package can be installed directly,

```shell
conda install -c conda-forge --use-local <path to the package file>
```

or uploaded to Anaconda Cloud for others to download and install. You need to
register an account on Anaconda Cloud for
[uploading the package](https://docs.anaconda.com/anaconda-cloud/user-guide/getting-started/).

```shell
conda install anaconda-client
anaconda login
anaconda upload -l main <path to the package file>
```

After uploading the package to the cloud, you can see it on
the [Anaconda Cloud](https://anaconda.org/) website or via the following command

```shell
conda search -c <anaconda username> singa
```

Each specific SINGA package is identified by the version and build string. To
install a specific SINGA package, you need to provide all the information, e.g.,

```shell
conda install -c <anaconda username> -c conda-forge singa=2.1.0.dev=cpu_py36
```

To make the installation command simple, you can create the following additional
packages which depend on the latest CPU and GPU SINGA packages.

```console
# for singa-cpu
conda build tool/conda/cpu/  --python=3.6
conda build tool/conda/cpu/  --python=3.7
# for singa-gpu
conda build tool/conda/gpu/  --python=3.6
conda build tool/conda/gpu/  --python=3.7
```

Therefore, when you run

```shell
conda install -c <anaconda username> -c conda-forge singa-xpu
```

(`xpu` is either 'cpu' or 'gpu'), the corresponding real SINGA package is
installed as the dependent library.

## Use native tools to build SINGA on Ubuntu

Refer to SINGA
[Dockerfiles](https://github.com/apache/singa/blob/master/tool/docker/devel/ubuntu/cuda9/Dockerfile#L30)
for instructions on installing the dependent libraries on Ubuntu 16.04. You
can also create a Docker container using the [devel images]() and build SINGA
inside the container. To build SINGA with GPU, DNNL, Python and unit tests, run
the following commands

```shell
mkdir build    # at the root of singa folder
cd build
cmake -DENABLE_TEST=ON -DUSE_CUDA=ON -DUSE_DNNL=ON -DUSE_PYTHON3=ON ..
make
cd python
pip install .
```

The details of the CMake options are explained in the last section of this page.
The last command installs the Python package. You can also run
`pip install -e .`, which creates symlinks instead of copying the Python files
into the site-packages folder.

If SINGA is compiled with ENABLE_TEST=ON, you can run the unit tests by

```shell
$ ./bin/test_singa
```

You can see all the test cases and results. If SINGA passes all
tests, then you have successfully installed SINGA.

## Use native tools to build SINGA on CentOS 7

Building from source will be different for CentOS 7 as package names
differ. Follow the instructions given below.

### Installing dependencies

Basic packages/libraries

```shell
sudo yum install freetype-devel libXft-devel ncurses-devel openblas-devel blas-devel lapack devel atlas-devel kernel-headers unzip wget pkgconfig zip zlib-devel libcurl-devel cmake curl unzip dh-autoreconf git python-devel glog-devel protobuf-devel
```

For build-essential

```shell
sudo yum group install "Development Tools"
```

For installing SWIG

```shell
sudo yum install pcre-devel
wget http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz
tar xvzf swig-3.0.10.tar.gz
cd swig-3.0.10.tar.gz
./configure --prefix=${RUN}
make
make install
```

For installing gfortran

```shell
sudo yum install centos-release-scl-rh
sudo yum --enablerepo=centos-sclo-rh-testing install devtoolset-7-gcc-gfortran
```

For installing pip and other packages

```shell
sudo yum install epel-release
sudo yum install python-pip
pip install matplotlib numpy pandas scikit-learn pydot
```

### Installation

Follow steps 1-5 of _Use native tools to build SINGA on Ubuntu_

### Testing

You can run the unit tests by,

```shell
$ ./bin/test_singa
```

You can see all the testing cases with testing results. If SINGA passes all
tests, then you have successfully installed SINGA.

## Compile SINGA on Windows

Instructions for building on Windows with Python support can be found on the
[install-win page](install-win.md).

## More details about the compilation options

### USE_MODULES (deprecated)

If protobuf and openblas are not installed, you can compile SINGA together with
them

```shell
$ In SINGA ROOT folder
$ mkdir build
$ cd build
$ cmake -DUSE_MODULES=ON ..
$ make
```

CMake would download OpenBLAS and Protobuf (2.6.1) and compile them together
with SINGA.

You can use `ccmake ..` to configure the compilation options. If some dependent
libraries are not in the system default paths, you need to export the following
environment variables

```shell
export CMAKE_INCLUDE_PATH=<path to the header file folder>
export CMAKE_LIBRARY_PATH=<path to the lib file folder>
```

### USE_PYTHON

Option for compiling the Python wrapper for SINGA,

```shell
$ cmake -DUSE_PYTHON=ON ..
$ make
$ cd python
$ pip install .
```

### USE_CUDA

Users are encouraged to install the CUDA and
[cuDNN](https://developer.nvidia.com/cudnn) for running SINGA on GPUs to get
better performance.

SINGA has been tested over CUDA 9/10, and cuDNN 7. If cuDNN is installed into
non-system folder, e.g. /home/bob/local/cudnn/, the following commands should be
executed for CMake and the runtime to find it

```shell
$ export CMAKE_INCLUDE_PATH=/home/bob/local/cudnn/include:$CMAKE_INCLUDE_PATH
$ export CMAKE_LIBRARY_PATH=/home/bob/local/cudnn/lib64:$CMAKE_LIBRARY_PATH
$ export LD_LIBRARY_PATH=/home/bob/local/cudnn/lib64:$LD_LIBRARY_PATH
```

The CMake options for CUDA and cuDNN should be switched on

```shell
# Dependent libraries are already installed
$ cmake -DUSE_CUDA=ON ..
$ make
```

### USE_DNNL

Users can enable DNNL to enhance the performance of CPU computation.

The DNNL installation guide can be found
[here](https://github.com/intel/mkl-dnn#installation).

SINGA has been tested over DNNL v1.1.

To build SINGA with DNNL support:

```shell
# Dependent libraries are already installed
$ cmake -DUSE_DNNL=ON ..
$ make
```

### USE_OPENCL

SINGA uses opencl-headers and viennacl (version 1.7.1 or newer) for OpenCL
support, which can be installed via

```shell
# On Ubuntu 16.04
$ sudo apt-get install opencl-headers, libviennacl-dev
# On Fedora
$ sudo yum install opencl-headers, viennacl
```

Additionally, you will need the OpenCL Installable Client Driver (ICD) for the
platforms that you want to run OpenCL on.

- For AMD and NVIDIA GPUs, the driver package should also install the correct
  OpenCL ICD.
- For Intel CPUs and/or GPUs, get the driver from the
  [Intel website.](https://software.intel.com/en-us/articles/opencl-drivers)
  Note that the drivers provided on that website only support recent CPUs and
  Iris GPUs.
- For older Intel CPUs, you can use the `beignet-opencl-icd` package.

Note that running OpenCL on CPUs is not currently recommended because it is
slow. Memory transfer is on the order of whole seconds (1000's of ms on CPUs as
compared to 1's of ms on GPUs).

More information on setting up a working OpenCL environment may be found
[here](https://wiki.tiker.net/OpenCLHowTo).

If the package version of ViennaCL is not at least 1.7.1, you will need to build
it from source:

Clone [the repository from here](https://github.com/viennacl/viennacl-dev),
checkout the `release-1.7.1` tag and build it. Remember to add its directory to
`PATH` and the built libraries to `LD_LIBRARY_PATH`.

To build SINGA with OpenCL support (tested on SINGA 1.1):

```shell
$ cmake -DUSE_OPENCL=ON ..
$ make
```

### PACKAGE

This setting is used to build the Debian package. Set PACKAGE=ON and build the
package with the make command like this:

```shell
$ cmake -DPACKAGE=ON
$ make package
```

## FAQ

- Q: Error from 'import singa'

  A: Please check the detailed error from
  `python -c "from singa import _singa_wrap"`. Sometimes it is caused by the
  dependent libraries, e.g. multiple versions of protobuf, missing cuDNN
  libraries, or a NumPy version mismatch. The following steps show solutions for
  different cases

  1. Check cuDNN and CUDA. If cuDNN is missing or does not match the wheel
     version, you can download the correct version of cuDNN into ~/local/cudnn/
     and

     ```shell
     $ echo "export LD_LIBRARY_PATH=/home/<yourname>/local/cudnn/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
     ```

  2. If the problem is related to protobuf, you can install protobuf (3.6.1)
     from source into a local folder, such as ~/local/. Decompress the tar file,
     and then

     ```shell
     $ ./configure --prefix=/home/<yourname>local
     $ make && make install
     $ echo "export LD_LIBRARY_PATH=/home/<yourname>/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
     $ source ~/.bashrc
     ```

  3. If it cannot find other libraries, including Python, then create a virtual
     environment using `pip` or `conda`;

  4. If it is not caused by any of the above, go to the folder of
     `_singa_wrap.so`,

     ```shell
     $ python
     >> import importlib
     >> importlib.import_module('_singa_wrap')
     ```

     Check the error message. For example, if the NumPy version mismatches, the
     error message would be,

     ```shell
     RuntimeError: module compiled against API version 0xb but this version of numpy is 0xa
     ```

     Then you need to upgrade NumPy.

* Q: Error from running `cmake ..`, which cannot find the dependent libraries.

  A: If you haven't installed the libraries, install them. If you installed the
  libraries in a folder that is outside of the system folder, e.g. /usr/local,
  you need to export the following variables

  ```shell
  $ export CMAKE_INCLUDE_PATH=<path to your header file folder>
  $ export CMAKE_LIBRARY_PATH=<path to your lib file folder>
  ```

- Q: Error from `make`, e.g. the linking phase

  A: If your libraries are outside the system default paths, you need
  to export the following variables

  ```shell
  $ export LIBRARY_PATH=<path to your lib file folder>
  $ export LD_LIBRARY_PATH=<path to your lib file folder>
  ```

* Q: Error from header files, e.g. 'cblas.h no such file or directory exists'

  A: You need to include the folder of the cblas.h into CPLUS_INCLUDE_PATH,
  e.g.,

  ```shell
  $ export CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$CPLUS_INCLUDE_PATH
  ```

* Q: While compiling SINGA, I get the error `SSE2 instruction set not enabled`

  A: You can try the following command:

  ```shell
  $ make CFLAGS='-msse2' CXXFLAGS='-msse2'
  ```

* Q: I get `ImportError: cannot import name enum_type_wrapper` from
  google.protobuf.internal when I try to import .py files.

  A: You need to install the Python binding of protobuf, which could be
  installed via

  ```shell
  $ sudo apt-get install protobuf
  ```

  or from source

  ```shell
  $ cd /PROTOBUF/SOURCE/FOLDER
  $ cd python
  $ python setup.py build
  $ python setup.py install
  ```

* Q: When I build OpenBLAS from source, I am told that I need a Fortran
  compiler.

  A: You can compile OpenBLAS by

  ```shell
  $ make ONLY_CBLAS=1
  ```

  or install it using

  ```shell
  $ sudo apt-get install libopenblas-dev
  ```

* Q: When I build Protocol Buffers, it reports that `GLIBC++_3.4.20` is not
  found in `/usr/lib64/libstdc++.so.6`?

  A: This means the linker found libstdc++.so.6 but that library belongs to an
  older version of GCC than was used to compile and link the program. The
  program depends on code defined in the newer libstdc++ that belongs to the
  newer version of GCC, so the linker must be told how to find the newer
  libstdc++ shared library. The simplest way to fix this is to find the correct
  libstdc++ and export it to LD_LIBRARY_PATH. For example, if GLIBC++\_3.4.20 is
  listed in the output of the following command,

        $ strings /usr/local/lib64/libstdc++.so.6|grep GLIBC++

  then you just set your environment variable as

        $ export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

* Q: When I build glog, it reports that "src/logging_unittest.cc:83:20: error:
  ‘gflags’ is not a namespace-name"

  A: It may be that you have installed gflags with a different namespace, such
  as "google", so glog cannot find the `gflags` namespace. Because gflags is not
  necessary to build glog, you can change the configure.ac file to
  ignore gflags.

        1. cd to glog src directory
        2. change line 125 of configure.ac  to "AC_CHECK_LIB(gflags, main, ac_cv_have_libgflags=0, ac_cv_have_libgflags=0)"
        3. autoreconf

  After this, you can build glog again.

* Q: When using a virtual environment, every time I run `pip install`, it
  reinstalls NumPy. However, NumPy is not used when I run `import numpy`.

  A: It could be caused by the `PYTHONPATH`, which should be set to empty when
  you are using a virtual environment to avoid conflicts with the path of the
  virtual environment.

* Q: When compiling PySINGA from source, there is a compilation error because
  <numpy/objectarray.h> is missing

  A: Please install NumPy and export the path of its header files as

        $ export CPLUS_INCLUDE_PATH=`python -c "import numpy; print numpy.get_include()"`:$CPLUS_INCLUDE_PATH

* Q: When I run SINGA in Mac OS X, I got the error "Fatal Python error:
  PyThreadState_Get: no current thread Abort trap: 6"

  A: This error happens typically when you have multiple versions of Python on
  your system and you installed SINGA via pip (this problem is resolved for
  installation via conda), e.g., the version that comes with the OS and the one
  installed by Homebrew. The Python linked by PySINGA must be the same as the
  Python interpreter. You can check your interpreter by `which python` and check the Python
  linked by PySINGA via `otool -L <path to _singa_wrap.so>`. To fix this
  error, compile SINGA with the correct version of Python. In particular, if you
  build PySINGA from source, you need to specify the paths when invoking
  [cmake](http://stackoverflow.com/questions/15291500/i-have-2-versions-of-python-installed-but-cmake-is-using-older-version-how-do)

        $ cmake -DPYTHON_LIBRARY=`python-config --prefix`/lib/libpython2.7.dylib -DPYTHON_INCLUDE_DIR=`python-config --prefix`/include/python2.7/ ..

  If you installed PySINGA from binary packages, e.g. Debian packages or wheels,
  then you need to change the Python interpreter, e.g., reset the \$PATH to put the correct
  path of Python at the front position.
