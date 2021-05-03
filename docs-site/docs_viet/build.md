---
id: build
title: Cài đặt SINGA từ Nguồn (Source)
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Các tệp nguồn có thể được tải dưới dạng
[tar.gz file](https://dist.apache.org/repos/dist/dev/singa/), hoặc git repo

```shell
$ git clone https://github.com/apache/singa.git
$ cd singa/
```

Nếu bạn muốn tham gia đóng góp code cho SINGA, tham khảo
[mục contribute-code](contribute-code.md) với các bước làm và yêu cầu kĩ thuật.

## Sử dụng Conda để cài SINGA

Conda-build là phần mềm giúp cài đặt thư viện chương trình từ dữ liệu đám mây anaconda và thực hiện các tập lệnh tạo chương trình. 

Để cài đặt conda-build (sau khi cài conda)

```shell
conda install conda-build
```

### Tạo phiên bản CPU

Để tạo phiên bản CPU cho SINGA

```shell
conda build tool/conda/singa/
```

Lệnh trên đã được kiểm tra trên Ubuntu (14.04, 16.04 và 18.04) và macOS
10.11. Tham khảo [trang Travis-CI](https://travis-ci.org/apache/singa) để biết thêm chi tiết.

###  Tạo phiên bản GPU 

Để tạo phiên bản GPU cho SINGA, máy để cài phải có Nvida GPU, và CUDA driver (>= 384.81), phải được cài đặt CUDA toolkit (>=9) và cuDNN (>=7). Hai Docker images dưới đây cung cấp environment để chạy:

1. apache/singa:conda-cuda9.0
2. apache/singa:conda-cuda10.0

Sau khi environment để chạy đã sẵn sàng, bạn cần phải export phiên bản CUDA trước, sau đó chạy lệnh conda để cài SINGA: 

```shell
export CUDA=x.y (e.g. 9.0)
conda build tool/conda/singa/
```

### Sau khi chạy chương trình

Vị trí đặt tệp tin của gói chương trình được tạo (`.tar.gz`) hiển thị trên màn hình.
Gói chương trình được tạo có thể được cài đặt trực tiếp, 

```shell 
conda install -c conda-forge --use-local <path to the package file>
```

hoặc tải lên dữ liệu đám mây anaconda cloud để người dùng khác có thể tải và cài đặt. Bạn cần phải đăng kí một tài khoản trên anaconda để có thể 
[tải lên gói chương trình](https://docs.anaconda.com/anaconda-cloud/user-guide/getting-started/).

```shell
conda install anaconda-client
anaconda login
anaconda upload -l main <path to the package file>
```

Sau khi tải gói chương trình lên dữ liệu đám mây, bạn có thể tìm thấy gói trên website của
[Anaconda Cloud](https://anaconda.org/) hoặc qua lệnh

```shell
conda search -c <anaconda username> singa
```

Mỗi gói chương trình của SINGA đuợc nhận diện theo phiên bản hoặc dòng lệnh cài đặt. Để cài một gói chương trình SINGA cụ thể, bạn cần phải cung cấp toàn bộ thông tin, vd. 

```shell
conda install -c <anaconda username> -c conda-forge singa=2.1.0.dev=cpu_py36
```

Để cho lệnh cài đặt không phức tạp, bạn có thể tạo các gói chương trình bổ sung sau dựa trên các gói chương trình cho SINGA CPU và GPU mới nhất .

```console
# for singa-cpu
conda build tool/conda/cpu/  --python=3.6
conda build tool/conda/cpu/  --python=3.7
# for singa-gpu
conda build tool/conda/gpu/  --python=3.6
conda build tool/conda/gpu/  --python=3.7
```

Bởi vậy, khi bạn chạy 

```shell
conda install -c <anaconda username> -c conda-forge singa-xpu
```

(`xpu` nghĩa là hoặc 'cpu' hoặc 'gpu'), gói SINGA tương ứng thực sự được cài đặt như một library phụ thuộc.

## Sử dụng các phương tiện cơ bản để cài đặt SINGA trên Ubuntu

Tham khảo 
[Dockerfiles](https://github.com/apache/singa/blob/master/tool/docker/devel/ubuntu/cuda9/Dockerfile#L30)
của SINGA để xem hướng dẫn cài đặt các chương trình library phụ thuộc trên Ubuntu 16.04. Bạn có thể tạo một Docker container sử dụng [devel images]() và cài SINGA trong container. Để cài SINGA với GPU, DNNL, Python và unit tests, chạy lệnh theo hướng dẫn sau

```shell
mkdir build    # tại thư mục nguồn của singa
cd build
cmake -DENABLE_TEST=ON -DUSE_CUDA=ON -DUSE_DNNL=ON -DUSE_PYTHON3=ON ..
make
cd python
pip install .
```

Chi tiết các lựa chọn CMake đuợc giải thích ở phần cuối cùng của trang này. Câu lệnh cuối cùng để cài gói Python. Bạn cúng có thể chạy 
`pip install -e .`, để tạo symlinks thay vì copy các tâp tin Python vào mục site-package.

Nếu SINGA được compile với ENABLE_TEST=ON, bạn có thể chạy unit test bằng cách 

```shell
$ ./bin/test_singa
```

Bạn sẽ thấy tất cả các trường hợp test kèm theo kết quả test. Nếu SINGA thông qua tất cả các test, bạn đã cài đặt SINGA thành công. 

## Sử dụng các phương tiện cơ bản để cài đặt SINGA trên Centos7

Tạo từ nguồn sẽ khác trên Centos7 bởi tên của gói chương trình là khác nhau. Làm theo hướng dẫn dưới đây

### Cài các chương trình phụ thuộc (dependent libraries) 

Gói/chương trình cơ bản

```shell
sudo yum install freetype-devel libXft-devel ncurses-devel openblas-devel blas-devel lapack devel atlas-devel kernel-headers unzip wget pkgconfig zip zlib-devel libcurl-devel cmake curl unzip dh-autoreconf git python-devel glog-devel protobuf-devel
```

Cho build-essential

```shell
sudo yum group install "Development Tools"
```

Để cài đặt swig

```shell
sudo yum install pcre-devel
wget http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz
tar xvzf swig-3.0.10.tar.gz
cd swig-3.0.10.tar.gz
./configure --prefix=${RUN}
make
make install
```

Để cài đặt gfortran

```shell
sudo yum install centos-release-scl-rh
sudo yum --enablerepo=centos-sclo-rh-testing install devtoolset-7-gcc-gfortran
```

Để cài đặt pip và các gói chương trình khác 

```shell
sudo yum install epel-release
sudo yum install python-pip
pip install matplotlib numpy pandas scikit-learn pydot
```

### Cài đặt

Làm theo bước 1-5 của _Use native tools để cài SINGA trên Ubuntu_

### Kiểm tra (testing)

Bạn có thể chạy unit tests bằng cách,

```shell
$ ./bin/test_singa
```

Bạn sẽ thấy tất cả các trường hợp test kèm theo kết quả test. Nếu SINGA thông qua tất cả các test, bạn đã cài đặt SINGA thành công. 

## Compile SINGA trên Windows

Hướng dẫn cài đặt trên Windows với Python vui lòng xem tại
[mục install-win](install-win.md).

## Chi tiết bổ sung về các lựa chọn biên dịch (compilation) 

### USE_MODULES (không còn sử dụng)

Nếu protobuf và openblas không được cài đặt, bạn có thể compile SINGA cùng với chúng. 

```shell
$ In SINGA ROOT folder
$ mkdir build
$ cd build
$ cmake -DUSE_MODULES=ON ..
$ make
```

cmake sẽ tải OpenBlas và Protobuf (2.6.1) sau đó compile cùng với SINGA.

Bạn có thể sử dụng `ccmake ..` để định dạng các lựa chọn biên dịch (compilation). Nếu chương trình phụ thuộc (dependent libraries) nào không có trong đường dẫn hệ thống mặc định,bạn cần phải export các biến environment sau: 

```shell
export CMAKE_INCLUDE_PATH=<path to the header file folder>
export CMAKE_LIBRARY_PATH=<path to the lib file folder>
```

### USE_PYTHON

Là lựa chọn để compile Python wrapper cho SINGA,

```shell
$ cmake -DUSE_PYTHON=ON ..
$ make
$ cd python
$ pip install .
```

### USE_CUDA

Chúng tôi khuyến khích cài đặt CUDA và
[cuDNN](https://developer.nvidia.com/cudnn) để chạy SINGA trên GPUs nhằm có kết quả tốt nhất. 

SINGA đã được kiểm nghiệm chạy trên CUDA 9/10, và cuDNN 7. Nếu cuDNN được cài đặt vào thư mục không thuộc hệ thống, vd. /home/bob/local/cudnn/, cần chạy các lệnh sau để cmake và runtime có thể tìm được

```shell
$ export CMAKE_INCLUDE_PATH=/home/bob/local/cudnn/include:$CMAKE_INCLUDE_PATH
$ export CMAKE_LIBRARY_PATH=/home/bob/local/cudnn/lib64:$CMAKE_LIBRARY_PATH
$ export LD_LIBRARY_PATH=/home/bob/local/cudnn/lib64:$LD_LIBRARY_PATH
```

Các lựa chọn cmake cho CUDA và cuDNN cần được kích hoạt

```shell
# các Dependent libs đã được cài đặt
$ cmake -DUSE_CUDA=ON ..
$ make
```

### USE_DNNL

Người dùng có thể kích hoạt DNNL để cải thiện hiệu quả cho chương trình CPU.

Hướng dẫn cài đặt DNNL
[tại đây](https://github.com/intel/mkl-dnn#installation).

SINGA đã được thử nghiệm chạy trên DNNL v1.1.

Để chạy SINGA với DNNL:

```shell
# các Dependent libs đã được cài đặt
$ cmake -DUSE_DNNL=ON ..
$ make
```

### USE_OPENCL

SINGA sử dụng opencl-headers và viennacl (phiên bản 1.7.1 hoặc mới hơn) để hỗ trợ OpenCL, có thể được cài đặt qua 

```shell
# Trên Ubuntu 16.04
$ sudo apt-get install opencl-headers, libviennacl-dev
# Trên Fedora
$ sudo yum install opencl-headers, viennacl
```

Bên cạnh đó, bạn cần OpenCL Installable Client Driver (ICD) cho nền tảng mà bạn muốn chạy OpenCL.

- Với AMD và nVidia GPUs, driver package nên cài đúng bản OpenCL ICD.
- Với Intel CPUs và/hoặc GPUs, có thể tải driver từ
  [Intel website.](https://software.intel.com/en-us/articles/opencl-drivers)
  Lưu ý rằng driver này chỉ hỗ trợ các phiên bản mới của CPUs và Iris GPUs.
- Với các bản Intel CPUs cũ hơn, bạn có thể sử dụng gói `beignet-opencl-icd`.

Lưu ý rằng chạy OpenCL trên CPUs không được khuyến khích bởi tốc độ chậm. Di chuyển bộ nhớ theo trình tự tính theo từng giây (1000's của ms trên CPUs so với 1's của ms trên GPUs).

Có thể xem thêm thông tin về cách thiết lập environment có chạy OpenCL tại [đây](https://wiki.tiker.net/OpenCLHowTo).

Nếu phiên bản của gói chương trình ViennaCL thấp hơn 1.7.1, bạn cần phải tạo từ nguồn:

Clone [nguồn tại đây](https://github.com/viennacl/viennacl-dev),
chọn (checkout) tag `release-1.7.1` để cài đặt. Bạn cần nhớ thêm đường dẫn vào phần `PATH` và tạo libraries vào `LD_LIBRARY_PATH`.

Để cài SINGA với hỗ trợ OpenCL (đã thử trên SINGA 1.1):

```shell
$ cmake -DUSE_OPENCL=ON ..
$ make
```

### GÓI CHƯƠNG TRÌNH (PACKAGE)

Cài đặt này được sử dụng để tạo gói chương trình Debian package. Để PACKAGE=ON và tạo gói chương trình với lệnh như sau: 

```shell
$ cmake -DPACKAGE=ON
$ make package
```

## Câu hỏi thường gặp (Q&A)

- Q: Gặp lỗi khi 'import singa'

  A: Vui lòng kiểm tra chi tiết lỗi từ `python -c "from singa import _singa_wrap"`. Đôi khi lỗi xảy ra bởi các dependent libraries, vd. protobuf có nhiều phiên bản, nếu thiếu cudnn, phiên bản numpy sẽ không tương thích. Các bước sau đưa ra giải pháp cho từng trường hợp: 

  1. Kiểm tra cudnn và cuda. Nếu thiếu cudnn hoặc không tương thích với phiên bản của wheel, bạn có thể tải phiên bản đúng của cudnn vào thư mục ~/local/cudnn/ và

     ```shell
     $ echo "export LD_LIBRARY_PATH=/home/<yourname>/local/cudnn/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
     ```

  2. Nếu lỗi liên quan tới protobuf. Bạn có thể cài đặt (3.6.1) từ nguồn vào một thư mục trong máy của bạn(local). chẳng hạn ~/local/; Giải nén file tar, sau đó

     ```shell
     $ ./configure --prefix=/home/<yourname>local
     $ make && make install
     $ echo "export LD_LIBRARY_PATH=/home/<yourname>/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
     $ source ~/.bashrc
     ```

  3. Nếu không tìm được libs nào bao gồm python, thì taọ virtual env sử dụng `pip` hoặc `conda`;

  4. Nếu lỗi không do các nguyên nhân trên thì đi tới thư mục của `_singa_wrap.so`,

     ```shell
     $ python
     >> import importlib
     >> importlib.import_module('_singa_wrap')
     ```

    kiểm tra thông báo lỗi. Ví dụ nếu phiên bản numpy không tương thích, thông báo lỗi sẽ là 

     ```shell
     RuntimeError: module compiled against API version 0xb but this version of numpy is 0xa
     ```

    sau đó bạn cần phải nâng cấp numpy. 

* Q: Lỗi khi chạy `cmake ..`, không tìm được dependent libraries.

  A: Nếu bạn vẫn chưa cài đặt libraries đó, thì cài đặt chúng. Nếu bạn cài libraries trong thư mục bên ngoài thư mục system, chẳng hạn như /usr/local, bạn cần export các biến sau đây 

  ```shell
  $ export CMAKE_INCLUDE_PATH=<path to your header file folder>
  $ export CMAKE_LIBRARY_PATH=<path to your lib file folder>
  ```

- Q: Lỗi từ `make`, vd. linking phase

  A: Nếu libraries nằm trong thư mục không phải là thư mục system mặc định trong đường dẫn, bạn cần export các biến sau 

  ```shell
  $ export LIBRARY_PATH=<path to your lib file folder>
  $ export LD_LIBRARY_PATH=<path to your lib file folder>
  ```

* Q: Lỗi từ các tệp tin headers vd. 'cblas.h no such file or directory exists'

  A: Bạn cần bao gồm các thư mục cblas.h vào CPLUS_INCLUDE_PATH,
  e.g.,

  ```shell
  $ export CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$CPLUS_INCLUDE_PATH
  ```

* Q: Khi compile SINGA, gặp lỗi `SSE2 instruction set not enabled`

  A: Bạn có thể thử lệnh sau:

  ```shell
  $ make CFLAGS='-msse2' CXXFLAGS='-msse2'
  ```

* Q:Gặp lỗi `ImportError: cannot import name enum_type_wrapper` từ google.protobuf.internal khi tôi cố gắng import các tệp tin dạng .py.

  A: Bạn cần cài đặt python cho protobuf, có thể cài đặt qua

  ```shell
  $ sudo apt-get install protobuf
  ```

 hoặc từ nguồn

  ```shell
  $ cd /PROTOBUF/SOURCE/FOLDER
  $ cd python
  $ python setup.py build
  $ python setup.py install
  ```

* Q: Khi tôi tạo OpenBLAS từ nguồn, tôi gặp yêu cầu cần phải có Fortran compiler.

  A: Bạn có thể compile OpenBLAS bằng cách

  ```shell
  $ make ONLY_CBLAS=1
  ```

  hoặc cài dặt sử dụng

  ```shell
  $ sudo apt-get install libopenblas-dev
  ```

* Q: Khi tôi tạo protocol buffer, thì bị thông báo `GLIBC++_3.4.20` không được tìm thấy trong `/usr/lib64/libstdc++.so.6`?

  A: Nghĩa là linker trong libstdc++.so.6 nhưng chương trình này thuộc về một phiên bản cũ hơn của GCC đã được dùng để compile và link chương trình. Chương trình phụ thuộc vào code viết trong phiên bản libstdc++ cập nhật thuộc về phiên bản mới hơn của GCC, vì vậy linker phải chỉ ra cách để cài phiên bản libstdc++ mới hơn được chia sẻ trong library. Cách đơn giản nhất để sửa lỗi này là tìm phiên bản đúng cho libstdc++ và export nó vào LD_LIBRARY_PATH. Ví dụ nếu GLIBC++\_3.4.20 có trong output của lệnh sau

        $ strings /usr/local/lib64/libstdc++.so.6|grep GLIBC++

  thì bạn chỉ cần tạo biến environment 

        $ export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

* Q: Khi tạo glog, nhận thông báo "src/logging_unittest.cc:83:20: error:
  ‘gflags’ is not a namespace-name"

  A: Có thể do bạn đã cài gflags với một namespace khác như là "google". vì thế glog không thể tìm thấy 'gflags' namespace. Do cài glog thì không cần phải có gflags. Nên bạn cần sửa tệp tin configure.ac thành ignore gflags.

        1. cd to glog src directory
        2. change line 125 of configure.ac  to "AC_CHECK_LIB(gflags, main, ac_cv_have_libgflags=0, ac_cv_have_libgflags=0)"
        3. autoreconf

  Sau đó bạn có thể cài lại glog.

* Q: Khi sử dụng virtual environment, bất cứ khi nào tôi chạy pip install, numpy sẽ tự cài lại numpy. Tuy nhiên, numpy này không được sử dụng khi tôi `import numpy`

  A: Lỗi có thể gây ra bởi `PYTHONPATH` vốn nên được để trống (empty) khi bạn sử dụng virtual environment nhằm tránh conflict với đường dẫn của virtual environment.

* Q: Khi compile PySINGA từ nguồn, có lỗi compilation do thiếu <numpy/objectarray.h>

  A: Vui lòng cài đặt numpy và export đường dẫn của tệp tin numpy header như sau

        $ export CPLUS_INCLUDE_PATH=`python -c "import numpy; print numpy.get_include()"`:$CPLUS_INCLUDE_PATH

* Q: Khi chạy SINGA trên Mac OS X, tôi gặp lỗi "Fatal Python error:
  PyThreadState_Get: no current thread Abort trap: 6"

  A: Lỗi này thường xảy ra khi bạn có nhiều phiên bản Python trong hệ thống, và bạn cài SINGA qua pip (vấn đề này có thể được giải quyết nếu cài đặt bằng conda), vd. một bên qua OS và một bên cài đặt qua Homebrew. Python dùng trong PySINGA phải là Python interpreter. Bạn có thể kiểm tra trình thông dịch (interpreter) của mình bằng `which python` và kiểm tra bản Python dùng trong PySINGA qua `otool -L <path to _singa_wrap.so>`. Để sửa lỗi này, bạn compile SINGA với đúng phiên bản mà SINGA cần. Cụ thể, nếu bạn tạo PySINGA từ nguồn, bạn cần cụ thể đường dẫn khi gọi [cmake](http://stackoverflow.com/questions/15291500/i-have-2-versions-of-python-installed-but-cmake-is-using-older-version-how-do)

        $ cmake -DPYTHON_LIBRARY=`python-config --prefix`/lib/libpython2.7.dylib -DPYTHON_INCLUDE_DIR=`python-config --prefix`/include/python2.7/ ..

  Nếu cài đặt PySINGA từ gói binary packages, vd. debian hay wheel, thì bạn cần thay đổi trình thông dịch của python (python interpreter), vd., reset \$PATH để đường dẫn dúng của Python ở đằng trước. 
