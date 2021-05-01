---
id: contribute-code
title: Tham gia viết code
---

<!-- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License. -->

## Định dạng mã code

Nền tảng code của SINGA tuân theo định dạng Google cho cả code [CPP](http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml) và
[Python](http://google.github.io/styleguide/pyguide.html).

Một cách đơn giản để thực hiện định dạng lập trình Google là sử dụng linting và các công cụ định dạng trong Visual Studio Code editor:

- [C/C++ extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
- [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [cpplint extension](https://marketplace.visualstudio.com/items?itemName=mine.cpplint)
- [Clang-Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)

Sau khi cài extensions, chỉnh sửa tệp tin `settings.json`.

```json
{
  "[cpp]": {
    "editor.defaultFormatter": "xaver.clang-format"
  },
  "cpplint.cpplintPath": "path/to/cpplint",

  "editor.formatOnSave": true,
  "python.formatting.provider": "yapf",
  "python.linting.enabled": true,
  "python.linting.lintOnSave": true,
  "clang-format.language.cpp.style": "google",
  "python.formatting.yapfArgs": ["--style", "{based_on_style: google}"]
}
```

Dựa vào nền tảng bạn đang sử dụng. tệp tin user settings được đặt tại đây:

1. Windows %APPDATA%\Code\User\settings.json
2. macOS "\$HOME/Library/Application Support/Code/User/settings.json"
3. Linux "\$HOME/.config/Code/User/settings.json"

Thông số cấu hình cụ thể có trong các tệp tin config file tuơng ứng. Những công cụ này sẽ tự động tìm kiếm các tập tin cấu hình configuration files trong root của dự án, vd. `.pylintrc`.

#### Cài đặt công cụ

Tốt nhất là tất cả người tham gia viết mã code sử dụng cùng một phiên bản công cụ định dạng mã code (clang-format 9.0.0 và yapf 0.29.0), để tất cả định dạng mã code sẽ giống nhau dù thuộc về các PRs khác nhau, nhằm tránh tạo conflict trong github pull request.

Trước tiên, cài đặt LLVM 9.0 cung cấp clang-format phiên bản 9.0.0. Trang tải LLVM là:

- [LLVM](http://releases.llvm.org/download.html#9.0.0)

  - Trên Ubuntu

    ```sh
    sudo apt-get install clang-format-9
    ```

  - Trên Windows. Tải gói pre-built và cài đặt

Sau đó, cài cpplint, pylint và yapf

- Ubuntu or OSX:

  ```
  $ sudo pip install cpplint
  $ which cpplint
  /path/to/cpplint

  $ pip install yapf==0.29.0
  $ pip install pylint
  ```

- Windows: Cài Anaconda cho gói quản lý package management.

  ```
  $ pip install cpplint
  $ where cpplint
  C:/path/to/cpplint.exe

  $ pip install yapf==0.29.0
  $ pip install pylint
  ```

#### Sử dụng

- Sau khi kích hoạt, linting sẽ tự động được áp dụng khi bạn chỉnh sửa các tập tin mã code nguồn (source code file). Lỗi và cảnh báo sẽ hiển thị trên thanh Visual Studio Code `PROBLEMS`.
- Định dạng mã code có thể thực hiện bằng cách sử dụng Command Palette(`Shift+Ctrl+P` cho
  Windows hay `Shift+Command+P` cho OSX) và gõ `Format Document`.

#### Gửi 

Bạn cần phải chữa lỗi định dạng nếu có trước khi gửi đi pull requests.

## Tạo Environment

Chúng tôi khuyến khích dùng Visual Studio Code để viết code. Có thể cài các Extensions như Python, C/C++,
Code Spell Checker, autoDocstring, vim, Remote Development. Tham khảo cấu hình (vd., `settings.json`) của các extensions [tại đây](https://gist.github.com/nudles/3d23cfb6ffb30ca7636c45fe60278c55).

Nếu bạn cập nhật mã code CPP, bạn cần recompile SINGA
[từ nguồn](./build.md). Nên sử dụng các công cụ cài đặt cơ bản trong `*-devel` Docker images hay `conda build`.

Nếu bạn chỉ cập nhật mã code Python, bạn cần cài đặt SINGAS một lần, sau đó copy các tập tin Python cập nhật để thay thế chúng trong thư mục cài đặt Python,

```shell
cp python/singa/xx.py  <path to conda>/lib/python3.7/site-packages/singa/
```

## Trình Tự

Vui lòng tham khảo mục [git workflow](./git-workflow.md).
