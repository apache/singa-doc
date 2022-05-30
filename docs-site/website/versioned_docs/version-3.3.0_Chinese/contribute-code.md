---
id: version-3.3.0_Chinese-contribute-code
title: How to Contribute Code
original_id: contribute-code
---

<!-- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License. -->

## 代码风格

SINGA代码库在[CPP](http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml)和[Python](http://google.github.io/styleguide/pyguide.html)代码中都遵循Google风格。

强制执行Google编码风格的一个简单方法是使用Visual Studio Code编辑器中的linting和格式化工具:

- [C/C++扩展](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
- [Python扩展](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [cpplint扩展](https://marketplace.visualstudio.com/items?itemName=mine.cpplint)
- [Clang-Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)

安装扩展后，编辑`settings.json`文件：

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

根据您的操作系统，用户设置文件位于以下位置：

1. Windows %APPDATA%\Code\User\settings.json
2. macOS "\$HOME/Library/Application Support/Code/User/settings.json"
3. Linux "\$HOME/.config/Code/User/settings.json"

配置是在相应的配置文件中指定的。而这些工具会自动查找项目根目录下的配置文件，比如`.pylintrc`。

#### 安装必要工具

最理想的情况是所有贡献者都使用相同版本的代码格式化工具（clang-format 9.0.0和yapf 0.29.0），这样在不同PR中的代码格式化就会完全相同，从而摆脱github pull request冲突。

首先，安装LLVM 9.0，它提供了clang-format 9.0.0版本，LLVM的下载页面如下:

- [LLVM](http://releases.llvm.org/download.html#9.0.0)

  - Ubuntu系统：

    ```sh
    sudo apt-get install clang-format-9
    ```

  - Windows系统，下载预编译包并安装。

然后，安装cpplint, pylint和yapf

- Ubuntu或OSX:

  ```
  $ sudo pip install cpplint
  $ which cpplint
  /path/to/cpplint

  $ pip install yapf==0.29.0
  $ pip install pylint
  ```

- Windows: 安装Anaconda进行包管理

  ```
  $ pip install cpplint
  $ where cpplint
  C:/path/to/cpplint.exe

  $ pip install yapf==0.29.0
  $ pip install pylint
  ```

#### 使用

- 配置后，在编辑源代码文件时，linting会自动启用。错误和警告会在Visual Studio Code `PROBLEMS`面板中列出。
- 代码格式化可以通过调出Command Palette(Windows中为`Shift+Ctrl+P`，OS X中为`Shift+Command+P`)并输入`Format Document`来完成。

#### 提交

修正格式错误以后就可以提交pull request了。

## 开发环境

推荐使用Visual Studio Code作为编辑器。可以安装Python、C/C++、代码拼写检查器、autoDocstring、vim、远程开发等扩展。这些扩展的参考配置（即`settings.json`）可以在[这里](https://gist.github.com/nudles/3d23cfb6ffb30ca7636c45fe60278c55)查看。

如果更新CPP代码，需要从[源文件](./build.md)重新编译SINGA。建议使用`*-devel Docker`镜像中的原生构建工具或使用`conda build`。

如果要只更新Python代码，您可以安装一次SINGA，然后复制更新后的Python文件来替换Python安装文件夹中的文件。

```shell
cp python/singa/xx.py  <path to conda>/lib/python3.7/site-packages/singa/
```

## 工作流程

请参阅[git工作流程页面](./git-workflow.md).
