---
id: contribute-code
title: How to Contribute Code
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## Coding Style

The SINGA codebase follows the Google Style for both [CPP](http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml) and [Python](http://google.github.io/styleguide/pyguide.html) code.

A simple way to enforce the Google coding styles is to use the linting and formating tools in the Visual Studio Code editor:

- [C/C++ extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
- [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [cpplint extension](https://marketplace.visualstudio.com/items?itemName=mine.cpplint)
- [Clang-Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)

Once the extensions are installed, edit the `settings.json` file.

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

Depending on your platform, the user settings file is located here:

1. Windows %APPDATA%\Code\User\settings.json
2. macOS "\$HOME/Library/Application Support/Code/User/settings.json"
3. Linux "\$HOME/.config/Code/User/settings.json"

Configurations are specified in corresponding config files. And these tools would look up for configuration files in the root of the project automatically, e.g. `.pylintrc`.

#### Tool Installation

It is ideal when all the contributors uses the same version of code formatting tool (clang-format 9.0.0 and yapf 0.29.0), so that all code formatting in different PRs would be identical to get rid of github pull request conflicts.

First, install LLVM 9.0 which provides clang-format version 9.0.0. The download page of LLVM is:

- [LLVM](http://releases.llvm.org/download.html#9.0.0)

Second, install cpplint, pylint and yapf

- OSX:

  ```
  $ sudo pip install cpplint
  $ which cpplint
  /path/to/cpplint

  $ pip install yapf==0.29.0
  $ pip install pylint
  ```

- Windows: Install Anaconda for package management.

  ```
  $ pip install cpplint
  $ where cpplint
  C:/path/to/cpplint.exe

  $ pip install yapf==0.29.0
  $ pip install pylint
  ```

#### Usage

- After the configuration, linting should be automatically applied when editing source code file. Errors and warnings are listed in Visual Studio Code `PROBLEMS` panel.
- Code Formatting could be done by bringing up Command Palette(`Shift+Ctrl+P` in Windows or `Shift+Command+P` in OSX) and type `Format Document`.

#### Submission

You need to fix the format errors before submitting the pull requests.

## Developing Environment

Visual Studio Code is recommended as the editor. Extensions like Python, C/C++, Code Spell Checker, autoDocstring, vim, Remote Development could be installed. A reference configuration (i.e., `settings.json`) of these extensions is [here](https://gist.github.com/nudles/3d23cfb6ffb30ca7636c45fe60278c55).

If you update the CPP code, you need to recompile SINGA [from source](./build.md). It is recommended to use the native building tools in the `*-devel` Docker images or `conda build`.

If you only update the Python code, you can install SINGAS once, and then copy the updated Python files to replace those in the Python installation folder,

```shell
cp python/singa/xx.py  <path to conda>/lib/python3.7/site-packages/singa/
```

## Workflow

Please refer to the [git workflow page](./git-workflow).
