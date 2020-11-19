---
id: version-3.1.0_Chinese-contribute-docs
title: How to Contribute to Documentation
original_id: contribute-docs
---

<!-- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License. -->

文档有两种类型，即markdown文件和API使用参考。本指南介绍了一些工具，并指导如何准备md文件和API注释。

md文件将通过[Docusaurus](https://docusaurus.io/)构建成HTML页面；API注释（来自源代码）将用于使用Sphinx（对应Python）和Doxygen（对应CPP）生成API参考页面。

## Markdown文件

请尽量遵循[Google Documentation style](https://developers.google.com/style)。例如：

1. 删除指令中的"please"。如：'Please click...' VS 'Click...'。
2. 遵循标准的[大小写规则](https://owl.purdue.edu/owl/general_writing/mechanics/help_with_capitals.html)。
3. 在说明中用"you"代替"we"。
4. 使用现在时态，避免使用`will`。
5. 尽量使用主动语态而不是被动。

此外，为了使文件一致：

1. 句子不宜过长, e.g., 长度<=80
2. 尽量使用相对路径，假设我们在repo的根目录下，例如，`doc-site/docs`指的是`singa-doc/docs-site/docs`。
3. 使用背标将命令、路径、类函数和变量亮出来，例如，`Tensor`, `singa-doc/docs-site/docs`。
4. 为了突出其他术语/概念，使用 _斜体_ or **加粗**


本项目使用的[prettier tool](https://prettier.io/)会在我们进行git提交时，根据[配置](https://github.com/apache/singa-doc/blob/master/docs-site/.prettierrc)自动格式化代码。例如，它会将markdown文件中的文本最多折叠成80个字符（注释行除外）。

在介绍一个概念（如`Tensor`类）时，要提供概述（目的和与其他概念的关系）、API和例子，还可以用Google colab来演示其用法。

详细的编辑md文件和建立网站的方法请参考[本页面](https://github.com/apache/singa-doc/tree/master/docs-site)。

## API References

### CPP API

请遵循[Google CPP注释风格](https://google.github.io/styleguide/cppguide.html#Comments).

要生成文档，请从doc文件夹中运行 "doxygen"（推荐使用Doxygen >= 1.8）。

### Python API

请遵循[Google Python DocString风格](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

## Visual Studio Code (vscode)

如果你使用vscode作为编辑器，我们推荐使用以下插件。

### Docstring Snippet

[autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)生成函数、类等的docstring，要注意选择使用`google`的docString格式。

### Spell Check

[Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker)可以用来检查代码的注释，或`.md`和`.rst`文件。

要只对Python代码的注释进行拼写检查，可以在`File - Preferences - User Snippets - python.json`中添加以下代码段：

    "cspell check" : {
    "prefix": "cspell",
    "body": [
        "# Directives for doing spell check only for python and c/cpp comments",
        "# cSpell:includeRegExp #.* ",
        "# cSpell:includeRegExp (\"\"\"|''')[^\1]*\1",
        "# cSpell: CStyleComment",
    ],
    "description": "# spell check only for python comments"
    }

如果要只对Cpp代码的注释进行拼写检查，可以在`File - Preferences - User Snippets - cpp.json`中添加以下代码段：

    "cspell check" : {
    "prefix": "cspell",
    "body": [
        "// Directive for doing spell check only for cpp comments",
        "// cSpell:includeRegExp CStyleComment",
    ],
    "description": "# spell check only for cpp comments"
    }
