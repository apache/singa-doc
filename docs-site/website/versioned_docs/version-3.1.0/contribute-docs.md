---
id: version-3.1.0-contribute-docs
title: How to Contribute to Documentation
original_id: contribute-docs
---

<!-- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License. -->

There are two types of documentation, namely markdown files and API usage
reference. This guideline introduces some tools and instruction in preparing the
source markdown files and API comments.

The markdown files will be built into HTML pages via
[Docusaurus](https://docusaurus.io/); The API comments (from the source code)
will be used to generate API reference pages using Sphinx (for Python) and
Doxygen (for CPP).

## Markdown Files

Try to follow the
[Google Documentation style](https://developers.google.com/style). For example,

1. Remove 'please' from an instruction. 'Please click...' VS 'Click ...'.
2. Follow the
   [standard captitalization rules](https://owl.purdue.edu/owl/general_writing/mechanics/help_with_capitals.html).
3. Use 'you' instead of 'we' in the instructions.
4. Use present tense and avoid 'will'
5. Prefer active voice than passive voice.

In addition, to make the documentation consistent,

1. Keep the line short, e.g., length<=80
2. Use the relative path assuming that we are in the root folder of the repo,
   e.g., `doc-site/docs` refers to `singa-doc/docs-site/docs`
3. Higlight the command, path, class function and variable using backticks,
   e.g., `Tensor`, `singa-doc/docs-site/docs`.
4. To hightlight other terms/concepts, use _graph_ or **graph**

The [prettier tool](https://prettier.io/) used by this project will auto-format
the code according to the
[configuration](https://github.com/apache/singa-doc/blob/master/docs-site/.prettierrc)
when we do `git commit`. For example, it will wrap the text in the markdown file
to at most 80 characters (except the lines for comments).

When introducing a concept (e.g., the `Tensor` class), provide the overview (the
purpose and relation to other concepts), APIs and examples. Google colab can be
used to demonstrate the usage.

Refer to [this page](https://github.com/apache/singa-doc/tree/master/docs-site)
for the details on how to edit the markdown files and build the website.

## API References

### CPP API

Follow the
[Google CPP Comments Style](https://google.github.io/styleguide/cppguide.html#Comments).

To generate docs, run "doxygen" from the doc folder (Doxygen >= 1.8 recommended)

### Python API

Follow the
[Google Python DocString Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

## Visual Studio Code (vscode)

If you use vscode as the editor, the following plugins are useful.

### Docstring Snippet

[autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
generates the docstring of functions, classes, etc. Choose the DocString Format
to `google`.

### Spell Check

[Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker)
can be configured to check the comments of the code, or .md and .rst files.

To do spell check only for comments of Python code, add the following snippet
via `File - Preferences - User Snippets - python.json`

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

To do spell check only for comments of Cpp code, add the following snippet via
`File - Preferences - User Snippets - cpp.json`

    "cspell check" : {
    "prefix": "cspell",
    "body": [
        "// Directive for doing spell check only for cpp comments",
        "// cSpell:includeRegExp CStyleComment",
    ],
    "description": "# spell check only for cpp comments"
    }
