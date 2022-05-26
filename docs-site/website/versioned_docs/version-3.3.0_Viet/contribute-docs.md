---
id: version-3.3.0_Viet-contribute-docs
title: Tham gia chỉnh sửa Hướng Dẫn Sử Dụng
original_id: contribute-docs
---

<!-- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License. -->

Hướng Dẫn Sử Dụng có hai dạng, dạng tập tin markdown và dạng sử dụng API
reference. Tài liệu này giới thiệu vài công cụ và chỉ dẫn trong việc chuẩn bị
các tập tin nguồn markdown và chú thích API.

Tập tin markdown sẽ được sử dụng trong việc tạo trang HTML qua
[Docusaurus](https://docusaurus.io/); Chú thích API (từ nguồn mã code) sẽ được
sử dụng để tạo các trang tham khảo API sử dụng Sphinx (cho Python) và Doxygen
(cho CPP).

## Tập Tin Markdown

Làm theo [định dạng Văn bản Google](https://developers.google.com/style). Ví dụ,

1. Bỏ 'vui lòng' (please) khỏi bất cứ hướng dẫn sử dụng nào. 'Please click...'
   thành 'Click ...'.
2. Làm theo
   [qui tắc viết hoa tiêu chuẩn](https://owl.purdue.edu/owl/general_writing/mechanics/help_with_capitals.html).
3. Sử dụng 'bạn' thay cho 'chúng tôi' trong hướng dẫn.
4. Sử dụng thì hiện tại và tránh sử dụng từ 'sẽ'
5. Nên dùng dạng chủ động thay vì bị động

Thêm vào đó, để cho nội dung hướng dẫn sủ dụng thống nhất,

1. Viết câu ngắn, chẳng hạn độ dài <=80
2. Sử dụng đường dẫn liên quan, mặc định chúng ta đang ở thư mục root của repo,
   vd., `doc-site/docs` để chỉ `singa-doc/docs-site/docs`
3. Nhấn mạnh câu lệnh, đường dẫn, class, function và biến sử dụng backticks,
   vd., `Tensor`, `singa-doc/docs-site/docs`.
4. Để nêu bật các điều khoản/khái niệm, sử dụng _graph_ hoặc **graph**

[Cộng cụ prettier](https://prettier.io/) được sử dụng bởi dự án này sẽ tự làm
định dạng code dựa trên
[cấu hình](https://github.com/apache/singa-doc/blob/master/docs-site/.prettierrc)
khi thực hiện `git commit`. Ví dụ, nó sẽ gói chữ trong các tập tin markdown
thành nhiều nhất 80 kí tự (trừ các dòng chú thích).

Khi giới thiệu một khái niệm (concept) (vd., class `Tensor`), đưa ra khái quát
chung (mục đích và mối quan hệ với các khái niệm khác), APIs và ví dụ. Google
colab có thể được sử dụng để mô phỏng điều này.

Tham khảo [trang](https://github.com/apache/singa-doc/tree/master/docs-site) để
biết thêm chi tiết về cách chỉnh sửa các tập tin markdown và xây dựng website.

## Tham Khảo API

### CPP API

Thực hiện theo
[Mẫu chú thích của Google CPP](https://google.github.io/styleguide/cppguide.html#Comments).

Để tạo văn bản, chạy "doxygen" từ thư mục doc (khuyến khích Doxygen >= 1.8)

### Python API

Thực hiện theo
[Mẫu Google Python DocString](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

## Visual Studio Code (vscode)

Nếu bạn sử dụng vscode để viết code, các plugins sau sẽ giúp ích.

### Docstring Snippet

[autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
tạo docstring của functions, classes, v.v. Lựa chọn định dạng DocString to
`google`.

### Kiểm Tra Lỗi Chính Tả

[Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker)
có thể được cơ cấu để kiểm tra chú thích trong mã code, hoặc các tập tin .md và
.rst.

Để kiểm tra lỗi chính tả cho các dòng chú thích trong code Python, thêm vào các
snippet sau qua `File - Preferences - User Snippets - python.json`

    "cspell check" : {
    "prefix": "cspell",
    "body": [
        "# Chỉ dẫn kiểm tra lỗi chính tả cho các dòng chú thích trong code python và c/cpp",
        "# cSpell:includeRegExp #.* ",
        "# cSpell:includeRegExp (\"\"\"|''')[^\1]*\1",
        "# cSpell: CStyleComment",
    ],
    "description": "# chỉ kiểm tra lỗi chính tả cho chú thích trong python"
    }

Để kiểm tra lỗi chính tả cho các dòng chú thích trong code Cpp, thêm vào các
snippet sau qua `File - Preferences - User Snippets - cpp.json`

    "cspell check" : {
    "prefix": "cspell",
    "body": [
        "// Chỉ dẫn kiểm tra lỗi chính tả cho các dòng chú thích trong code cpp",
        "// cSpell:includeRegExp CStyleComment",
    ],
    "description": "# chỉ kiểm tra lỗi chính tả cho chú thích trong cpp"
    }
