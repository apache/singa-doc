---
id: version-3.3.0_Viet-how-to-release
title: Chuẩn bị trước khi phát hành
original_id: how-to-release
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Đây là hướng dẫn chuẩn bị cho việc phát hành
[quá trình chuẩn bị trước khi phát hành](http://www.apache.org/dev/release-publishing.html)
SINGA.

1. Lựa chọn người quản lý cho việc phát hành. Người quản lý chịu trách nhiệm
   điều phối quá trình phát hành. Chữ ký của người quản lý (.asc) sẽ được tải
   lên cùng với bản phát hành. Nggười quản lý tạo KEY (RSA 4096-bit) và tải nó
   lên public key server. Để được tin cậy kết nối trên web, người quản lý cần
   các người dùng Apache khác chứng thực (signed) Key của mình. Anh ta trước
   tiên cần yêu cầu mentor giúp chứng thực key.
   [Cách tạo Key](http://www.apache.org/dev/release-signing.html)?

2. Kiểm tra bản quyền.
   [FAQ](https://www.apache.org/legal/src-headers.html#faq-docs);
   [Các bản SINGA đã phát hành](https://issues.apache.org/jira/projects/SINGA/issues/SINGA-447)

   - Nền tảng code không bao gồm code của bên thứ 3 mà không tương thích với
     APL;
   - Các chương trình dependencies phải tương thích với APL. Các licenses giống
     với GNU là không tương thích;
   - Các tệp tin nguồn viết bởi chúng tôi PHẢI bao gồm license header của
     Apache: http://www.apache.org/legal/src-headers.html. Chúng tôi cung cấp
     script để chạy header trên tất cả các tệp tin.
   - Cập nhật tệp tin LICENSE. Nếu code có chứa mã code của một bên thứ 3 trong
     bản phát hành mà không phải APL, phải nêu rõ ở phần cuối của tập tin THÔNG
     BÁO.

3. Nâng cấp phiên bản. Kiểm tra mã code và Tài liệu hướng dẫn

   - Quá trình cài đặt không bị lỗi nào.
   - Bao gồm tests cho những mục nhỏ (nhiều nhất có thể)
   - Gói chương trình Conda chạy không bị lỗi.
   - Tài liệu hướng dẫn trực tuyến trên trang web Apache là mới nhất.

4. Chuẩn bị tệp tin RELEASE_NOTES (Lưu ý phát hành). Bao gồm các mục, Giới
   thiệu, Tính năng nổi bật, Lỗi Bugs, (đường dẫn tới JIRA hoặc Github PR),
   Những thay đổi, Danh sách thư viện Dependency, Các vấn đề không tương thích.
   Làm theo
   [ví dụ](http://commons.apache.org/proper/commons-digester/commons-digester-3.0/RELEASE-NOTES.txt).

5. Gói các phiên bản phát hành. Bản phát hành cần được gói gọn thành:
   apache-singa-VERSION.tar.gz. Trong bản phát hành không nên chứa bất kì tệp
   tin dạng binary nào, bao gồm cả các tệp tin git. Tuy nhiên, các tệp CMake
   compilation dựa vào git tag để tạo số phiên bản; để bỏ qua dependency này,
   bạn cần cập nhật tệp tin CMakeLists.txt theo cách thủ công để tạo số phiên
   bản.

   ```
   # xoá các dòng sau
   include(GetGitRevisionDescription)
   git_describe(VERSION --tags --dirty=-d)
   string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" VERSION_MAJOR "${VERSION}")
   string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR "${VERSION}")
   string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH "${VERSION}")

   # thay đổi số
   SET(PACKAGE_VERSION 3.0.0)
   SET(VERSION 3.0.0)
   SET(SINGA_MAJOR_VERSION 3)  # 0 -
   SET(SINGA_MINOR_VERSION 0)  # 0 - 9
   SET(SINGA_PATCH_VERSION 0)  # 0 - 99
   ```

   Tải gói chương trình lên
   [stage repo](https://dist.apache.org/repos/dist/dev/singa/). Cần bao gồm các
   tệp tin tar, signature, KEY và tệp tin SHA256 checksum. Không sử dụng MD5.
   Xem chính sách tại
   [đây](http://www.apache.org/dev/release-distribution#sigs-and-sums). Thư mục
   stage cần bao gồm:

   - apache-singa-VERSION.tar.gz
   - apache-singa-VERSION.acs
   - apache-singa-VERSION.SHA256

   Các lệnh để tạo tệp tin và tải chúng lên stage svn repo:

   ```sh
   # trong singa repo
   rm -rf .git
   rm -rf rafiki/*
   cd ..
   tar -czvf apache-singa-VERSION.tar.gz  singa/

   mkdir stage
   cd stage
   svn co https://dist.apache.org/repos/dist/dev/singa/
   cd singa
   # copy tệp tin KEYS từ singa repo sang thư mục này nếu không có
   cp ../../singa/KEYS .
   mkdir VERSION
   # copy tệp tin tar.gz
   mv ../../apache-singa-VERSION.tar.gz VERSION/
   cd VERSION
   sha512sum apache-singa-VERSION.tar.gz > apache-singa-VERSION.tar.gz.sha512
   gpg --armor --output apache-singa-VERSION.tar.gz.asc --detach-sig apache-singa-VERSION.tar.gz
   cd ..
   svn add VERSION
   svn commit
   ```

6) Kêu gọi vote bằng cách gửi email. Xem ví dụ dưới đây.

   ```
   To: dev@singa.apache.org
   Subject: [VOTE] Release apache-singa-X.Y.Z (release candidate N)

   Hi all,

   I have created a build for Apache SINGA 3.1.0, release candidate 2.

   The release note is at
   https://github.com/apache/singa/blob/master/RELEASE_NOTES.

   The artifacts to be voted on are located here:
   https://dist.apache.org/repos/dist/dev/singa/3.1.0.rc2/apache-singa-3.1.0.rc2.tar.gz
    
   The hashes of the artifacts are as follows:
   SHA512: 84545499ad36da108c6a599edd1d853f82d331bc03273b5278515554866f0c698e881f956b2eabcb6b29c07fa9fa4ff1add5a777b58db8a6a2362cf383b5c04d 

   Release artifacts are signed with the followingkey:
   https://dist.apache.org/repos/dist/dev/singa/KEYS

   The signature file is:
   https://dist.apache.org/repos/dist/dev/singa/3.1.0.rc2/apache-singa-3.1.0.rc2.tar.gz.asc

   The Github tag is at:
   https://github.com/apache/singa/releases/tag/3.1.0.rc2

   The documentation website is at
   http://singa.apache.org/docs/next/installation/

   Some examples are available for testing:
   https://github.com/apache/singa/tree/master/examples
   ```

Please vote on releasing this package. The vote is open for at least 72 hours
and passes if a majority of at least three +1 votes are cast.

[ ] +1 Release this package as Apache SINGA X.Y.Z [ ] 0 I don't feel strongly
about it, but I'm okay with the release [ ] -1 Do not release this package
because...

Here is my vote: +1

```

7) Sau đó đợi ít nhất 48 giờ để nhận phản hồi. Bất kì PMC, committer hay contributor
đều có thể kiểm tra các tính năng trước khi phát hành, và đưa ra nhận xét. Mọi người nên kiểm tra trước khi
đưa ra vote +1. Nếu vote được thông qua, vui lòng gửi email kết quả. Nếu không thì cần lặp lại trình tự từ đầu.

```

To: dev@singa.apache.org Subject: [RESULT][vote] Release apache-singa-X.Y.Z
(release candidate N)

Thanks to everyone who has voted and given their comments. The tally is as
follows.

N binding +1s: <names>

N non-binding +1s: <names>

No 0s or -1s.

I am delighted to announce that the proposal to release Apache SINGA X.Y.Z has
passed.

````

8) Tải gói chương trình để
[phân bổ](http://www.apache.org/dev/release-publishing.html#distribution)
tới https://dist.apache.org/repos/dist/release/singa/.

9) Cập nhật trang Tải (Download) trên website SINGA. Tệp tin tar.gz PHẢI được tải từ mirror, sử dụng closer.cgi script; các tạo tác khác PHẢI được tải từ trang chủ Apache. Xem chi tiết tại
[đây](http://www.apache.org/dev/release-download-pages.html). Một vài nhận xét chúng tôi nhận được trong các đợt phát hành trước: "Trang Tải chỉ nên được dẫn tới các bản phát hành chính thức, vì vậy phải bao gồm đường dẫn tới GitHub.", "Đường dẫn tới KEYS, sigs và
hashes không nên sử dụng dist.apache.org; mà nên dùng
https://www.apache.org/dist/singa/...;", "Và bạn chỉ cần một đường dẫn tới KEYS,
và cần có hướng dẫn cách sử dụng KEYS + sig hay hash để chứng thực hoàn tất việc tải."

10) Xoá tag RC và tập hợp gói conda packages.

11) Xuất bản thông tin phát hành.

 ```
 To: announce@apache.org, dev@singa.apache.org
 Subject: [ANNOUNCE] Apache SINGA X.Y.Z released

 We are pleased to announce that SINGA X.Y.Z is released.

 SINGA is a general distributed deep learning platform
 for training big deep learning models over large datasets.
 The release is available at: http://singa.apache.org/downloads.html
 The main features of this release include XXX
 We look forward to hearing your feedback, suggestions,
 and contributions to the project.

 On behalf of the SINGA team, {SINGA Team Member Name}
 ```
````
