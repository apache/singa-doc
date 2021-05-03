---
id: git-workflow
title: Quy Trình Sử Dụng Git
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## Dành cho Lập Trình Viên

1. Fork [SINGA Github repository](https://github.com/apache/singa) về tài khoản Github của bạn.

2. Clone **repo** (viết tắt của repository) từ tài khoản Github của bạn

   ```shell
   git clone https://github.com/<Github account>/singa.git
   git remote add upstream https://github.com/apache/singa.git
   ```

3. Tạo branch mới (vd., `feature-foo` hoặc `fixbug-foo`), chỉnh sửa và 
   commit code của bạn ở đây .

   ```shell
   git checkout dev
   git checkout -b feature-foo
   # write your code
   git add <created/updated files>
   git commit
   ```

   Nội dung lệnh commit nên bao gồm:

   - Tiêu đề (Title) mô tả.
   - Mô tả chi tiết. Nếu lệnh commit là sửa lỗi (bugs), tốt nhất là nên 
     bao gồm việc mô tả ngắn gọn lại vấn đề. Nếu thêm tính năng mới, có thể bao gồm động cơ thúc đẩy/mục đích của 
     tính năng mới này.

   Nếu branch của bạn có nhiều commits nhỏ, bạn cần súc tích lại các commits bằng cách 

   ```shell
   git rebase -i <commit id>
   ```

   Bạn có thể
   [squash và reword](https://help.github.com/en/articles/about-git-rebase) các 
   commits.

4. Khi bạn đang xử lý các mã code, branch `dev` của SINGA có thể đang được cập nhật bởi người khác; 
   Trong trường hợp này, bạn cần pull dev mới nhất

   ```shell
   git checkout dev
   git pull upstream dev:dev
   ```

5. [Rebase](https://git-scm.com/book/en/v2/Git-Branching-Rebasing) `feature-foo`
   vào branch `dev` và push commits vào tài khoản Github của bạn (
   branch mới). Lệnh rebase nhằm giúp cho lịch sử commit của bạn rõ ràng. Các lệnh git dưới đây nên 
   được thực hiện sau khi bạn commit các việc làm của mình:

   ```shell
   git checkout feature-foo
   git rebase dev
   git push origin feature-foo:feature-foo
   ```

   Lệnh rebase thực hiện các
   [bước sau](https://git-scm.com/book/en/v2/Git-Branching-Rebasing):
   "Lệnh này thực hiện bắt đầu từ hình thái ban đầu của hai branches
   (mà bạn đang sử dụng hoặc bạn đang rebase vào), xác định sự khác nhau
   ở mỗi commit của branch bạn đang sử dụng, lưu các điểm khác nhau vào 
   tập tin tạm thời, điều chỉnh branch hiện tại để có cùng commit với
   branch mà bạn đang rebase vào, rồi cuối cùng áp dụng từng thay đổi một theo thứ tự."
   Bởi vậy, sau khi thực hiện, bạn sẽ vẫn ở feature branch, nhưng commit IDs/hashes 
   của bạn được thay đổi do các điểm khác nhau đã được commit trong quá trình rebase; 
   và branch của bạn giờ đây chứa bản code cập nhật nhất từ branch dev và branch của bạn.

6. Tạo một pull request (PR) vào branch dev của apache/singa trên website Github.
   Nếu bạn muốn thông báo cho các thành viên khác đang làm việc trên cùng một tập tin, 
   bạn có thể tìm tập tin đó trên Github và nhấn vào "Blame" để xem chú thích từng dòng một
   ai đã thay đổi code lần cuối cùng. Sau đó, bạn có thể thêm @username trong mục mô tả PR
   để nhắc họ. Hãy nói rõ rằng những đóng góp này là công sức của bạn và rằng bạn cấp bản quyền 
   công sức này cho dự án theo dạng bản quyền dự án mở. Những commits khác (vd., sửa lỗi) 
   vào branch mới này sẽ được tự động cập nhật vào pull request của bạn bởi Github.

7. Đợi thành viên xét duyệt PR. Trong quá trình này, dev branch của SINGA có thể được những người khác cập nhật,
   do vậy bạn cần phải [merge the latest dev](https://docs.fast.ai/dev/git.html#how-to-keep-your-feature-branch-up-to-date)
   để xử lý các conflicts. Một số người 
   [rebase PR vào branch dev mới nhất](https://github.com/edx/edx-platform/wiki/How-to-Rebase-a-Pull-Request)
   thay vì merging. Tuy nhiên, nếu các thành viên khác fetch PR này để thêm các tính năng mới rồi gửi PR, việc rebase sẽ gây ra **duplicate commits** (với hash khác) ở PR mới. Xem
   [Nguyên tắc vàng để Rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)
   để biết thêm chi tiết khi nào cần tránh rebase. Một giải pháp đơn giản để cập nhật PR (nhằm xử lý conflicts hay lỗi commit) là checkout một branch mới từ branch dev cập nhật nhất của Apache SINGAS repo; copy và paste các mã code được cập nhật/thêm vào; commit và gửi một PR mới.

## Dành cho commiters

Commiters có thể merge pull requests (PRs) vào dev branch của repo upstream. 
Trước khi merge mỗi PR, committer nên

- kiểm tra thông điệp commit (nội dung và định dạng)
- kiểm tra những thay đổi so với code hiện tại. Thay đổi về API nên được ghi lại.
- kiểm tra kết quả Travis testing cho định dạng mã code/tài liệu và unit tests

Có hai cách để merge một pull request:

- Trên Github. Làm theo [hướng dẫn](https://gitbox.apache.org/setup/) để
  kết nối tài khoản Apache với tài khoản Github của bạn. Sau đó bạn có thể trực tiếp 
  merge PRs trên GitHub.
- Để merge pull request https://github.com/apache/singa/pull/xxx qua command
  line, thực hiện theo hướng dẫn sau: 

  ```shell
  git clone https://github.com/apache/singa.git
  git remote add asf https://gitbox.apache.org/repos/asf/singa.git
  git fetch origin pull/xxx/head:prxxx
  git checkout dev
  git merge --no-ff prxxx
  git push asf dev:dev
  ```

  Không sử dụng rebase để merge PR; và vô hiệu hoá fast forward.
