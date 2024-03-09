---
id: version-4.2.0_Chinese-git-workflow
title: Git Workflow
original_id: git-workflow
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## 对于开发者

1. 将[SINGA Github repository](https://github.com/apache/singa) fork 到你自己的
   Github 账户。

2. 从你自己的 git 仓库中 clone **repo** (short for repository):

   ```shell
   git clone https://github.com/<Github account>/singa.git
   git remote add upstream https://github.com/apache/singa.git
   ```

3. 创建一个新的分支（例如 `feature-foo` 或 `fixbug-foo`），在这个分支上工作并提
   交你的代码:

   ```shell
   git checkout dev
   git checkout -b feature-foo
   # write your code
   git add <created/updated files>
   git commit
   ```

   commit 信息应包括：

   - 一个概括性的标题。
   - 详细的描述。如果提交是为了修复 bug，描述中最好包括问题的简短复现；如果是新
     功能，可以描述新功能的动机/目的。

   如果您的分支有很多小的 commit，您需要通过:

   ```shell
   git rebase -i <commit id>
   ```

   你可以[压制和重写](https://help.github.com/en/articles/about-git-rebase)提交
   的内容。

4. 当你在写代码的时候，SINGA 的`dev`分支可能已经被别人更新了；在这种情况下，你需
   要拉取最新的`dev`分支：

   ```shell
   git checkout dev
   git pull upstream dev:dev
   ```

5. 将`feature-foo`
   [rebase](https://git-scm.com/book/en/v2/Git-Branching-Rebasing)到`dev`分支上
   ，并将提交的内容推送到自己的 Github 账户（你刚刚创建的新分支），rebase 操作是
   为了清理提交历史。提交当前工作后，应执行以下 git 指令：

   ```shell
   git checkout feature-foo
   git rebase dev
   git push origin feature-foo:feature-foo
   ```

   Rebase 命令
   的[操作步骤](https://git-scm.com/book/en/v2/Git-Branching-Rebasing)如下: "这
   个操作的工作原理是进入到两个分支（你所在的分支和你要 rebase 的分支）的共同来
   源 -> 获取你所在分支的每次 commit 所引入的差异 -> 将这些差异保存到临时文件中
   -> 将当前分支重置为与你要 rebase 的分支相同的 commit -> 最后依次修改每个差异
   。"

   因此，执行后，你还是在特性分支上，但你自己的提交 ID/hash 被改变了，因为 diffs
   是在 rebase 时提交的；而且你的分支现在有来自`dev`分支和你自己分支的最新代码。

6. 在 Github 网站上创建一个针对 apache/singa `dev`分支的 pull request（PR）。如
   果你想通知其他在相同文件上工作过的贡献者，你可以在 Github 上找到文件，然后点
   击 "Blame"，就可以看到最后修改代码的人的逐行注释。然后，你可以在 PR 描述中加
   上@username，就可以立即 ping 到他们。请说明该贡献是你的原创作品，并且你在项目
   的开源许可下将该作品授权给项目。你的新分支的进一步提交（例如，bug 修复）将由
   Github 自动添加到这个 pull request 中。

7. 接下来等待 committer 审核 PR。在这段时间里，SINGA 的`dev`可能已经被其他人更新
   了，这时你需
   要[合并](https://docs.fast.ai/dev/git.html#how-to-keep-your-feature-branch-up-to-date)最
   新的`dev`来解决冲突。有些人将 PR 重
   新[rebase 到最新的 dev](https://github.com/edx/edx-platform/wiki/How-to-Rebase-a-Pull-Request)上
   ，而不是合并。但是，如果其他开发者获取这个 PR 来添加新的功能，然后再发送 PR，
   那么 rebase 操作会在未来的 PR 中引入**重复的提交**（不同的哈希）。关于何时避
   免使用 rebase 的细节，请参
   见[The Golden Rule of Rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)。
   另一种简单的更新 PR 的方法（修复冲突或提交错误）是，从 Apache SINGAS repo 的
   最新开发分支中 checkout 出一个新的分支，复制并粘贴更新/添加的代码，然后
   commit 并发送一个新的 PR。

## 对于 Committers

Committer 可以将 PR 合并到上游 repo 的 dev 分支。在合并每一个 PR 之前，提交者应
该做到：

- 检查 commit 信息(内容和格式)
- 检查对现有代码的修改，API 的变化应该被记录下来
- 检查 Travis 测试结果，检查代码/文档格式和单元测试。

合并 PR 的方式有两种:

- 在 Github 上，按照[说明](https://gitbox.apache.org/setup/)将你的 Apache 账户与
  Github 账户链接，之后你就可以直接在 GitHub 上合并 PR 了。
- 通过命令行合并 pull request
  到https://github.com/apache/singa/pull/xxx，应执行以下指令：

  ```shell
  git clone https://github.com/apache/singa.git
  git remote add asf https://gitbox.apache.org/repos/asf/singa.git
  git fetch origin pull/xxx/head:prxxx
  git checkout dev
  git merge --no-ff prxxx
  git push asf dev:dev
  ```

  不要使用 rebase 来合并 PR，并禁用 fast forward。
