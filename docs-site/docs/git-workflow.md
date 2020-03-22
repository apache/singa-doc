---
id: git-workflow
title: Git Workflow
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## For Developers

1. Fork the [SINGA Github repository](https://github.com/apache/singa) to your own Github account.

2. Clone the **repo** (short for repository) from your Github

   ```shell
   git clone https://github.com/<Github account>/singa.git
   git remote add upstream https://github.com/apache/singa.git
   ```

3. Create a new branch (e.g., `feature-foo` or `fixbug-foo`), work on it and commit your code.

   ```shell
   git checkout dev
   git checkout -b feature-foo
   # write your code
   git add <created/updated files>
   git commit
   ```

   The commit message should include:

   - A descriptive Title.
   - A detailed description. If the commit is to fix a bug, the description should ideally include a short reproduction of the problem. For new features, it may include the motivation/purpose of this new feature.

   If your branch has many small commits, you need to clean those commits via

   ```shell
   git rebase -i <commit id>
   ```

   You can [squash and reword](https://help.github.com/en/articles/about-git-rebase) the commits.

4. When you are working on the code, the `dev` of SINGA may have been updated by others; In this case, you need to pull the latest dev

   ```shell
   git checkout dev
   git pull upstream dev:dev
   ```

5. [Rebase](https://git-scm.com/book/en/v2/Git-Branching-Rebasing) `feature-foo` onto the `dev` branch and push commits to your own Github account (the new branch). The rebase operation is to make the commit history clean. The following git instructors should be executed after committing the current work:

   ```shell
   git checkout feature-foo
   git rebase dev
   git push origin feature-foo:feature-foo
   ```

   The rebase command does the [following steps](https://git-scm.com/book/en/v2/Git-Branching-Rebasing): "This operation works by going to the common ancestor of the two branches (the one you’re on and the one you’re rebasing onto), getting the diff introduced by each commit of the branch you’re on, saving those diffs to temporary files, resetting the current branch to the same commit as the branch you are rebasing onto, and finally applying each change in turn." Therefore, after executing it, you will be still on the feature branch, but your own commit IDs/hashes are changed since the diffs are committed during rebase; and your branch now has the latest code from the dev branch and your own branch.

6. Open a pull request (PR) against the dev branch of apache/singa on Github website. If you want to inform other contributors who worked on the same files, you can find the file(s) on Github and click "Blame" to see a line-by-line annotation of who changed the code last. Then, you can add @username in the PR description to ping them immediately. Please state that the contribution is your original work and that you license the work to the project under the project's open source license. Further commits (e.g., bug fix) to your new branch will be added to this pull request automatically by Github.

7. Wait for committers to review the PR. During this time, the dev of SINGA may have been updated by others, and then you need to [merge the latest dev](https://docs.fast.ai/dev/git.html#how-to-keep-your-feature-branch-up-to-date) to resolve conflicts. Some people [rebase the PR onto the latest dev](https://github.com/edx/edx-platform/wiki/How-to-Rebase-a-Pull-Request) instead of merging. However, if other developers fetch this PR to add new features and then send PR, the rebase operation would introduce **duplicate commits** (with different hash) in the future PR. See [The Golden Rule of Rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing) for the details of when to avoid using rebase. Another simple solution to update the PR (to fix conflicts or commit errors) is to checkout a new branch from the latest dev branch of Apache SINGAS repo; copy and paste the updated/added code; commit and send a new PR.

## For Committers

Committers can merge the pull requests (PRs) into the dev branch of the upstream repo. Before merging each PR, the committer should

- check the commit message (content and format)
- check the changes to existing code. API changes should be recorded
- check the Travis testing results for code/doc format and unit tests

There are two approaches to merge a pull request:

- On Github. Follow the [instructions](https://gitbox.apache.org/setup/) to connect your Apache account with your Github account. After that you can directly merge PRs on GitHub.
- To merge pull request https://github.com/apache/singa/pull/xxx via command line, the following instructions should be executed,

  ```shell
  git clone https://github.com/apache/singa.git
  git remote add asf https://gitbox.apache.org/repos/asf/singa.git
  git fetch origin pull/xxx/head:prxxx
  git checkout dev
  git merge --no-ff prxxx
  git push asf dev:dev
  ```

  Do not use rebase to merge the PR; and disable fast forward.
