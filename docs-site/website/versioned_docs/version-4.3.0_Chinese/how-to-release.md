---
id: version-4.3.0_Chinese-how-to-release
title: How to Prepare a Release
original_id: how-to-release
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

这是 SINGA 的[发布准备流程](http://www.apache.org/dev/release-publishing.html)指
南。

1. 选择一个发布管理者。发布管理者（RM）是发布过程的协调者，他的签名文件（.asc）
   将会与发布一起上传。RM 生成 KEY (RSA 4096 位)并将其上传到公钥服务器，首先需要
   得到其他 Apache 用户对他的密钥的认可（签名），才能连接到信任网，RM 需要先求助
   其他项目管理者帮忙认证他的密钥
   。[如何生成密钥？](http://www.apache.org/dev/release-signing.html)

2. 检查 license。 [FAQ](https://www.apache.org/legal/src-headers.html#faq-docs);
   [SINGA Issue](https://issues.apache.org/jira/projects/SINGA/issues/SINGA-447)

   - 代码库不能包含与 APL 不兼容的第三方代码。
   - 依赖项与 APL 兼容，GNU 类 license 不兼容。
   - 我们编写的所有源文件都必须包含 Apache license 头
     ：http://www.apache.org/legal/src-headers.html. 链接中有一个脚本可以帮助将
     这个头同步到所有文件。
   - 更新 LICENSE 文件。如果我们在发行包中包含了任何非 APL 的第三方代码，必须要
     在 NOTICE 文件的最后注明。

3. 复查版本。检查代码和文档。

   - 编译过程无错误。
   - (尽可能地)包含进单元测试。
   - Conda 包运行无误。
   - Apache 网站上的在线文档是最新的。

4. 准备好 RELEASE_NOTES 文件。包括以下项目，介绍，特性，错误（链接到 JIRA 或
   Github PR），变更，依赖列表，不兼容问题，可以按照这
   个[例子](<(http://commons.apache.org/proper/commons-digester/commons-digester-3.0/RELEASE-NOTES.txt)>)来
   写。

5. 打包候选版本。该版本应该打包成：apache-singa-VERSION.tar.gz。这个版本不应该包
   含任何二进制文件，包括 git 文件。但是 CMake 的编译依赖于 git 标签来获取版本号
   ；要删除这个依赖，你需要手动更新 CMakeLists.txt 文件来设置版本号：

   ```
   # remove the following lines
   include(GetGitRevisionDescription)
   git_describe(VERSION --tags --dirty=-d)
   string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" VERSION_MAJOR "${VERSION}")
   string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR "${VERSION}")
   string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH "${VERSION}")

   # set the numbers manually
   SET(PACKAGE_VERSION 3.0.0)
   SET(VERSION 3.0.0)
   SET(SINGA_MAJOR_VERSION 3)  # 0 -
   SET(SINGA_MINOR_VERSION 0)  # 0 - 9
   SET(SINGA_PATCH_VERSION 0)  # 0 - 99
   ```

   将软件包上传到[stage repo](https://dist.apache.org/repos/dist/dev/singa/)。应
   包括 tar 文件、签名、KEY 和 SHA256 校验和文件。MD5 不再使用，详细规则
   在[这里](https://dist.apache.org/repos/dist/dev/singa/)。阶段文件夹应该包括：

   - apache-singa-VERSION.tar.gz
   - apache-singa-VERSION.acs
   - apache-singa-VERSION.SHA256

   创建这些文件并上传到 stage svn repo 的命令如下：

   ```sh
   # in singa repo
   rm -rf .git
   rm -rf rafiki/*
   cd ..
   tar -czvf apache-singa-VERSION.tar.gz  singa/

   mkdir stage
   cd stage
   svn co https://dist.apache.org/repos/dist/dev/singa/
   cd singa
   # copy the KEYS file from singa repo to this folder if it is not here
   cp ../../singa/KEYS .
   mkdir VERSION
   # copy the tar.gz file
   mv ../../apache-singa-VERSION.tar.gz VERSION/
   cd VERSION
   sha512sum apache-singa-VERSION.tar.gz > apache-singa-VERSION.tar.gz.sha512
   gpg --armor --output apache-singa-VERSION.tar.gz.asc --detach-sig apache-singa-VERSION.tar.gz
   cd ..
   svn add VERSION
   svn commit
   ```

6) 通过发送电子邮件的方式进行投票。现举例如下：

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



   Please vote on releasing this package. The vote is open for at least 72 hours and passes if a majority of at least three +1 votes are cast.

   [ ] +1 Release this package as Apache SINGA X.Y.Z

   [ ] 0 I don't feel strongly about it, but I'm okay with the release

   [ ] -1 Do not release this package because...

   Here is my vote: +1

```

7. 等待至少 48 小时的测试回复。任何 PMC、提交者或贡献者都可以测试发布的功能，以
   及反馈。大家在投票+1 之前应该检查这些。如果投票通过，则发送如下的结果邮件，否
   则，从头开始重复刚刚的步骤。

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

```

8. 将软件包上传至
   https://dist.apache.org/repos/dist/release/singa/，以便[distribution](http://www.apache.org/dev/release-publishing.html#distribution)。

9. 更新 SINGA 网站的下载页面。tar.gz 文件必须从镜像下载，使用 closer.cgi 脚本；
   其他工件必须从 Apache 主站点下载。更多细节请
   看[这里](http://www.apache.org/dev/release-download-pages.html)。我们在之前的
   版本中得到的一些反馈。“下载页面必须只链接到正式发布的版本，所以不能包含到
   GitHub 的链接”，“链接到 KEYS, sig 和 Hash 的链接不能使用 dist.apache.org 而应
   该使用
   https://www.apache.org/dist/singa/...”“而且你只需要一个KEYS链接，而且应该描述如何使用KEYS+sig或Hash来验证下载。”

10. 删除 RC 标签并编译 conda 包。

11. 发布 release 信息：

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
