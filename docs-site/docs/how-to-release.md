---
id: how-to-release
title: How to Prepare a Release
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

This is a guide for the [release preparing process](http://www.apache.org/dev/release-publishing.html) in SINGA.

1. Select a release manager. The release manager (RM) is the coordinator for the release process. It is the RM's signature (.asc) that is uploaded together with the release. The RM generates KEY (RSA 4096-bit) and uploads it to a public key server. The RM needs to get his key endorsed (signed) by other Apache user, to be connected to the web of trust. He should first ask the mentor to help signing his key. [How to generate the key](http://www.apache.org/dev/release-signing.html)?

2. Check license. [FAQ](https://www.apache.org/legal/src-headers.html#faq-docs); [SINGA Issue](https://issues.apache.org/jira/projects/SINGA/issues/SINGA-447)

   - The codebase does not include third-party code which is not compatible to APL;
   - The dependencies are compatible with APL. GNU-like licenses are NOT compatible;
   - All source files written by us MUST include the Apache license header: http://www.apache.org/legal/src-headers.html. There's a script in there which helps propagating the header to all files.
   - Update the LICENSE file. If we include any third party code in the release package which is not APL, must state it at the end of the NOTICE file.

3. Bump the version. Check code and documentation

   - The build process is error-free.
   - Unit tests are included (as much as possible)
   - Conda packages run without errors.
   - The online documentation on the Apache website is up to date.

4. Prepare the RELEASE_NOTES file. Include the following items, Introduction, Features, Bugs (link to JIRA or Github PR), Changes, Dependency list, Incompatibility issues. Follow this [example](http://commons.apache.org/proper/commons-digester/commons-digester-3.0/RELEASE-NOTES.txt).

5. Prepare DISCLAIMER file. Modify from the [template](http://incubator.apache.org/guides/branding.html#disclaimers)

6. Package the release candidate. The release should be packaged into : apache-singa-VERSION.tar.gz. The release should not include any binary files including git files. Upload the release to for [stage](https://dist.apache.org/repos/dist/dev/VERSION/). The tar file, signature, KEY and SHA256 checksum file should be included. MD5 is no longer used. Policy is [here](http://www.apache.org/dev/release-distribution#sigs-and-sums)

   - apache-singa-VERSION.tar.gz
   - KEY
   - XX.acs
   - .SHA256

7. Call for vote by sending an email

   ```
   To: dev@singa.apache.org
   Subject: [VOTE] Release apache-singa-X.Y.Z (release candidate N)

   Hi all,

   I have created a build for Apache SINGA X.Y.Z, release candidate N.
   The artifacts to be voted on are located here:  xxxx
   The hashes of the artifacts are as follows: xxx
   Release artifacts are signed with the following key: xxx
   Please vote on releasing this package. The vote is open for at least
   72 hours and passes if a majority of at least three +1 votes are cast.

   [ ] +1 Release this package as Apache SINGA X.Y.Z
   [ ] 0 I don't feel strongly about it, but I'm okay with the release
   [ ] -1 Do not release this package because...

   Here is my vote:
   +1
   ```

8. Wait at least 48 hours for test responses. Any PMC, committer or contributor can test features for releasing, and feedback. Everyone should check these before vote +1. If the vote passes, then send the result email. Otherwise, repeat from the beginning.

   ```
   To: dev@singa.apache.org
   Subject: [RESULT] [VOTE] Release apache-singa-X.Y.Z (release candidate N)

   Thanks to everyone who has voted and given their comments.
   The tally is as follows.

   N binding +1s:
   <names>

   N non-binding +1s:
   <names>

   No 0s or -1s.

   I am delighted to announce that the proposal to release
   Apache SINGA X.Y.Z has passed.
   ```

9. Upload the package for [distribution](http://www.apache.org/dev/release-publishing.html#distribution) to https://dist.apache.org/repos/dist/release/VERSION/.

10. Update the Download page of SINGA website. The tar.gz file MUST be downloaded from mirror, using closer.cgi script; other artifacts MUST be downloaded from main Apache site. More details [here](http://www.apache.org/dev/release-download-pages.html). Some feedback we got during the previous releases: "Download pages must only link to formal releases, so must not include links to GitHub.", "Links to KEYS, sigs and hashes must not use dist.apache.org; instead use https://www.apache.org/dist/singa/...;", "Also you only need one KEYS link, and there should be a description of how to use KEYS + sig or hash to verify the downloads."

11. Remove the RC tag and compile the conda packages.

12. Publish the release information.

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
