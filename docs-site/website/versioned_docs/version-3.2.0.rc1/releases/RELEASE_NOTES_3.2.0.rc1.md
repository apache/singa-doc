---
id: version-3.2.0.rc1-RELEASE_NOTES_3.2.0.rc1
title: Apache SINGA-3.2.0.rc1 Release Notes
original_id: RELEASE_NOTES_3.2.0.rc1
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA is a distributed deep learning library.

This release includes following changes:

- New examples

  - Add one cifar-10 distributed CNN example for benchmarking the performance of the distributed
    training.
  - Add one large CNN example for training with a dataset from the filesysetm.

- Enhance distributed training 

  - Improve the data augmentation module for faster distributed training.
  - Add device synchronization for more accurate time measurements during the distributed training.

- Add Support for half-precision floating-point format (fp16) in deep learning models and 
  computational kernels.

- Update new onnx APIs and fix onnx examples accordingly, namely, DenseNet121, ShuffleNetv1, 
  ShuffleNetv2, SqueezeNet, VGG19.

- Add a new method to resize images by given width and height.

- Use docusaurus versioning to simplify the process of generating the project homepage.

- Promote code quality

  - Unify the formats of docstrings that describe the contents and usage of the module.
  - Unify the parameters of command-line arguments.
  
- Fix bugs

  - Fix the CI build error by downloading the tbb binaries.
  - Add disabling graph option for accessing parameter or gradient tensors during distributed  
    training.
  - Solve the warnings of deprecated functions in the distributed optimizer module.
