---
id: version-3.2.0.rc1-RELEASE_NOTES_3.1.0
title: Apache SINGA-3.1.0 Release Notes
original_id: RELEASE_NOTES_3.1.0
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA is a distributed deep learning library.

This release includes following changes:

- Tensor core:

  - Support tensor transformation (reshape, transpose) for tensors up to 6
    dimensions.
  - Implement traverse_unary_transform in Cuda backend, which is similar to CPP
    backend one.

- Add new tensor operators into the autograd module, including CosSim,
  DepthToSpace, Embedding, Erf, Expand, Floor, Pad, Round, Rounde, SpaceToDepth,
  UpSample, Where. The corresponding ONNX operators are thus supported by SINGA.

- Add Embedding and Gemm into the layer module.

- Add SGD operators to opt module, including RMSProp, Adam, and AdaGrad.

- Extend the sonnx module to support DenseNet121, ShuffleNetv1, ShuffleNetv2,
  SqueezeNet, VGG19, GPT2, and RoBERTa.

- Reconstruct sonnx to

  - Support creating operators from both layer and autograd.
  - Re-write SingaRep to provide a more powerful intermediate representation of
    SINGA.
  - Add a SONNXModel which implements from Model to provide uniform API and
    features.

- Add one example that trains a BiLSTM model over the InsuranceQA data.

- Replace the Travis CI with Github workflow. Add quality and coverage
  management.

- Add compiling and packaging scripts to creat wheel packages for distribution.

- Fix bugs
  - Fix IMDB LSTM model example training script.
  - Fix Tensor operation Mult on Broadcasting use cases.
  - Gaussian function on Tensor now can run on Tensor with odd size.
  - Updated a testing helper function gradients() in autograd to lookup param
    gradient by param python object id for testing purpose.
