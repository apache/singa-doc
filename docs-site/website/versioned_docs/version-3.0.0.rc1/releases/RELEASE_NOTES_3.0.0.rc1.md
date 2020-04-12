---
id: version-3.0.0.rc1-RELEASE_NOTES_3.0.0.rc1
title: singa-3.0.0.rc1 Release Notes
original_id: RELEASE_NOTES_3.0.0.rc1
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Release Notes - SINGA - Version singa-3.0.0.rc1

SINGA is a distributed deep learning library.

This release includes following changes:

- Code quality has been promoted by introducing linting check in CI and auto
  code formatter. For linting, the tools, `cpplint` and `pylint`, are used and
  configured to comply
  [google coding styles](http://google.github.io/styleguide/) details in
  `tool/linting/`. Similarly, formatting tools, `clang-format` and `yapf`
  configured with google coding styles, are the recommended one for developers
  to clean code before submitting changes, details in `tool/code-format/`.
  [LGTM](https://lgtm.com) is enabled on Github for code quality check; License
  check is also enabled.

- New Tensor APIs are added for naming consistency, and feature enhancement:

  - size(), mem_size(), get_value(), to_proto(), l1(), l2(): added for the sake
    of naming consistency
  - AsType(): convert data type between `float` and `int`
  - ceil(): perform element-wise ceiling of the input
  - concat(): concatenate two tensor
  - index selector: e.g. tensor1[:,:,1:,1:]
  - softmax(in, axis): allow to perform softmax on a axis on a multi-dimensional
    tensor

- 14 new operators are added into the autograd module: Gemm, GlobalAveragePool,
  ConstantOfShape, Dropout, ReduceSum, ReduceMean, Slice, Ceil, Split, Gather,
  Tile, NonZero, Cast, OneHot. Their unit tests are added as well.

- 14 new operators are added to sonnx module for both backend and frontend:
  [Gemm](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm),
  [GlobalAveragePool](https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalAveragePool),
  [ConstantOfShape](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConstantOfShape),
  [Dropout](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout),
  [ReduceSum](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum),
  [ReduceMean](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean),
  [Slice](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice),
  [Ceil](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil),
  [Split](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Split),
  [Gather](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather),
  [Tile](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tile),
  [NonZero](https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonZero),
  [Cast](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast),
  [OneHot](https://github.com/onnx/onnx/blob/master/docs/Operators.md#OneHot).
  Their tests are added as well.

- Some ONNX models are imported into SINGA, including
  [Bert-squad](https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad),
  [Arcface](https://github.com/onnx/models/tree/master/vision/body_analysis/arcface),
  [FER+ Emotion](https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus),
  [MobileNet](https://github.com/onnx/models/tree/master/vision/classification/mobilenet),
  [ResNet18](https://github.com/onnx/models/tree/master/vision/classification/resnet),
  [Tiny Yolov2](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny_yolov2),
  [Vgg16](https://github.com/onnx/models/tree/master/vision/classification/vgg),
  and Mnist.

- Some operators now support
  [multidirectional broadcasting](https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md#multidirectional-broadcasting),
  including Add, Sub, Mul, Div, Pow, PRelu, Gemm

- [Distributed training with communication optimization].
  [DistOpt](./python/singa/opt.py) has implemented multiple optimization
  techniques, including gradient sparsification, chunk transmission, and
  gradient compression.

- Computational graph construction at the CPP level. The operations submitted to
  the Device are buffered. After analyzing the dependency, the computational
  graph is created, which is further analyzed for speed and memory optimization.
  To enable this feature, use the [Module API](./python/singa/module.py).

- New website based on Docusaurus. The documentation files are moved to a
  separate repo [singa-doc](https://github.com/apache/singa-doc). The static
  website files are stored at
  [singa-site](https://github.com/apache/singa-site).

- DNNL([Deep Neural Network Library](https://github.com/intel/mkl-dnn)), powered
  by Intel, is integrated into
  `model/operations/[batchnorm|pooling|convolution]`, the changes is opaque to
  the end users. The current version is dnnl v1.1 which replaced previous
  integration of mkl-dnn v0.18. The framework could boost the performance of dl
  operations when executing on CPU. The dnnl dependency is installed through
  conda.

- Some Tensor APIs are marked as deprecated which could be replaced by
  broadcast, and it can support better on multi-dimensional operations. These
  APIs are add_column(), add_row(), div_column(), div_row(), mult_column(),
  mult_row()

- Conv and Pooling are enhanced to support fine-grained padding like (2,3,2,3),
  and
  [SAME_UPPER, SAME_LOWER](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv)
  pad mode and shape checking.

- Reconstruct soonx,
  - Support two types of weight value (Initializer and Constant Node);
  - For some operators (BatchNorm, Reshape, Clip, Slice, Gather, Tile, OneHot),
    move some inputs to its attributes;
  - Define and implement the type conversion map.
