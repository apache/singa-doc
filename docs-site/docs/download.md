---
id: downloads
title: Download SINGA
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

## Verify

To verify the downloaded tar.gz file, download the
[KEYS](https://www.apache.org/dist/singa/KEYS) and ASC files and then execute
the following commands

```shell
% gpg --import KEYS
% gpg --verify downloaded_file.asc downloaded_file
```

You can also check the SHA512 or MD5 values to see if the download is completed.
## V5.0.0 (14 April 2025):

- [Apache SINGA 5.0.0](http://www.apache.org/dyn/closer.cgi/singa/5.0.0/apache-singa-5.0.0.tar.gz)
  [\[SHA512\]](https://www.apache.org/dist/singa/5.0.0/apache-singa-5.0.0.tar.gz.sha512)
  [\[ASC\]](https://www.apache.org/dist/singa/5.0.0/apache-singa-5.0.0.tar.gz.asc)
- [Release Notes 5.0.0](http://singa.apache.org/docs/releases/RELEASE_NOTES_5.0.0)
- Major changes:
  * Add the implementations of the healthcare model zoo.
    * Add the implementation for the cardiovascular disease.
    * Add the implementation for the diabetic disease.
    * Add the implementation for the hematologic disease.
    * Add the implementation for the kidney disease.
    * Add the implementation for the malaria disease.
    * Add the implementation for the thyroid eye disease.

  * Optimize the distributed training by updating the MSOptimizer and MSSGD.

  * Improve the efficiency of the transformer example.

  * Add the sparsification version of the model for the model selection example.

  * Update data processing for the benchmark dataset.

  * Update the pom.xml file to include paths for datasets.

  * Update the online documentations for the healthcare model zoo.

## V4.3.0 (7 September 2024):

- [Apache SINGA 4.3.0](https://archive.apache.org/dist/singa/4.3.0/apache-singa-4.3.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/singa/4.3.0/apache-singa-4.3.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/singa/4.3.0/apache-singa-4.3.0.tar.gz.asc)
- [Release Notes 4.3.0](http://singa.apache.org/docs/releases/RELEASE_NOTES_4.3.0)
- Major changes:
  * Add the implementation for the Transformer example.

  * Enhance examples
    - Update the readme file for the dynamic model slicing example.
    - Update the HFL example by setting the maximum number of epochs.
    - Add the multiprocess training implementation for the cnn ms example.
    - Add the sparsification version of the model for the cnn ms example.
  
  * Extend the matrix multiplication operator to more dimensions.

  * Update the data types and tensor operations for model training.

  * Add the implementation for the new sum error loss.

  * Update the website
    - Add the news for the SIGMOD Systems Award.

  * Fix bugs
    - Fix the Github Actions for online code testing.


## V4.2.0 (15 March 2024):

- [Apache SINGA 4.2.0](https://archive.apache.org/dist/singa/4.2.0/apache-singa-4.2.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/singa/4.2.0/apache-singa-4.2.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/singa/4.2.0/apache-singa-4.2.0.tar.gz.asc)
- [Release Notes 4.2.0](http://singa.apache.org/docs/releases/RELEASE_NOTES_4.2.0)
- Major changes:

  * Add support for deep learning models running on top of PolarDB
    - Implement efficient model selection for a given dataset stored in the database
    - Add support for dynamic model creation
    - Add support for flexible setting of model training configurations
    - Optimize the in-database analytics modules for scalability, efficiency and memory consumption

  * New example
    - Add a horizontal federated learning example using the Bank dataset

  * Enhance examples
    - Add sample training data for testing the model selection application

  * Update the website
    - Update the star button in the main page
    - Refine the display of star statistics

  * Update the python versions for wheel files

  * Fix bugs
    - Fix the rat check files
    - Update the license files

## V4.1.0 (05 November 2023):

- [Apache SINGA 4.1.0](https://archive.apache.org/dist/singa/4.1.0/apache-singa-4.1.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/singa/4.1.0/apache-singa-4.1.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/singa/4.1.0/apache-singa-4.1.0.tar.gz.asc)
- [Release Notes 4.1.0](http://singa.apache.org/docs/releases/RELEASE_NOTES_4.1.0)
- Major changes:

  * New examples
    - Add an example for malaria detection using cell images.
    - Add an example for structured data learning.

  * Add support for models running on top of RDBMS
    - Add support for in-database model definition and selection in RDBMS.
    - Implement training-free model evaluation metrics for in-database model
      selection.
    - Implement a coordinator to balance between training-free and
      training-based model evaluations for in-database model selection.

  * Enhance distributed training
    - Add implementations for the sum error loss.
    - Improve the optimizer to return model gradients.
    - Improve the iterative checking for tensors and strings in the ModelMeta
      class.

  * Enhance example code
    - Add support for flexible setting of training configurations for models,
      e.g., learning rates, weight decay, momentum, etc.
    - Add implementations for dynamic models with varying layer sizes.

  * Update the website
    - Add illustrations for database integration.
    - Update users of Apache SINGA.

  * Fix bugs
    - Update the NVIDIA_GPGKEY in the Dockerfile for building wheel files.
    - Update the versions of dependencies in the wheel file.
    - Fix the collections module in the model.py file.

## V4.0.0 (07 April 2023):

- [Apache SINGA 4.0.0](https://archive.apache.org/dist/singa/4.0.0/apache-singa-4.0.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/singa/4.0.0/apache-singa-4.0.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/singa/4.0.0/apache-singa-4.0.0.tar.gz.asc)
- [Release Notes 4.0.0](http://singa.apache.org/docs/releases/RELEASE_NOTES_4.0.0)
- Major changes:

  * Enhance distributed training
    - Add support for configuration of number of GPUs to be used.
    - Increase max epoch for better convergence.
    - Print intermediate mini-batch information.
    - Add support for switching between CPU and GPU devices.

  * Enhance example code
    - Update the args of normalize forward function in the transforms of the
      BloodMnist example.
    - Update the xceptionnet in the cnn example.
    - Add arguments for weight decay, momentum and learning rates in the cnn
      example.
    - Add training scripts for more datasets and model types in the cnn example.
    - Add resnet dist version for the large dataset cnn example.
    - Add cifar 10 multi process for the large dataset cnn example.
    - Add sparsification implementation for mnist in the large dataset cnn
      example.
    - Update the cifar datasets downloading to local directories.
    - Extend the cifar datasets load function for customized directorires.

  * Enhance the webpage
    - Update online documentation for distributed training.

  * Promote code quality
    - Update inline comments for prepreocessing and data loading

  * Update the PIL image module

  * Update the runtime Dockerfile

  * Update the conda files

## V3.3.0 (07 June 2022):

- [Apache SINGA 3.3.0](https://archive.apache.org/dist/singa/3.3.0/apache-singa-3.3.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/singa/3.3.0/apache-singa-3.3.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/singa/3.3.0/apache-singa-3.3.0.tar.gz.asc)
- [Release Notes 3.3.0](http://singa.apache.org/docs/releases/RELEASE_NOTES_3.3.0)
- Major changes:
  * New examples
    - Add one CNN example for the BloodMnist dataset, a sub set of MedMNIST.
    - Add one example for the medical image analysis.
  * Enhance distributed training
    - Add key information printing, e.g., throughput and communication time, for
      distributed training.
    - Optimize printing and logging modules for faster distributed training.
  * Enhance example code
    - Add more datasets and model implementations for the cifar_distributed_cnn
      example.
    - Update the running script for the cifar_distributed_cnn example to include
      more models.
    - Update the dataset path for the largedataset_cnn example for more
      flexibility.
    - Add more model implementations for the largedataset_cnn example.
  * Enhance the webpage
    - Reconstruct the singa webpage to include project features.
    - Update the Git web site by deploying it via .asf.yaml.
    - Update the Chinese and Vietnamese documentations.
  * Debug and add assertions for input tensor data types in the opt.py.
  * Change pointer type to void for generalizing data types.
  * Fix bugs
    - Fix the python test error due to operations not implemented for some data
      types.
    - Fix the model of pad from bytes to str.

## V3.2.0 (15 August 2021):

- [Apache SINGA 3.2.0](https://archive.apache.org/dist/singa/3.2.0/apache-singa-3.2.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/singa/3.2.0/apache-singa-3.2.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/singa/3.2.0/apache-singa-3.2.0.tar.gz.asc)
- [Release Notes 3.2.0](http://singa.apache.org/docs/releases/RELEASE_NOTES_3.2.0)
- Major changes:
  * New examples
    - Add one cifar-10 distributed CNN example for benchmarking the performance
      of the distributed training.
    - Add one large CNN example for training with a dataset from the filesysetm.
  * Enhance distributed training
    - Improve the data augmentation module for faster distributed training.
    - Add device synchronization for more accurate time measurements during the
      distributed training.
  * Add Support for half-precision floating-point format (fp16) in deep learning
    models and computational kernels.
  * Update new onnx APIs and fix onnx examples accordingly, namely, DenseNet121,
    ShuffleNetv1, ShuffleNetv2, SqueezeNet, VGG19.
  * Add a new method to resize images by given width and height.
  * Use docusaurus versioning to simplify the process of generating the project
    homepage.
  * Promote code quality
    - Unify the formats of docstrings that describe the contents and usage of
      the module.
    - Unify the parameters of command-line arguments.
  * Fix bugs
    - Fix the CI build error by downloading the tbb binaries.
    - Add disabling graph option for accessing parameter or gradient tensors
      during distributed training.
    - Solve the warnings of deprecated functions in the distributed optimizer
      module.

## V3.1.0 (30 October 2020):

- [Apache SINGA 3.1.0](https://archive.apache.org/dist/singa/3.1.0/apache-singa-3.1.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/singa/3.1.0/apache-singa-3.1.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/singa/3.1.0/apache-singa-3.1.0.tar.gz.asc)
- [Release Notes 3.1.0](http://singa.apache.org/docs/releases/RELEASE_NOTES_3.1.0)
- Major changes:
  * Update Tensor core:
    - Support tensor transformation (reshape, transpose) for tensors up to 6
      dimensions.
    - Implement traverse_unary_transform in Cuda backend, which is similar to
      CPP backend one.
  * Add new tensor operators into the autograd module.
  * Reconstruct sonnx to
    - Support creating operators from both layer and autograd.
    - Re-write SingaRep to provide a more powerful intermediate representation
      of SINGA.
    - Add a SONNXModel which implements from Model to provide uniform API and
      features.
  * Replace the Travis CI with Github workflow. Add quality and coverage
    management.
  * Add compiling and packaging scripts to create wheel packages for
    distribution.
  * Fix bugs
    - Fix IMDB LSTM model example training script.
    - Fix Tensor operation Mult on Broadcasting use cases.
    - Gaussian function on Tensor now can run on Tensor with odd size.
    - Updated a testing helper function gradients() in autograd to lookup param
      gradient by param python object id for testing purpose.

## V3.0.0 (18 April 2020):

- [Apache SINGA 3.0.0](https://archive.apache.org/dist/singa/3.0.0/apache-singa-3.0.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/singa/3.0.0/apache-singa-3.0.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/singa/3.0.0/apache-singa-3.0.0.tar.gz.asc)
- [Release Notes 3.0.0](http://singa.apache.org/docs/releases/RELEASE_NOTES_3.0.0)
- New features and major changes,
  - Enhanced ONNX. Multiple ONNX models have been tested in SINGA.
  - Distributed training with MPI and NCCL Communication optimization through
    gradient sparsification and compression, and chunk transmission.
  - Computational graph construction and optimization for speed and memory using
    the graph.
  - New documentation website (singa.apache.org) and API reference website
    (apache-singa.rtfd.io).
  - CI for code quality check.
  - Replace MKLDNN with DNNL
  - Update tensor APIs to support broadcasting operations.
  - New autograd operators to support ONNX models.

## Incubating v2.0.0 (20 April 2019):

- [Apache SINGA 2.0.0 (incubating)](https://archive.apache.org/dist/incubator/singa/2.0.0/apache-singa-incubating-2.0.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/incubator/singa/2.0.0/apache-singa-incubating-2.0.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/2.0.0/apache-singa-incubating-2.0.0.tar.gz.asc)
- [Release Notes 2.0.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_2.0.0.html)
- New features and major updates,
  - Enhance autograd (for Convolution networks and recurrent networks)
  - Support ONNX
  - Improve the CPP operations via Intel MKL DNN lib
  - Implement tensor broadcasting
  - Move Docker images under Apache user name
  - Update dependent lib versions in conda-build config

## Incubating v1.2.0 (6 June 2018):

- [Apache SINGA 1.2.0 (incubating)](https://archive.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz)
  [\[SHA512\]](https://archive.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz.sha512)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz.asc)
- [Release Notes 1.2.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_1.2.0.html)
- New features and major updates,
  - Implement autograd (currently support MLP model)
  - Upgrade PySinga to support Python 3
  - Improve the Tensor class with the stride field
  - Upgrade cuDNN from V5 to V7
  - Add VGG, Inception V4, ResNet, and DenseNet for ImageNet classification
  - Create alias for conda packages
  - Complete documentation in Chinese
  - Add instructions for running Singa on Windows
  - Update the compilation, CI
  - Fix some bugs

## Incubating v1.1.0 (12 February 2017):

- [Apache SINGA 1.1.0 (incubating)](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz.asc)
- [Release Notes 1.1.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_1.1.0.html)
- New features and major updates,
  - Create Docker images (CPU and GPU versions)
  - Create Amazon AMI for SINGA (CPU version)
  - Integrate with Jenkins for automatically generating Wheel and Debian
    packages (for installation), and updating the website.
  - Enhance the FeedFowardNet, e.g., multiple inputs and verbose mode for
    debugging
  - Add Concat and Slice layers
  - Extend CrossEntropyLoss to accept instance with multiple labels
  - Add image_tool.py with image augmentation methods
  - Support model loading and saving via the Snapshot API
  - Compile SINGA source on Windows
  - Compile mandatory dependent libraries together with SINGA code
  - Enable Java binding (basic) for SINGA
  - Add version ID in checkpointing files
  - Add Rafiki toolkit for providing RESTFul APIs
  - Add examples pretrained from Caffe, including GoogleNet

## Incubating v1.0.0 (8 September 2016):

- [Apache SINGA 1.0.0 (incubating)](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz.asc)
- [Release Notes 1.0.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_1.0.0.html)
- New features and major updates,
  - Tensor abstraction for supporting more machine learning models.
  - Device abstraction for running on different hardware devices, including CPU,
    (Nvidia/AMD) GPU and FPGA (to be tested in later versions).
  - Replace GNU autotool with cmake for compilation.
  - Support Mac OS
  - Improve Python binding, including installation and programming
  - More deep learning models, including VGG and ResNet
  - More IO classes for reading/writing files and encoding/decoding data
  - New network communication components directly based on Socket.
  - Cudnn V5 with Dropout and RNN layers.
  - Replace website building tool from maven to Sphinx
  - Integrate Travis-CI

## Incubating v0.3.0 (20 April 2016):

- [Apache SINGA 0.3.0 (incubating)](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz.asc)
- [Release Notes 0.3.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_0.3.0.html)
- New features and major updates,
  - Training on GPU cluster enables training of deep learning models over a GPU
    cluster.
  - Python wrapper improvement makes it easy to configure the job, including
    neural net and SGD algorithm.
  - New SGD updaters are added, including Adam, AdaDelta and AdaMax.
  - Installation has fewer dependent libraries for single node training.
  - Heterogeneous training with CPU and GPU.
  - Support cuDNN V4.
  - Data prefetching.
  - Fix some bugs.

## Incubating v0.2.0 (14 January 2016):

- [Apache SINGA 0.2.0 (incubating)](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz.asc)
- [Release Notes 0.2.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_0.2.0.html)
- New features and major updates,
  - Training on GPU enables training of complex models on a single node with
    multiple GPU cards.
  - Hybrid neural net partitioning supports data and model parallelism at the
    same time.
  - Python wrapper makes it easy to configure the job, including neural net and
    SGD algorithm.
  - RNN model and BPTT algorithm are implemented to support applications based
    on RNN models, e.g., GRU.
  - Cloud software integration includes Mesos, Docker and HDFS.
  - Visualization of neural net structure and layer information, which is
    helpful for debugging.
  - Linear algebra functions and random functions against Blobs and raw data
    pointers.
  - New layers, including SoftmaxLayer, ArgSortLayer, DummyLayer, RNN layers and
    cuDNN layers.
  - Update Layer class to carry multiple data/grad Blobs.
  - Extract features and test performance for new data by loading previously
    trained model parameters.
  - Add Store class for IO operations.

## Incubating v0.1.0 (8 October 2015):

- [Apache SINGA 0.1.0 (incubating)](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz)
  [\[MD5\]](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz.md5)
  [\[ASC\]](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz.asc)
- [Amazon EC2 image](https://console.aws.amazon.com/ec2/v2/home?region=ap-southeast-1#LaunchInstanceWizard:ami=ami-b41001e6)
- [Release Notes 0.1.0 (incubating)](http://singa.apache.org/docs/releases/RELEASE_NOTES_0.1.0.html)
- Major features include,
  - Installation using GNU build utility
  - Scripts for job management with zookeeper
  - Programming model based on NeuralNet and Layer abstractions.
  - System architecture based on Worker, Server and Stub.
  - Training models from three different model categories, namely, feed-forward
    models, energy models and RNN models.
  - Synchronous and asynchronous distributed training frameworks using CPU
  - Checkpoint and restore
  - Unit test using gtest
