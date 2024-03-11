---
id: RELEASE_NOTES_4.0.0
title: Apache SINGA-4.0.0 Release Notes
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA is a distributed deep learning library.

This release includes following changes:

- Enhance distributed training

  - Add support for configuration of number of GPUs to be used.
  - Increase max epoch for better convergence.
  - Print intermediate mini-batch information.
  - Add support for switching between CPU and GPU devices.

- Enhance example code

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

- Enhance the webpage

  - Update online documentation for distributed training.

- Promote code quality

  - Update inline comments for prepreocessing and data loading.

- Update the PIL image module

- Update the runtime Dockerfile

- Update the conda files
