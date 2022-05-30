---
id: RELEASE_NOTES_3.3.0
title: Apache SINGA-3.3.0 Release Notes
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA is a distributed deep learning library.

This release includes following changes:

- New examples

  - Add one CNN example for the BloodMnist dataset, a sub set of MedMNIST.
  - Add one example for the medical image analysis.

- Enhance distributed training 

  - Add key information printing, e.g., throughput and communication time, for distributed training.
  - Optimize printing and logging modules for faster distributed training.

- Enhance example code

  - Add more datasets and model implementations for the cifar_distributed_cnn example.
  - Update the running script for the cifar_distributed_cnn example to include more models.
  - Update the dataset path for the largedataset_cnn example for more flexibility.
  - Add more model implementations for the largedataset_cnn example.

- Enhance the webpage

  - Reconstruct the singa webpage to include project features.
  - Update the Git web site by deploying it via .asf.yaml.
  - Update the Chinese and Vietnamese documentations.

- Debug and add assertions for input tensor data types in the opt.py.

- Change pointer type to void for generalizing data types.
  
- Fix bugs

  - Fix the python test error due to operations not implemented for some data types.
  - Fix the model of pad from bytes to str.
