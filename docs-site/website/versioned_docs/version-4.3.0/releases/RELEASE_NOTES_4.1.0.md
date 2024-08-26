---
id: version-4.3.0-RELEASE_NOTES_4.1.0
title: Apache SINGA-4.1.0 Release Notes
original_id: RELEASE_NOTES_4.1.0
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA is a distributed deep learning library.

This release includes following changes:

- New examples

  - Add an example for malaria detection using cell images.
  - Add an example for structured data learning.

- Add support for models running on top of RDBMS

  - Add support for in-database model definition and selection in RDBMS.
  - Implement training-free model evaluation metrics for in-database model
    selection.
  - Implement a coordinator to balance between training-free and training-based
    model evaluations for in-database model selection.

- Enhance distributed training

  - Add implementations for the sum error loss.
  - Improve the optimizer to return model gradients.
  - Improve the iterative checking for tensors and strings in the ModelMeta
    class.

- Enhance example code

  - Add support for flexible setting of training configurations for models,
    e.g., learning rates, weight decay, momentum, etc.
  - Add implementations for dynamic models with varying layer sizes.

- Update the website

  - Add illustrations for database integration.
  - Update users of Apache SINGA.

- Fix bugs
  - Update the NVIDIA_GPGKEY in the Dockerfile for building wheel files.
  - Update the versions of dependencies in the wheel file.
  - Fix the collections module in the model.py file.
