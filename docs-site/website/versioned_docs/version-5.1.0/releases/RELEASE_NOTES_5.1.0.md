---
id: version-5.1.0-RELEASE_NOTES_5.1.0
title: Apache SINGA-5.1.0 Release Notes
original_id: RELEASE_NOTES_5.1.0
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA is a distributed deep learning library.

This release includes following changes:

  - Add the implementations for Low-Rank Adaptation of LLMs (LORA).

  - Add the implementations for Parameter-Efficient Fine-Tuning (PEFT).
    - Add the implementations of distributed models for SINGA PEFT.
    - Add the implementations of tuners for SINGA PEFT.
    - Add the implementations of the autograd functions for SINGA PEFT.
    - Add the multiprocess implementations for SINGA PEFT.

  - Update the healthcare model zoo.
    - Restructure the folders for the healthcare model zoo.
    - Update the training files and model files for the healthcare model zoo.
    - Update the documentations for the healthcare model zoo.

  - Add more healthcare examples.
    - Add the implementations of the candidiasis disease application.
    - Add the implementations of the cardiovascular disease application.

  - Add the implementations for more attention mechanisms.

  - Update the encoder layers for the transformer model.

  - Fix typos in Python module docstrings.
