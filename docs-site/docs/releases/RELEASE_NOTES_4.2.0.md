---
id: RELEASE_NOTES_4.2.0
title: Apache SINGA-4.2.0 Release Notes
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA is a distributed deep learning library.

This release includes following changes:

- Add support for deep learning models running on top of PolarDB

  - Implement efficient model selection for a given dataset stored in the
    database
  - Add support for dynamic model creation
  - Add support for flexible setting of model training configurations
  - Optimize the in-database analytics modules for scalability, efficiency and
    memory consumption

- New example

  - Add a horizontal federated learning example using the Bank dataset

- Enhance examples

  - Add sample training data for testing the model selection application

- Update the website

  - Update the star button in the main page
  - Refine the display of star statistics

- Update the python versions for wheel files

- Fix bugs
  - Fix the rat check files
  - Update the license files
