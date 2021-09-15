---
id: version-3.2.0_Viet-optimizer
title: Optimizer
original_id: optimizer
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA hỗ trợ đa dạng các thuật toán tối ưu (optimizers) phổ biến bao gồm
stochastic gradient descent với momentum, Adam, RMSProp, và AdaGrad, etc. Với
mỗi thuật toán tối ưu, SINGA hỗ trợ để sử dụng decay schedular để lên kế hoạch
learning rate áp dụng trong các epochs khác nhau. Các mỗi thuật toán tối ưu và
decay schedulers có trong `singa/opt.py`.

## Tạo thuật toán tối ưu

1. SGD với momentum

```python
# xác định hyperparameter learning rate
lr = 0.001
# xác định hyperparameter momentum
momentum = 0.9
# xác định hyperparameter weight decay
weight_decay = 0.0001

from singa import opt
sgd = opt.SGD(lr=lr, momentum=momentum, weight_decay=weight_decay)
```

2. RMSProp

```python
# xác định hyperparameter learning rate
lr = 0.001
# xác định hyperparameter rho
rho = 0.9
# xác định hyperparameter epsilon
epsilon = 1e-8
# xác định hyperparameter weight decay
weight_decay = 0.0001

from singa import opt
sgd = opt.RMSProp(lr=lr, rho=rho, epsilon=epsilon, weight_decay=weight_decay)
```

3. AdaGrad

```python
# xác định hyperparameter learning rate
lr = 0.001
# xác định hyperparameter epsilon
epsilon = 1e-8
# xác định hyperparameter weight decay
weight_decay = 0.0001

from singa import opt
sgd = opt.AdaGrad(lr=lr, epsilon=epsilon, weight_decay=weight_decay)
```

4. Adam

```python
# xác định hyperparameter learning rate
lr = 0.001
# xác định hyperparameter beta 1
beta_1= 0.9
# xác định hyperparameter beta 2
beta_1= 0.999
# xác định hyperparameter epsilon
epsilon = 1e-8
# xác định hyperparameter weight decay
weight_decay = 0.0001

from singa import opt
sgd = opt.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, weight_decay=weight_decay)
```

## Tạo Decay Scheduler

```python
from singa import opt

# xác định learning rate ban đầu
lr_init = 0.001
# xác định rate của decay trong decay scheduler
decay_rate = 0.95
# xác định learning rate schedule có ở dạng staircase shape
staircase=True
# xác định bước decay của decay scheduler (trong ví dụ này lr giảm sau mỗi 2 bước)
decay_steps = 2

# tạo decay scheduler, schedule của lr trở thành lr_init * (decay_rate ^ (step // decay_steps) )
lr = opt.ExponentialDecay(0.1, 2, 0.5, True)
# sử dụng lr để tạo một thuật toán tối ưu
sgd = opt.SGD(lr=lr, momentum=0.9, weight_decay=0.0001)
```

## Sử dụng thuật toán tối ưu trong Model API

Khi tạo mô hình model, cần đính kèm thuật toán tối ưu vào model.

```python
# tạo CNN sử dụng Model API
model = CNN()

# khởi tạo thuật toán tối ưu và đính vào model
sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)
model.set_optimizer(sgd)
```

Sau đó, khi gọi hàm model, chạy phương pháp `train_one_batch` để sử dụng thuật
toán tối ưu.

Do vậy, một ví dụ cho lặp lại loop để tối ưu hoá model là:

```python
for b in range(num_train_batch):
    # tạo mini-batch tiếp theo
    x, y = ...

    # Copy dữ liệu vào tensors đầu vào
    tx.copy_from_numpy(x)
    ty.copy_from_numpy(y)

    # Train với một batch
    out, loss = model(tx, ty)
```
