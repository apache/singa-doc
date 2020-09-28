---
id: optimizer
title: Optimizer
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA supports various popular optimizers including stochastic gradient descent
with momentum, Adam, RMSProp, and AdaGrad, etc. For each of the optimizer, it
supports to use a decay schedular to schedule the learning rate to be applied in
different epochs. The optimizers and the decay schedulers are included in
`singa/opt.py`.

## Create an optimizer

1. SGD with momentum

```python
# define hyperparameter learning rate
lr = 0.001
# define hyperparameter momentum
momentum = 0.9
# define hyperparameter weight decay
weight_decay = 0.0001

from singa import opt
sgd = opt.SGD(lr=lr, momentum=momentum, weight_decay=weight_decay)
```

2. RMSProp

```python
# define hyperparameter learning rate
lr = 0.001
# define hyperparameter rho
rho = 0.9
# define hyperparameter epsilon
epsilon = 1e-8
# define hyperparameter weight decay
weight_decay = 0.0001

from singa import opt
sgd = opt.RMSProp(lr=lr, rho=rho, epsilon=epsilon, weight_decay=weight_decay)
```

3. AdaGrad

```python
# define hyperparameter learning rate
lr = 0.001
# define hyperparameter epsilon
epsilon = 1e-8
# define hyperparameter weight decay
weight_decay = 0.0001

from singa import opt
sgd = opt.AdaGrad(lr=lr, epsilon=epsilon, weight_decay=weight_decay)
```

4. Adam

```python
# define hyperparameter learning rate
lr = 0.001
# define hyperparameter beta 1
beta_1= 0.9
# define hyperparameter beta 2
beta_1= 0.999
# define hyperparameter epsilon
epsilon = 1e-8
# define hyperparameter weight decay
weight_decay = 0.0001

from singa import opt
sgd = opt.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, weight_decay=weight_decay)
```

## Create a Decay Scheduler

```python
from singa import opt

# define initial learning rate
lr_init = 0.001
# define the rate of decay in the decay scheduler
decay_rate = 0.95
# define whether the learning rate schedule is a staircase shape
staircase=True
# define the decay step of the decay scheduler (in this example the lr is decreased at every 2 steps)
decay_steps = 2

# create the decay scheduler, the schedule of lr becomes lr_init * (decay_rate ^ (step // decay_steps) )
lr = opt.ExponentialDecay(0.1, 2, 0.5, True)
# Use the lr to create an optimizer
sgd = opt.SGD(lr=lr, momentum=0.9, weight_decay=0.0001)
```

## Use the optimizer in Model API

When we create the model, we need to attach the optimizer to the model.

```python
# create a CNN using the Model API
model = CNN()

# initialize optimizer and attach it to the model
sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)
model.set_optimizer(sgd)
```

Then, when we call the model, it runs the `train_one_batch` method that utilizes
the optimizer.

Hence, an example of an iterative loop to optimize the model is:

```python
for b in range(num_train_batch):
    # generate the next mini-batch
    x, y = ...

    # Copy the data into input tensors
    tx.copy_from_numpy(x)
    ty.copy_from_numpy(y)

    # Training with one batch
    out, loss = model(tx, ty)
```
