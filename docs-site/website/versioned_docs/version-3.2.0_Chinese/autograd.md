---
id: version-3.2.0_Chinese-autograd
title: Autograd
original_id: autograd
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

实现autograd有两种典型的方式，一种是通过如[Theano](http://deeplearning.net/software/theano/index.html)的符号微分（symbolic differentiation）或通过如[Pytorch](https://pytorch.org/docs/stable/notes/autograd.html)的反向微分（reverse differentialtion）。SINGA遵循Pytorch方式，即通过记录计算图，并在正向传播后自动应用反向传播。自动传播算法的详细解释请参阅[这里](https://pytorch.org/docs/stable/notes/autograd.html)。我们接下来对SINGA中的相关模块进行解释，并举例说明其使用方法。

## 相关模块

在autograd中涉及三个类，分别是`singa.tensor.Tensor`，`singa.autograd.Operation`和`singa.autograd.Layer`。在本篇的后续部分中，我们使用Tensor、Operation和Layer来指代这三个类。

### Tensor

Tensor的三个属性被autograd使用：

- `.creator`是一个`Operation`实例。它记录了产生Tensor实例的这个操作。
- `.request_grad`是一个布尔变量。它用于指示autograd算法是否需要计算张量的梯度。例如，在反向传播的过程中，线性层的权重矩阵和卷积层（非底层）的特征图的张量梯度应该被计算。
- `.store_grad`是一个布尔变量。它用于指示张量的梯度是否应该被存储并由后向函数输出。例如，特征图的梯度是在反向传播过程中计算出来的，但不包括在反向函数的输出中。

开发者可以改变Tensor实例的`requires_grad`和`stores_grad`。例如，如果将后者设置为True，那么相应的梯度就会被包含在后向函数的输出。需要注意的是，如果`stores_grad`是True，那么 `requires_grad`一定是真，反之亦然。


### Operation

它将一个或多个`Tensor`实例作为输入，然后输出一个或多个`Tensor`实例。例如，ReLU可以作为一个具体的Operation子类来实现。当一个`Operation`实例被调用时（实例化后），会执行以下两个步骤。

1.记录源操作，即输入张量的`创建者`。
2.通过调用成员函数`.forward()`进行计算。

有两个成员函数用于前向和反向传播，即`.forward()`和`.backward()`。它们以`Tensor.data`作为输入（类型为`CTensor`），并输出`Ctensor`s。要添加一个特定的操作，子类`Operation`应该实现自己的`.forward()`和`.backward()`函数。在后向传播过程中，autograd的`backward()`函数会自动调用`backward()`函数来计算输入的梯度（根据`require_grad`字段的参数和约束）。

### Layer

对于那些需要参数的Operation，我们把它们封装成一个新的类，`Layer`。例如，卷积操作被封装到卷积层(Convolution layer)中。`层`管理（存储）参数，并调用相应的`Operation`来实现变换。

## 样例

在[example folder](https://github.com/apache/singa/tree/master/examples/autograd)中提供了很多样例。在这里我我们分析两个最具代表性的例子。

### 只使用Operation

下一段代码展示了一个只使用`Operation`的多层感知机（MLP）模型：

#### 调用依赖包

```python
from singa.tensor import Tensor
from singa import autograd
from singa import opt
```

#### 创建权重矩阵和偏置向量

在将`requires_grad`和`stores_grad`都设置为`True`的情况下，创建参数张量。

```python
w0 = Tensor(shape=(2, 3), requires_grad=True, stores_grad=True)
w0.gaussian(0.0, 0.1)
b0 = Tensor(shape=(1, 3), requires_grad=True, stores_grad=True)
b0.set_value(0.0)

w1 = Tensor(shape=(3, 2), requires_grad=True, stores_grad=True)
w1.gaussian(0.0, 0.1)
b1 = Tensor(shape=(1, 2), requires_grad=True, stores_grad=True)
b1.set_value(0.0)
```

#### 训练

```python
inputs = Tensor(data=data)  # data matrix
target = Tensor(data=label) # label vector
autograd.training = True    # for training
sgd = opt.SGD(0.05)   # optimizer

for i in range(10):
    x = autograd.matmul(inputs, w0) # matrix multiplication
    x = autograd.add_bias(x, b0)    # add the bias vector
    x = autograd.relu(x)            # ReLU activation operation

    x = autograd.matmul(x, w1)
    x = autograd.add_bias(x, b1)

    loss = autograd.softmax_cross_entropy(x, target)

    for p, g in autograd.backward(loss):
        sgd.update(p, g)
```

### 使用Operation和Layer

下面的[例子](https://github.com/apache/singa/blob/master/examples/autograd/mnist_cnn.py)使用autograd模块提供的层实现了一个CNN模型。

#### 创建层

```python
conv1 = autograd.Conv2d(1, 32, 3, padding=1, bias=False)
bn1 = autograd.BatchNorm2d(32)
pooling1 = autograd.MaxPool2d(3, 1, padding=1)
conv21 = autograd.Conv2d(32, 16, 3, padding=1)
conv22 = autograd.Conv2d(32, 16, 3, padding=1)
bn2 = autograd.BatchNorm2d(32)
linear = autograd.Linear(32 * 28 * 28, 10)
pooling2 = autograd.AvgPool2d(3, 1, padding=1)
```

#### 定义正向传播函数

在正向传播中的operations会被自动记录，用于反向传播。

```python
def forward(x, t):
    # x is the input data (a batch of images)
    # t is the label vector (a batch of integers)
    y = conv1(x)           # Conv layer
    y = autograd.relu(y)   # ReLU operation
    y = bn1(y)             # BN layer
    y = pooling1(y)        # Pooling Layer

    # two parallel convolution layers
    y1 = conv21(y)
    y2 = conv22(y)
    y = autograd.cat((y1, y2), 1)  # cat operation
    y = autograd.relu(y)           # ReLU operation
    y = bn2(y)
    y = pooling2(y)

    y = autograd.flatten(y)        # flatten operation
    y = linear(y)                  # Linear layer
    loss = autograd.softmax_cross_entropy(y, t)  # operation
    return loss, y
```

#### 训练

```python
autograd.training = True
for epoch in range(epochs):
    for i in range(batch_number):
        inputs = tensor.Tensor(device=dev, data=x_train[
                               i * batch_sz:(1 + i) * batch_sz], stores_grad=False)
        targets = tensor.Tensor(device=dev, data=y_train[
                                i * batch_sz:(1 + i) * batch_sz], requires_grad=False, stores_grad=False)

        loss, y = forward(inputs, targets) # forward the net

        for p, gp in autograd.backward(loss):  # auto backward
            sgd.update(p, gp)
```

### Using the Model API

下面的[样例](https://github.com/apache/singa/blob/master/examples/cnn/model/cnn.py)使用[Model API](./graph)实现了一个CNN模型。.

#### 定义Model的子类

定义模型类，它应该是Model的子类。只有这样，在训练阶段使用的所有操作才会形成一个计算图以便进行分析。图中的操作将被按时序规划并有效执行，模型类中也可以包含层。

```python
class MLP(model.Model):  # the model is a subclass of Model

    def __init__(self, data_size=10, perceptron_size=100, num_classes=10):
        super(MLP, self).__init__()

        # init the operators, layers and other objects
        self.relu = layer.ReLU()
        self.linear1 = layer.Linear(perceptron_size)
        self.linear2 = layer.Linear(num_classes)
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()

    def forward(self, inputs):  # define the forward function
        y = self.linear1(inputs)
        y = self.relu(y)
        y = self.linear2(y)
        return y

    def train_one_batch(self, x, y):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)
        self.optimizer(loss)
        return out, loss

    def set_optimizer(self, optimizer):  # attach an optimizer
        self.optimizer = optimizer
```

#### 训练

```python
# create a model instance
model = MLP()
# initialize optimizer and attach it to the model
sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)
model.set_optimizer(sgd)
# input and target placeholders for the model
tx = tensor.Tensor((batch_size, 1, IMG_SIZE, IMG_SIZE), dev, tensor.float32)
ty = tensor.Tensor((batch_size, num_classes), dev, tensor.int32)
# compile the model before training
model.compile([tx], is_train=True, use_graph=True, sequential=False)

# train the model iteratively
for b in range(num_train_batch):
    # generate the next mini-batch
    x, y = ...

    # Copy the data into input tensors
    tx.copy_from_numpy(x)
    ty.copy_from_numpy(y)

    # Training with one batch
    out, loss = model(tx, ty)
```

#### 保存模型checkpoint

```python
# define the path to save the checkpoint
checkpointpath="checkpoint.zip"

# save a checkpoint
model.save_states(fpath=checkpointpath)
```

#### 加载模型checkpoint

```python
# define the path to load the checkpoint
checkpointpath="checkpoint.zip"

# load a checkpoint
import os
if os.path.exists(checkpointpath):
    model.load_states(fpath=checkpointpath)
```

### Python API

关于Python API的更多细节，请参考[这里](https://singa.readthedocs.io/en/latest/autograd.html#module-singa.autograd)。
