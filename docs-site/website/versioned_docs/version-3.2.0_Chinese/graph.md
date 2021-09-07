---
id: version-3.2.0_Chinese-graph
title: Model
original_id: graph
---

<!-- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License. -->

神经网络中的前向和反向传播可以用一组操作来表示，比如卷积和池化。每个操作都需要一些输入的[tensors](./tensor)，并应用一个[operator](./autograd)来生成输出的张量。通过将每个运算符表示为一个节点，将每个张量表示为一条边，所有的运算就形成了一个计算图。有了计算图，可以通过调度运算的执行和内存的智能分配/释放来进行速度和内存优化。在SINGA中，用户只需要使用[Model](https://github.com/apache/singa/blob/master/python/singa/model.py) API定义神经网络模型，计算图则会在C++后台自动构建和优化。


这样，一方面，用户使用[Model](./graph) API按照PyTorch那样的命令式编程风格实现网络。而与PyTorch在每次迭代中重新创建操作不同的是，SINGA在第一次迭代后就会缓冲操作，隐式地创建计算图（当该功能被启用时）。因此，另一方面，SINGA的计算图与使用声明式编程的库（如TensorFlow）创建的计算图类似，因而它可以享受在图上进行的优化。

## 样例

下面的代码说明了`Model`API的用法：

1. 将新模型实现为Model类的子类：

```Python
class CNN(model.Model):

    def __init__(self, num_classes=10, num_channels=1):
        super(CNN, self).__init__()
        self.conv1 = layer.Conv2d(num_channels, 20, 5, padding=0, activation="RELU")
        self.conv2 = layer.Conv2d(20, 50, 5, padding=0, activation="RELU")
        self.linear1 = layer.Linear(500)
        self.linear2 = layer.Linear(num_classes)
        self.pooling1 = layer.MaxPool2d(2, 2, padding=0)
        self.pooling2 = layer.MaxPool2d(2, 2, padding=0)
        self.relu = layer.ReLU()
        self.flatten = layer.Flatten()
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()

    def forward(self, x):
        y = self.conv1(x)
        y = self.pooling1(y)
        y = self.conv2(y)
        y = self.pooling2(y)
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        return y

    def train_one_batch(self, x, y):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)
        self.optimizer(loss)
        return out, loss
```

2. 创建model、optimizer、device等的实例。编译模型：

```python
model = CNN()

# initialize optimizer and attach it to the model
sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)
model.set_optimizer(sgd)

# initialize device
dev = device.create_cuda_gpu()

# input and target placeholders for the model
tx = tensor.Tensor((batch_size, 1, IMG_SIZE, IMG_SIZE), dev, tensor.float32)
ty = tensor.Tensor((batch_size, num_classes), dev, tensor.int32)

# compile the model before training
model.compile([tx], is_train=True, use_graph=True, sequential=False)
```

3. 迭代训练：

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

这个例子的Google Colab notebook可以在[这里](https://colab.research.google.com/drive/1fbGUs1AsoX6bU5F745RwQpohP4bHTktq)找到。


更多例子：

- [MLP](https://github.com/apache/singa/blob/master/examples/mlp/model.py)
- [CNN](https://github.com/apache/singa/blob/master/examples/cnn/model/cnn.py)
- [ResNet](https://github.com/apache/singa/blob/master/examples/cnn/model/resnet.py)

## 实现

### 图的构建

SINGA分三步构建计算图：

1. 将操作保存在缓冲区。
2. 分析操作的依赖性。
3. 根据依赖关系创建节点和边。

以MLP模型的dense层的矩阵乘法运算为例，该操作会在[MLP model](https://github.com/apache/singa/blob/master/examples/mlp/model.py)的前向函数中被调用：

```python
class MLP(model.Model):

    def __init__(self, data_size=10, perceptron_size=100, num_classes=10):
        super(MLP, self).__init__()
        self.linear1 = layer.Linear(perceptron_size)
        ...

    def forward(self, inputs):
        y = self.linear1(inputs)
        ...
```

`线性`层由`mutmul`运算符组成，`autograd`通过SWIG调用CPP中提供的`Mult`函数来实现`matmul`运算符。

```python
# implementation of matmul()
singa.Mult(inputs, w)
```

At the backend, the `Mult` function is implemented by calling `GEMV` a CBLAS
function. Instead of calling `GEMV` directly, `Mult` submits `GEMV` and the
arguments to the device as follows,
在后端，`Mult`函数是通过调用`GEMV`一个CBLAS函数来实现的。但`Mult`没有直接调用`GEMV`，而是将`GEMV`和参数提交给设备，具体如下。

```c++
// implementation of Mult()
C->device()->Exec(
    [a, A, b, B, CRef](Context *ctx) mutable {
        GEMV<DType, Lang>(a, A, B, b, &CRef, ctx);
    },
    read_blocks, {C->block()});
```

`Device`的`Exec`函数对函数及其参数进行缓冲。此外，它还拥有这个函数要读写的块的信息（块是指张量的内存块）。

一旦`Model.forward()`被执行一次，所有的操作就会被`Device`缓冲。接下来，对所有操作的读写信息进行分析，用来建立计算图。例如，如果一个块`b`被一个操作O1写入，之后又被另一个操作O2读出，我们就会知道O2依赖于O1并且有一条从A到B的有向边，它代表了块`b`（或其张量）。之后我们就构建了一个有向无环图，如下图所示。该图会构建一次。

![The computational graph of MLP](assets/GraphOfMLP.png)

<br/>**Figure 1 - MLP例子的计算图**

### 优化

目前，基于计算图进行了以下优化：

**惰性分配** 当创建张量/块时，设备不会立即为它们分配内存。相反，是在第一次访问块时，才会分配内存。

**自动回收**  每个张量/块的参考计数是根据图计算出来的。在执行操作之前，参考计数是读取这个块的操作次数。在执行过程中，一旦执行了一个操作，每一个输入块的参考数就会减少1，如果一个块的参考数达到了0，就意味着这个块在剩下的操作中不会再被读取。因此，它的内存可以被安全释放。此外，SINGA还会跟踪图外的块的使用情况。如果一个块被Python代码使用（而不是被autograd操作符使用），它将不会被回收。

**内存共享**  SINGA使用内存池，如[CnMem](https://github.com/NVIDIA/cnmem)来管理CUDA内存。有了自动回收和内存池，SINGA就可以在张量之间共享内存。考虑两个操作`c=a+b`和`d=2xc`。在执行第二个操作之前，根据惰性分配原则，应该分配d的内存。假设`a`在其余操作中没有使用。根据自动回收，`a`的块将在第一次操作后被释放。因此，SINGA会向CUDA流提交四个操作：加法、释放`a`、分配`b`和乘法。这样，内存池就可以将`a`释放的内存与`b`共享，而不是要求GPU为`b`做真正的malloc。

其他的优化技术，如来自编译器的优化技术，如常见的子表达式消除和不同CUDA流上的并行化操作也可以应用。

## 新的操作符

`autograd`模块中定义的每个运算符都实现了两个功能：前向和反向，通过在后台调用运算符来实现。如果要在`autograd`中添加一个新的运算符，需要在后台添加多个运算符。

以[Conv2d](https://github.com/apache/singa/blob/master/python/singa/autograd.py)运算符为例，在Python端，根据设备类型，从后台调用运算符来实现前向和反向功能：

```python
class _Conv2d(Operation):

    def forward(self, x, W, b=None):
        ......
        if training:
            if self.handle.bias_term:
                self.inputs = (x, W, b) # record x, W, b
            else:
                self.inputs = (x, W)

        if (type(self.handle) != singa.ConvHandle):
            return singa.GpuConvForward(x, W, b, self.handle)
        else:
            return singa.CpuConvForward(x, W, b, self.handle)

    def backward(self, dy):
        if (type(self.handle) != singa.ConvHandle):
            dx = singa.GpuConvBackwardx(dy, self.inputs[1], self.inputs[0],
                                        self.handle)
            dW = singa.GpuConvBackwardW(dy, self.inputs[0], self.inputs[1],
                                        self.handle)
            db = singa.GpuConvBackwardb(
                dy, self.inputs[2],
                self.handle) if self.handle.bias_term else None
        else:
            dx = singa.CpuConvBackwardx(dy, self.inputs[1], self.inputs[0],
                                        self.handle)
            dW = singa.CpuConvBackwardW(dy, self.inputs[0], self.inputs[1],
                                        self.handle)
            db = singa.CpuConvBackwardb(
                dy, self.inputs[2],
                self.handle) if self.handle.bias_term else None
        if db:
            return dx, dW, db
        else:
            return dx, dW
```

对于后台的每一个操作符，应按以下方式实现：

- 假设操作符是`foo()`，它的真正实现应该包装在另一个函数中，例如`_foo()`。`foo()`将`_foo`和参数一起作为lambda函数传递给`Device`的`Exec`函数进行缓冲，要读和写的块也同时被传递给`Exec`。

- lambda表达式中使用的所有参数都需要根据以下规则获取：

  - `值捕获`: 如果参数变量是一个局部变量，或者将被立刻释放（例如，中间时序）。否则，一旦`foo()`存在，这些变量将被销毁。
  - `引用捕获`：如果变量是记录在python端或者是一个持久变量（例如Conv2d类中的参数W和ConvHand）。

  - `可变捕获`: 如果在`_foo()`中修改了由值捕获的变量，则lambda表达式应带有mutable（可变）标签。

下面是一个在后台实现的操作的[例子](https://github.com/apache/singa/blob/master/src/model/operation/convolution.cc)：

```c++
Tensor GpuConvBackwardx(const Tensor &dy, const Tensor &W, const Tensor &x,
                        const CudnnConvHandle &cch) {
  CHECK_EQ(dy.device()->lang(), kCuda);

  Tensor dx;
  dx.ResetLike(x);

  dy.device()->Exec(
      /*
       * dx is a local variable so it's captured by value
       * dy is an intermediate tensor and isn't recorded on the python side
       * W is an intermediate tensor but it's recorded on the python side
       * chh is a variable and it's recorded on the python side
       */
      [dx, dy, &W, &cch](Context *ctx) mutable {
        Block *wblock = W.block(), *dyblock = dy.block(), *dxblock = dx.block();
        float alpha = 1.f, beta = 0.f;
        cudnnConvolutionBackwardData(
            ctx->cudnn_handle, &alpha, cch.filter_desc, wblock->data(),
            cch.y_desc, dyblock->data(), cch.conv_desc, cch.bp_data_alg,
            cch.workspace.block()->mutable_data(),
            cch.workspace_count * sizeof(float), &beta, cch.x_desc,
            dxblock->mutable_data());
      },
      {dy.block(), W.block()}, {dx.block(), cch.workspace.block()});
      /* the lambda expression reads the blocks of tensor dy and w
       * and writes the blocks of tensor dx and chh.workspace
       */

  return dx;
}
```

## Benchmark

### 单节点

- 实验设定
  - 模型：
    - 使用层: ResNet50 in
      [resnet.py](https://github.com/apache/singa/blob/master/examples/cnn/autograd/resnet_cifar10.py)
    - 使用模型: ResNet50 in
      [resnet.py](https://github.com/apache/singa/blob/master/examples/cnn/model/resnet.py)
  - GPU: NVIDIA RTX 2080Ti
- 注释：
  - `s` ：second，秒
  - `it` ： iteration，迭代次数
  - `Mem`：peak memory usage of single GPU，单GPU显存峰值
  - `Throughout`：number of images processed per second，每秒处理的图像数
  - `Time`：total time，总时间
  - `Speed`：iterations per second。每秒迭代次数
  - `Reduction`：the memory usage reduction rate compared with that using layer，与使用层的内存使用率相比，内存使用率降低了多少
  - `Speedup`: speedup ratio compared with dev branch，与dev分支相比的加速率
- 结果：
  <table style="text-align: center">
      <tr>
          <th style="text-align: center">Batchsize</th>
          <th style="text-align: center">Cases</th>
          <th style="text-align: center">Mem(MB)</th>
          <th style="text-align: center">Time(s)</th>
          <th style="text-align: center">Speed(it/s)</th>
          <th style="text-align: center">Throughput</th>
          <th style="text-align: center">Reduction</th>
          <th style="text-align: center">Speedup</th>
      </tr>
      <tr>
          <td rowspan="4">16</td>
          <td nowrap>layer</td>
          <td>4975</td>
          <td>14.1952</td>
          <td>14.0893</td>
          <td>225.4285</td>
          <td>0.00%</td>
          <td>1.0000</td>
      </tr>
      <tr>
          <td nowrap>model:disable graph</td>
          <td>4995</td>
          <td>14.1264</td>
          <td>14.1579</td>
          <td>226.5261</td>
          <td>-0.40%</td>
          <td>1.0049</td>
      </tr>
      <tr>
          <td nowrap>model:enable graph, bfs</td>
          <td>3283</td>
          <td>13.7438</td>
          <td>14.5520</td>
          <td>232.8318</td>
          <td>34.01%</td>
          <td>1.0328</td>
      </tr>
      <tr>
          <td nowrap>model:enable graph, serial</td>
          <td>3265</td>
          <td>13.7420</td>
          <td>14.5540</td>
          <td>232.8635</td>
          <td>34.37%</td>
          <td>1.0330</td>
      </tr>
      <tr>
          <td rowspan="4">32</td>
          <td nowrap>layer</td>
          <td>10119</td>
          <td>13.4587</td>
          <td>7.4302</td>
          <td>237.7649</td>
          <td>0.00%</td>
          <td>1.0000</td>
      </tr>
      <tr>
          <td nowrap>model:disable graph</td>
          <td>10109</td>
          <td>13.2952</td>
          <td>7.5315</td>
          <td>240.6875</td>
          <td>0.10%</td>
          <td>1.0123</td>
      </tr>
      <tr>
          <td nowrap>model:enable graph, bfs</td>
          <td>6839</td>
          <td>13.1059</td>
          <td>7.6302</td>
          <td>244.1648</td>
          <td>32.41%</td>
          <td>1.0269</td>
      </tr>
      <tr>
          <td nowrap>model:enable graph, serial</td>
          <td>6845</td>
          <td>13.0489</td>
          <td>7.6635</td>
          <td>245.2312</td>
          <td>32.35%</td>
          <td>1.0314</td>
      </tr>
  </table>

### 多线程

- 实验设置：
  - API：
    - 使用层: ResNet50 in
      [resnet_dist.py](https://github.com/apache/singa/blob/master/examples/cnn/autograd/resnet_dist.py)
    - 使用模型: ResNet50 in
      [resnet.py](https://github.com/apache/singa/blob/master/examples/cnn/model/resnet.py)
  - GPU: NVIDIA RTX 2080Ti \* 2
  - MPI: 在同一节点上的两个MPI processes
- 注释: 与上面相同
- 结果：
  <table style="text-align: center">
      <tr>
          <th style="text-align: center">Batchsize</th>
          <th style="text-align: center">Cases</th>
          <th style="text-align: center">Mem(MB)</th>
          <th style="text-align: center">Time(s)</th>
          <th style="text-align: center">Speed(it/s)</th>
          <th style="text-align: center">Throughput</th>
          <th style="text-align: center">Reduction</th>
          <th style="text-align: center">Speedup</th>
      </tr>
      <tr>
          <td rowspan="4">16</td>
          <td nowrap>layer</td>
          <td>5439</td>
          <td>17.3323</td>
          <td>11.5391</td>
          <td>369.2522</td>
          <td>0.00%</td>
          <td>1.0000</td>
      </tr>
      <tr>
          <td nowrap>model:disable graph</td>
          <td>5427</td>
          <td>17.8232</td>
          <td>11.2213</td>
          <td>359.0831</td>
          <td>0.22%</td>
          <td>0.9725</td>
      </tr>
      <tr>
          <td nowrap>model:enable graph, bfs</td>
          <td>3389</td>
          <td>18.2310</td>
          <td>10.9703</td>
          <td>351.0504</td>
          <td>37.69%</td>
          <td>0.9507</td>
      </tr>
      <tr>
          <td nowrap>model:enable graph, serial</td>
          <td>3437</td>
          <td>17.0389</td>
          <td>11.7378</td>
          <td>375.6103</td>
          <td>36.81%</td>
          <td>1.0172</td>
      </tr>
      <tr>
          <td rowspan="4">32</td>
          <td nowrap>layer</td>
          <td>10547</td>
          <td>14.8635</td>
          <td>6.7279</td>
          <td>430.5858</td>
          <td>0.00%</td>
          <td>1.0000</td>
      </tr>
      <tr>
          <td nowrap>model:disable graph</td>
          <td>10503</td>
          <td>14.7746</td>
          <td>6.7684</td>
          <td>433.1748</td>
          <td>0.42%</td>
          <td>1.0060</td>
      </tr>
      <tr>
          <td nowrap>model:enable graph, bfs</td>
          <td>6935</td>
          <td>14.8553</td>
          <td>6.7316</td>
          <td>430.8231</td>
          <td>34.25%</td>
          <td>1.0006</td>
      </tr>
      <tr>
          <td nowrap>model:enable graph, serial</td>
          <td>7027</td>
          <td>14.3271</td>
          <td>6.9798</td>
          <td>446.7074</td>
          <td>33.37%</td>
          <td>1.0374</td>
      </tr>
  </table>

### 结论

- 在启用计算图的情况下进行训练，可以显著减少内存占用。
- 目前，在速度上有一点改进。在效率方面还可以做更多的优化。
