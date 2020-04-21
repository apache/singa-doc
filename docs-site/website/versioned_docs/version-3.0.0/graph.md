---
id: version-3.0.0-graph
title: Computational Graph
original_id: graph
---

<!-- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License. -->

The forward and backward propagation in a neural network can be represented
using a set of operations such as convolution and pooling. Each operation takes
some input [tensors](./tensor) and applies an [operator](./autograd) to generate
output tensors. By representing each operator as a node and each tensor as an
edge, all operations form a computational graph. With the computational graph,
speed and memory optimization can be conducted by scheduling the execution of
the operations and memory allocation/release intelligently. In SINGA, users only
need to define the neural network model using the
[Module](https://github.com/apache/singa/blob/master/python/singa/module.py)
API. The graph is constructed and optimized at the C++ backend automatically.

## Example

The following code illustrates the usage of the `Module` API.

1. Implement the new model as a subclass the Module class.

```Python
class CNN(module.Module):

    def __init__(self, optimizer):
        super(CNN, self).__init__()
        # define layers
        self.conv1 = autograd.Conv2d(1, 20, 5, padding=0)
        self.conv2 = autograd.Conv2d(20, 50, 5, padding=0)
        self.linear1 = autograd.Linear(4 * 4 * 50, 500)
        self.linear2 = autograd.Linear(500, 10)
        self.pooling1 = autograd.MaxPool2d(2, 2, padding=0)
        self.pooling2 = autograd.MaxPool2d(2, 2, padding=0)

        self.optimizer = optimizer

    def forward(self, x):
        # define the forward operations
        y = self.conv1(x)
        y = autograd.relu(y)
        y = self.pooling1(y)
        y = self.conv2(y)
        y = autograd.relu(y)
        y = self.pooling2(y)
        y = autograd.flatten(y)
        y = self.linear1(y)
        y = autograd.relu(y)
        y = self.linear2(y)
        return y

    def loss(self, x, ty):
        # define the training loss
        return autograd.softmax_cross_entropy(x, ty)

    def optim(self, loss):
        # update the parameters using SGD algorithms
        self.optimizer.backward_and_update(loss)
```

2. Create an instance of the model, and do some configurations

```python
model = CNN(sgd)
# set the mode of running the operations:
# True for training; False for evaluation
model.train(mode=True)
# set the device for running the operations
model.on_device(dev)
# whether to create the graph or run the operations imperatively
model.graph(mode=True)
```

3. Train the model iteratively

```python
for b in range(num_train_batch):
    # generate the next mini-batch
    x, y = ...

    # Copy the data into input tensors
    tx.copy_from_numpy(x)
    ty.copy_from_numpy(y)

    # run forward propagation
    out = model(tx)
    loss = model.loss(out, ty)
    # run backward propagation
    model.optim(loss)
```

A Google Colab notebook of this example is available
[here](https://colab.research.google.com/drive/1fbGUs1AsoX6bU5F745RwQpohP4bHTktq).

More examples:

- [MLP](https://github.com/apache/singa/blob/master/examples/mlp/module.py)
- [CNN](https://github.com/apache/singa/blob/master/examples/cnn/model/cnn.py)
- [ResNet](https://github.com/apache/singa/blob/master/examples/cnn/model/resnet.py)

## Implementation

### Graph Construction

SINGA constructs the computational graph in three steps:

1. buffer the operations
2. analyze the dependencies operations
3. create the nodes and edges based on the dependencies

Take the matrix multiplication operation from the dense layer of a
[MLP model](https://github.com/apache/singa/blob/master/examples/mlp/module.py)
as an example. The operation is called in the `forward` function of the MLP
class

```python
class MLP(module.Module):

    def forward(self, inputs):
        x = autograd.matmul(inputs, self.w0)
        ...
```

`autograd` implements the `matmul` operator by calling the function `Mult`
exposed from CPP via SWIG.

```python
# implementation of matmul()
singa.Mult(inputs, w)
```

At the backend, the `Mult` function is implemented by calling `GEMV` a CBLAS
function. Instead of calling `GEMV` directly, `Mult` submits `GEMV` and the
arguments to the device as follows,

```c++
// implementation of Mult()
C->device()->Exec(
    [a, A, b, B, CRef](Context *ctx) mutable {
        GEMV<DType, Lang>(a, A, B, b, &CRef, ctx);
    },
    read_blocks, {C->block()});
```

The `Exec` function of `Device` buffers the function and its arguments. In
addition, it also has the information about the blocks (a block is a chunk of
memory for a tensor) to be read and written by this function.

Once `Module.forward()` has been executed once, all operations are buffered by
`Device`. Next, the read/write information of all operations are analyzed to
create the computational graph. For example, if a block `b` is written by one
operation O1 and is later read by another operation O2, we would know O2 depends
on O1 and there is a directed edge from A to B, which represents block `b` (or
its tensor). After that a directed acyclic graph is constructed as shown below.
The graph is constructed once.

![The computational graph of MLP](assets/GraphOfMLP.png)

<br/>**Figure 1 - The computational graph of the MLP example.**

### Optimization

Currently, the following optimizations are done based on the computational
graph.

**Lazy allocation** When tensor/blocks are created, devices do not allocate
memory for them immediately. Instead, when the block is accessed for the first
time, the memory is allocated.

**Automatic recycling** The reference count of each tensor/block is calculated
based on the graph. Before executing the operations, the reference count is the
number of operations that read this block. During the execution, once an
operation is executed, the reference count of the every input block is decreased
by 1. If one block's reference count reaches 0, it means that this block will
not be read again in the remaining operations. Therefore, its memory can be
released safely. In addition, SINGA tracks the usage of the block outside of the
graph. If a block is used by Python code (not by autograd operators), it will
not be recycled.

**Memory sharing** SINGA uses memory pool, e.g.,
[CnMem](https://github.com/NVIDIA/cnmem) to manage CUDA memory. With _Automatic
recycling_ and memory pool, SINGA can share the memory among tensors. Consider
two operations `c = a + b` and `d=2xc`. Before executing the second operation,
according to _Lazy allocation_, the memory of d should be allocated. Suppose `a`
is not used in the rest operations. According to Automatic recycling, the block
of `a` will be released after the first operation. Therefore, SINGA would submit
four operations to the CUDA stream: addition, free `a`, malloc `b`, and
multiplication. The memory pool is then able to share the memory released by `a`
with `b` instead of ask the GPU to do real malloc for `b`.

Other optimization techniques e.g., from compliers, such as common
sub-expression elimination and parallelizing operations on different CUDA
streams can also be applied.

## New Operator

Each operator defined in `autograd` module implements two functions: forward and
backward, which are implemented by calling the operators from the backend. To
add a new operator in `autograd`, you need to add the multiple operators at the
backend.

Take the
[Conv2d](https://github.com/apache/singa/blob/master/python/singa/autograd.py)
operator as an example, at the Python side, the forward and backward function
are implemented by calling the operators from the backend depending on the
device type.

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

For each operator at the backend, it should be implemented in the following way:

- Suppose the operator is `foo()`; its real implementation should be wrapped in
  another function e.g., `_foo()`. `foo()` passes `_foo` together with the
  arguments as a lambda function to `Device`'s `Exec` function for buffering.
  The blocks to be read and written are also passed to `Exec`.

- All arguments used in the lambda expression need to be captured according to
  the following rules.

  - `capture by value`: If the argument variable is a local variable or will be
    immediately released (e.g. intermediate tensors). Otherwise, these variables
    will be destroyed once `foo()` exists.
  - `capture by reference`：If the variable is recorded on the python side or a
    persistent variable (e.g. parameter W and ConvHand in the Conv2d class).

  - `mutable`: The lambda expression should have the mutable tag if a variable
    captured by value is modified in `_foo()`

Here is one
[example](https://github.com/apache/singa/blob/master/src/model/operation/convolution.cc)
operator implemented at the backend.

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

### Single node

- Experiment settings
  - Model
    - Using layer: ResNet50 in
      [resnet.py](https://github.com/apache/singa/blob/master/examples/cnn/autograd/resnet_cifar10.py)
    - Using module: ResNet50 in
      [resnet.py](https://github.com/apache/singa/blob/master/examples/cnn/model/resnet.py)
  - GPU: NVIDIA RTX 2080Ti
- Notations
  - `s` ：second
  - `it` ： iteration
  - `Mem`：peak memory usage of single GPU
  - `Throughout`：number of images processed per second
  - `Time`：total time
  - `Speed`：iterations per second
  - `Reduction`：the memory usage reduction rate compared with that using layer
  - `Speedup`: speedup ratio compared with dev branch
- Result
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
          <td nowrap>module:disable graph</td>
          <td>4995</td>
          <td>14.1264</td>
          <td>14.1579</td>
          <td>226.5261</td>
          <td>-0.40%</td>
          <td>1.0049</td>
      </tr>
      <tr>
          <td nowrap>module:enable graph, bfs</td>
          <td>3283</td>
          <td>13.7438</td>
          <td>14.5520</td>
          <td>232.8318</td>
          <td>34.01%</td>
          <td>1.0328</td>
      </tr>
      <tr>
          <td nowrap>module:enable graph, serial</td>
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
          <td nowrap>module:disable graph</td>
          <td>10109</td>
          <td>13.2952</td>
          <td>7.5315</td>
          <td>240.6875</td>
          <td>0.10%</td>
          <td>1.0123</td>
      </tr>
      <tr>
          <td nowrap>module:enable graph, bfs</td>
          <td>6839</td>
          <td>13.1059</td>
          <td>7.6302</td>
          <td>244.1648</td>
          <td>32.41%</td>
          <td>1.0269</td>
      </tr>
      <tr>
          <td nowrap>module:enable graph, serial</td>
          <td>6845</td>
          <td>13.0489</td>
          <td>7.6635</td>
          <td>245.2312</td>
          <td>32.35%</td>
          <td>1.0314</td>
      </tr>
  </table>

### Multi processes

- Experiment settings
  - Model
    - using Layer: ResNet50 in
      [resnet_dist.py](https://github.com/apache/singa/blob/master/examples/cnn/autograd/resnet_dist.py)
    - using Module: ResNet50 in
      [resnet.py](https://github.com/apache/singa/blob/master/examples/cnn/model/resnet.py)
  - GPU: NVIDIA RTX 2080Ti \* 2
  - MPI: two MPI processes on one node
- Notations: the same as above
- Result
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
          <td nowrap>module:disable graph</td>
          <td>5427</td>
          <td>17.8232</td>
          <td>11.2213</td>
          <td>359.0831</td>
          <td>0.22%</td>
          <td>0.9725</td>
      </tr>
      <tr>
          <td nowrap>module:enable graph, bfs</td>
          <td>3389</td>
          <td>18.2310</td>
          <td>10.9703</td>
          <td>351.0504</td>
          <td>37.69%</td>
          <td>0.9507</td>
      </tr>
      <tr>
          <td nowrap>module:enable graph, serial</td>
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
          <td nowrap>module:disable graph</td>
          <td>10503</td>
          <td>14.7746</td>
          <td>6.7684</td>
          <td>433.1748</td>
          <td>0.42%</td>
          <td>1.0060</td>
      </tr>
      <tr>
          <td nowrap>module:enable graph, bfs</td>
          <td>6935</td>
          <td>14.8553</td>
          <td>6.7316</td>
          <td>430.8231</td>
          <td>34.25%</td>
          <td>1.0006</td>
      </tr>
      <tr>
          <td nowrap>module:enable graph, serial</td>
          <td>7027</td>
          <td>14.3271</td>
          <td>6.9798</td>
          <td>446.7074</td>
          <td>33.37%</td>
          <td>1.0374</td>
      </tr>
  </table>

### Conclusion

- Training with the computational graph enabled can significantly reduce the
  memory footprint.
- Currently, there is a little improvement in terms of speed. More optimizations
  can be done towards the efficiency.
