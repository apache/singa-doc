---
id: graph
title: Computational Graph
---

<!-- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License. -->

SINGA supports buffering operations and computational graph. By using computational graph, SINGA can schedule the execution of operations and the memory allocation and release which makes training more efficient while using less memory.

## Features
There are three main features of computational graph, namely the construction of the computational graph, lazy allocation, automatic recycling and synchronization pipeline. Details as follows:
* `Computational graph construction`: Construct a computational graph based on the user-defined neural network or expressions and then run the graph to accomplish the training task. The computational graph also includes operations like synch and fused synch in the communicator.
* `Lazy allocation`: When blocks need to be allocated, devices won't allocate memory for them immediately. Only when an operation uses this block for the first time, memory allocation will be performed.
* `Automatic recycling`: Automatically deallocate the intermediate tensors which won't be used again in the following operations when we are running the graph in an iteration.
* `Synchronization pipeline`: In previous synchronization operations, buffers were used to synchronize multiple tensors at once. But the communicator needs to collect all the tensors before copying them into the buffer. Synchronization pipeline can copy tensors to the buffer separately, which reduces the time for synchronous operations.

## Design
### Computational graph construction
* Use the technique of delayed execution to falsely perform operations in the forward propagation and backward propagation once. Buffer all the operations and the tensors read or written by each operation. 
* Calculate dependencies between all the operations to decide the order of execution. (Support directed cyclic graph)
* Execute all the operations in the order we just calculated to update all the parameters.
* The system will only analyze the same graph once. If new operations are added to the graph, the calculation graph will be re-analyzed.
* Provided a module class for users to use this feature more conveniently.
### Lazy allocation
* When a device needs to create a new block, just pass the device to that block instead of allocating a piece of memory from the mempool and passing the pointer to that block.
* When the block is accessed for the first time, let the device corresponding to the block allocate memory and then access it.
### Automatic recycling
* When calculating dependencies between the operations during the graph construction, the reference count of tensors can also be calculated.
* When an operation is completed, we can decrease the reference count of tensors the operation used.
* If a tensor's reference count reaches zero, it means the tensor won't be accessed by latter operations and we can recycle its memory.
* The program will track the usage of the block. If a block is used on the python side, it will not be recycled, which is convenient for debugging on the python side.
### Synchronization pipeline
* If a tensor needs fusion synchronization, it will be copied to the buffer immediately and don't need to gather all the tensors. Because the copy operation is advanced, it takes less time to do real synchronization. This optimizes the use of the GPU.


## How to use
* An example of CNN:
```Python

class CNN(module.Module):

    def __init__(self, optimizer):
        super(CNN, self).__init__()

        self.conv1 = autograd.Conv2d(1, 20, 5, padding=0)
        self.conv2 = autograd.Conv2d(20, 50, 5, padding=0)
        self.linear1 = autograd.Linear(4 * 4 * 50, 500)
        self.linear2 = autograd.Linear(500, 10)
        self.pooling1 = autograd.MaxPool2d(2, 2, padding=0)
        self.pooling2 = autograd.MaxPool2d(2, 2, padding=0)

        self.optimizer = optimizer

    def forward(self, x):
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
        return autograd.softmax_cross_entropy(x, ty)

    def optim(self, loss):
        self.optimizer.backward_and_update(loss)

# initialization other objects
# ......
model = CNN(sgd)
model.train()
model.on_device(dev)
model.graph(graph, sequential)

# Train
for b in range(num_train_batch):
    # Generate the patch data in this iteration
    # ......

    # Copy the patch data into input tensors
    tx.copy_from_numpy(x)
    ty.copy_from_numpy(y)

    # Train the model
    out = model(tx)
    loss = model.loss(out, ty)
    model.optim(loss)
```
* Some settings: [module.py](https://github.com/apache/singa/blob/master/python/singa/module.py)
    * `trainng`: whether to train the neural network defined in the class or for evaluation
    * `graph_mode`: the model class defined by users can be trained using computational graph or not.
    * `sequential`: execute operations in graph serially or in the order of BFS.
* More examples:
    * [MLP](https://github.com/apache/singa/blob/master/examples/autograd/mlp_module.py)
    * [CNN](https://github.com/apache/singa/blob/master/examples/autograd/cnn_module.py)
    * [ResNet](https://github.com/apache/singa/blob/master/examples/autograd/resnet_module.py)


## Evaluation
### Single node
* Experiment settings
    * Model
      * using layer: ResNet50 in [resnet.py](https://github.com/apache/singa/blob/master/examples/autograd/resnet.py)
      * using module: ResNet50 in [resnet_module.py](https://github.com/apache/singa/blob/master/examples/autograd/resnet_module.py)
    * GPU: Nvidia RTX 2080Ti
* Explanation
    * `s` ：second
    * `it` ： iteration
    * `Mem`：peak memory usage of single GPU
    * `Throughout`：number of pictures processed per second
    * `Time`：total time
    * `Speed`：iterations per second
    * `Reduction`：the memory usage reduction rate compared with dev branch
    * `Seepdup`: speedup ratio compared with dev branch
* Result
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
        <td nowrap>module:enable graph</td>
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
* Experiment settings
    * Model
      * using Layer: ResNet50 in [resnet_dist.py](https://github.com/apache/singa/blob/master/examples/autograd/resnet_dist.py)
      * using Module: ResNet50 in [resnet_module.py](https://github.com/apache/singa/blob/master/examples/autograd/resnet_module.py)
    * GPU: Nvidia RTX 2080Ti \* 2
    * MPI: two MPI processes on one node
* Explanation: the same as above
* Result
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

* Computational graph does not affect training time and memory usage if the graph is disabled (has backward compatibility).
* Computational graph can significantly reduce memory usage and training time.

## Include operations in graph

For new operations, if they need to included in the computational graph, they should be submitted to the device. Device class in CPP will add these operations in the computational graph and scheduler will schedule them automatically.

#### Requirements

 When submitting operations, there are some requirements.

* Need to pass in the function that the operation executes and the data blocks that the operation reads and writes

* For the function of the operation: All variables used in lambda expressions need to be captured according to the following rules.

  * `capture by value`: If the variable is a local variable or will be immediately released(e.g. intermediate tensors). If not captured by value, these variables will be destroyed after buffering. Buffering is just a way to defer real calculations.
  * `capture by reference`：If the variable is recorded on the python side or a global variable(e.g. The parameter W and ConvHand in the Conv2d class). 

  * `mutable`: The lambda expression should have mutable tag if a variable captured by value is modified in an expression

#### Example

* Python side: [_Conv2d](https://github.com/apache/singa/blob/dev/python/singa/autograd.py#L1191) records x, W, b and handle in the class.

```python
class _Conv2d(Operation):

    def __init__(self, handle, odd_padding=(0, 0, 0, 0)):
        super(_Conv2d, self).__init__()
        self.handle = handle  # record handle
        self.odd_padding = odd_padding
        if self.odd_padding != (0, 0, 0, 0):
            self.re_new_handle = True

    def forward(self, x, W, b=None):
		# other code
        # ......

        if training:
            if self.handle.bias_term:
                self.inputs = (x, W, b) # record x, W, b
            else:
                self.inputs = (x, W)

		# other code
        # ......

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
        if self.odd_padding != (0, 0, 0, 0):
            dx = utils.handle_odd_pad_bwd(dx, self.odd_padding)

        if db:
            return dx, dW, db

        else:
            return dx, dW
```

* C++ side: [convolution.cc](https://github.com/apache/singa/blob/dev/src/model/operation/convolution.cc)

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

## Future features

- [ ] Graph substitution: replace a subgraph of the input computation graph with another subgraph which is functionally equivalent to the original one. 
- [ ] Support recalculation and swapping out variables from the GPU to reduce memory usage.
- [ ] Perform operations in the graph in the order of DFS.
- [ ] Performing operations in parallel on single GPU.
