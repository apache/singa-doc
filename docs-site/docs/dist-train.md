---
id: dist-train
title: Distributed Training
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA supports data parallel training across multiple GPUs (on a single node or
across different nodes). The following figure illustrates the data parallel
training:

![MPI.png](assets/MPI.png)

In distributed training, each process (called a worker) runs a training script
over a single GPU. Each process has an individual communication rank. The
training data is partitioned among the workers and the model is replicated on
every worker. In each iteration, the workers read a mini-batch of data (e.g.,
256 images) from its partition and run the BackPropagation algorithm to compute
the gradients of the weights, which are averaged via All-Reduce (provided by
[NCCL](https://developer.nvidia.com/nccl)) for weight update following
stochastic gradient descent algorithms (SGD).

The all-reduce operation by NCCL can be used to reduce and synchronize the
gradients from different GPUs. Let's consider the training with 4 GPUs as shown
below. Once the gradients from the 4 GPUs are calculated, All-Reduce will the
sum of the gradients over the GPUs and make it available on every GPU. Then the
averaged gradients can be easily calculated.

![AllReduce.png](assets/AllReduce.png)

## Usage

SINGA implements a module called `DistOpt` for distributed training. It replaces
the normal SGD optimizer for updating the model parameters. The following
example illustrates the usage of `DistOpt` for training a CNN model over the
MNIST dataset. The full example is available
[here](https://github.com/apache/singa/blob/master/examples/autograd/mnist_dist.py).

### Example Code

1. Define the neural network model:

```python
class CNN:
    def __init__(self):
        self.conv1 = autograd.Conv2d(1, 20, 5, padding=0)
        self.conv2 = autograd.Conv2d(20, 50, 5, padding=0)
        self.linear1 = autograd.Linear(4 * 4 * 50, 500)
        self.linear2 = autograd.Linear(500, 10)
        self.pooling1 = autograd.MaxPool2d(2, 2, padding=0)
        self.pooling2 = autograd.MaxPool2d(2, 2, padding=0)

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

# create model
model = CNN()
```

2. Create the `DistOpt` instance:

```python
sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)
sgd = opt.DistOpt(sgd)
dev = device.create_cuda_gpu_on(sgd.rank_in_local)
```

`dev` represents the `Device` instance, where to load data and run the CNN
model.

3. Load and partition the training/validation data:

```python
train_x, train_y, test_x, test_y = load_dataset()
train_x, train_y = data_partition(train_x, train_y,
                                  sgd.rank_in_global, sgd.world_size)
test_x, test_y = data_partition(test_x, test_y,
                                sgd.rank_in_global, sgd.world_size)
```

A partition of the dataset is returned for this `dev`.

4. Initialize and synchronize the model parameters among all workers:

```python
def synchronize(tensor, dist_opt):
    dist_opt.all_reduce(tensor.data)
    tensor /= dist_opt.world_size

#Synchronize the initial parameter
tx = tensor.Tensor((batch_size, 1, IMG_SIZE, IMG_SIZE), dev, tensor.float32)
ty = tensor.Tensor((batch_size, num_classes), dev, tensor.int32)
...
out = model.forward(tx)
loss = autograd.softmax_cross_entropy(out, ty)
for p, g in autograd.backward(loss):
    synchronize(p, sgd)
```

5. Run BackPropagation and distributed SGD

```python
for epoch in range(max_epoch):
    for b in range(num_train_batch):
        x = train_x[idx[b * batch_size: (b + 1) * batch_size]]
        y = train_y[idx[b * batch_size: (b + 1) * batch_size]]
        tx.copy_from_numpy(x)
        ty.copy_from_numpy(y)
        out = model.forward(tx)
        loss = autograd.softmax_cross_entropy(out, ty)
        # do backpropagation and all-reduce
        sgd.backward_and_update(loss)
```

### Execution Instruction

There are two ways to launch the training: MPI or Python multiprocessing.

#### Python multiprocessing

It works on a single node with multiple GPUs, where each GPU is one worker.

1. Put all the above training codes in a function

```python
def train_mnist_cnn(nccl_id=None, gpu_num=None, gpu_per=None):
    ...
```

2. Create `mnist_multiprocess.py`

```python
if __name__ == '__main__':
    # Generate a NCCL ID to be used for collective communication
    nccl_id = singa.NcclIdHolder()

    # Define the number of GPUs to be used in the training process
    gpu_per_node = int(sys.argv[1])
    gpu_num = 1

    # Define and launch the multi-processing
	import multiprocessing
    process = []
    for gpu_num in range(0, gpu_per_node):
        process.append(multiprocessing.Process(target=train_mnist_cnn,
                       args=(nccl_id, gpu_num, gpu_per_node)))

    for p in process:
        p.start()
```

The arguments for creating the `DistOpt` instance should be updated as follows

```python
sgd = opt.DistOpt(sgd, nccl_id=nccl_id, gpu_num=gpu_num, gpu_per_node=gpu_per_node)
```

3. Run `mnist_multiprocess.py`

```sh
python mnist_multiprocess.py
```

#### MPI

It works for both single node and multiple nodes as long as there are multiple
GPUs.

1. Create `mnist_dist.py`

```python
if __name__ == '__main__':
    train_mnist_cnn()
```

2. Generate a hostfile for MPI, e.g. the hostfile below uses 4 processes (i.e.,
   4 GPUs) on a single node

```txt
localhost:4
```

3. Launch the training via `mpiexec`

```sh
mpiexec --hostfile host_file python3 mnist_dist.py
```

It could result in several times speed up compared to the single GPU training.

```
Starting Epoch 0:
Training loss = 673.246277, training accuracy = 0.760517
Evaluation accuracy = 0.930489, Elapsed Time = 0.757460s
Starting Epoch 1:
Training loss = 240.009323, training accuracy = 0.919705
Evaluation accuracy = 0.964042, Elapsed Time = 0.707835s
Starting Epoch 2:
Training loss = 168.806030, training accuracy = 0.944010
Evaluation accuracy = 0.967448, Elapsed Time = 0.710606s
Starting Epoch 3:
Training loss = 139.131454, training accuracy = 0.953676
Evaluation accuracy = 0.971755, Elapsed Time = 0.710840s
Starting Epoch 4:
Training loss = 117.479889, training accuracy = 0.960487
Evaluation accuracy = 0.974659, Elapsed Time = 0.711388s
Starting Epoch 5:
Training loss = 103.085609, training accuracy = 0.965812
Evaluation accuracy = 0.979267, Elapsed Time = 0.712624s
Starting Epoch 6:
Training loss = 97.565521, training accuracy = 0.966897
Evaluation accuracy = 0.979868, Elapsed Time = 0.714128s
Starting Epoch 7:
Training loss = 86.971985, training accuracy = 0.970903
Evaluation accuracy = 0.979868, Elapsed Time = 0.715277s
Starting Epoch 8:
Training loss = 79.487328, training accuracy = 0.973341
Evaluation accuracy = 0.982372, Elapsed Time = 0.715577s
Starting Epoch 9:
Training loss = 74.658951, training accuracy = 0.974793
Evaluation accuracy = 0.982672, Elapsed Time = 0.717571s
```

## Optimizations for Distributed Training

SINGA provides multiple optimization strategies for distributed training to
reduce the communication cost. Refer to the API for `DistOpt` for the
configuration of each strategy.

### No Optimizations

```python
sgd.backward_and_update(loss)
```

`loss` is the output tensor from the loss function, e.g., cross-entropy for
classification tasks.

### Half-precision Gradients

```python
sgd.backward_and_update_half(loss)
```

It converts each gradient value to 16-bit representation (i.e., half-precision)
before calling AllReduce.

### Partial Synchronization

```python
sgd.backward_and_partial_update(loss)
```

In each iteration, only a chunk of of gradients are averaged, which saves the
communication cost. The other gradients are used to update the parameters
locally. The chunk size is configured when creating the `DistOpt` instance.

### Gradient Sparsification

```python
sgd.backward_and_spars_update(loss)
```

It applies sparsification schemes to select a subset of gradients for
All-Reduce. There are two scheme:

- The top-K largest elements are selected
- All gradients whose absolute value are larger than predefined threshold.

The hype-parameters are configured when creating the `DistOpt` instance.
