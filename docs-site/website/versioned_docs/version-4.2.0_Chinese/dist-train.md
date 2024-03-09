---
id: version-4.2.0_Chinese-dist-train
title: Distributed Training
original_id: dist-train
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA 支持跨多个 GPU 的数据并行训练（在单个节点上或跨不同节点）。下图说明了数据
并行训练的情况：

![MPI.png](assets/MPI.png)

在分布式训练中，每个进程(称为 worker)在单个 GPU 上运行一个训练脚本，每个进程都有
一个单独的通信等级，训练数据被分发给各个 worker，模型在每个 worker 上被复制。在
每次迭代中，worker 从其分区中读取数据的一个 mini-batch（例如，256 张图像），并运
行 BackPropagation 算法来计算权重的梯度，通过 all-reduce（
由[NCCL](https://developer.nvidia.com/nccl)提供）进行平均，按照随机梯度下降算法
（SGD）进行权重更新。

NCCL 的 all-reduce 操作可以用来减少和同步不同 GPU 的梯度。假设我们使用 4 个 GPU
进行训练，如下图所示。一旦计算出 4 个 GPU 的梯度，all-reduce 将返回 GPU 的梯度之
和，并使其在每个 GPU 上可用，然后就可以轻松计算出平均梯度。

![AllReduce.png](assets/AllReduce.png)

## 使用

SINGA 提供了一个名为`DistOpt`（`Opt`的一个子类）的模块，用于分布式训练。它封装了
一个普通的 SGD 优化器，并调用`Communicator`进行梯度同步。下面的例子说明了如何使
用`DistOpt`在 MNIST 数据集上训练一个 CNN 模型。源代码
在[这里](https://github.com/apache/singa/blob/master/examples/cnn/)，或者可以使
用[Colab notebook]()。

### 代码示例

1. 定义神经网络模型：

```python
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

    def train_one_batch(self, x, y, dist_option='fp32', spars=0):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)

        # Allow different options for distributed training
        # See the section "Optimizations for Distributed Training"
        if dist_option == 'fp32':
            self.optimizer(loss)
        elif dist_option == 'fp16':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)
        return out, loss

# create model
model = CNN()
```

2. 创建`DistOpt`实例并将其应用到创建的模型上：

```python
sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)
sgd = opt.DistOpt(sgd)
model.set_optimizer(sgd)
dev = device.create_cuda_gpu_on(sgd.local_rank)
```

下面是关于代码中一些变量的解释：

(i) `dev`

dev 代表`Device`实例，在设备中加载数据并运行 CNN 模型。

(ii)`local_rank`

Local rank 表示当前进程在同一节点中使用的 GPU 数量。例如，如果你使用的节点有 2
个 GPU，`local_rank=0`表示这个进程使用的是第一个 GPU，而`local_rank=1`表示使用的
是第二个 GPU。使用 MPI 或多进程，你能够运行相同的训练脚本，唯一的区
别`local_rank`的值不同。

(iii)`global_rank`

global 中的 rank 代表了你使用的所有节点中所有进程的全局排名。让我们考虑这样的情
况：你有 3 个节点，每个节点有两个 GPU， `global_rank=0`表示使用第 1 个节点的第 1
个 GPU 的进程， `global_rank=2`表示使用第 2 个节点的第 1 个 GPU 的进程，
`global_rank=4`表示使用第 3 个节点的第 1 个 GPU 的进程。

3. 加载和分割训练/验证数据：

```python
def data_partition(dataset_x, dataset_y, global_rank, world_size):
    data_per_rank = dataset_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    return dataset_x[idx_start:idx_end], dataset_y[idx_start:idx_end]

train_x, train_y, test_x, test_y = load_dataset()
train_x, train_y = data_partition(train_x, train_y,
                                  sgd.global_rank, sgd.world_size)
test_x, test_y = data_partition(test_x, test_y,
                                sgd.global_rank, sgd.world_size)
```

这个`dev`的数据集的一个分区被返回。

这里，`world_size`代表你用于分布式训练的所有节点中的进程总数。

4. 初始化并同步所有 worker 的模型参数:

```python
#Synchronize the initial parameter
tx = tensor.Tensor((batch_size, 1, IMG_SIZE, IMG_SIZE), dev, tensor.float32)
ty = tensor.Tensor((batch_size, num_classes), dev, tensor.int32)
model.compile([tx], is_train=True, use_graph=graph, sequential=True)
...
#Use the same random seed for different ranks
seed = 0
dev.SetRandSeed(seed)
np.random.seed(seed)
```

5. 运行 BackPropagation 和分布式 SGD

```python
for epoch in range(max_epoch):
    for b in range(num_train_batch):
        x = train_x[idx[b * batch_size: (b + 1) * batch_size]]
        y = train_y[idx[b * batch_size: (b + 1) * batch_size]]
        tx.copy_from_numpy(x)
        ty.copy_from_numpy(y)
        # Train the model
        out, loss = model(tx, ty)
```

### 执行命令

有两种方式可以启动训练，MPI 或 Python multiprocessing。

#### Python multiprocessing

它可以在一个节点上使用多个 GPU，其中，每个 GPU 是一个 worker。

1. 将上述训练用的代码打包进一个函数：

```python
def train_mnist_cnn(nccl_id=None, local_rank=None, world_size=None):
    ...
```

2. 创建`mnist_multiprocess.py`。

```python
if __name__ == '__main__':
    # Generate a NCCL ID to be used for collective communication
    nccl_id = singa.NcclIdHolder()

    # Define the number of GPUs to be used in the training process
    world_size = int(sys.argv[1])

    # Define and launch the multi-processing
	import multiprocessing
    process = []
    for local_rank in range(0, world_size):
        process.append(multiprocessing.Process(target=train_mnist_cnn,
                       args=(nccl_id, local_rank, world_size)))

    for p in process:
        p.start()
```

下面是关于上面所创建的变量的一些说明：

(i) `nccl_id`

需要注意的是，我们在这里需要生成一个 NCCL ID，用于集体通信，然后将其传递给所有进
程。NCCL ID 就像一个门票，只有拥有这个 ID 的进程才能加入到 all-reduce 操作中。(
如果我们接下来使用 MPI，NCCL ID 的传递就没有必要了，因为在我们的代码中，这个 ID
会由 MPI 自动广播。)

(ii) `world_size`

world_size 是您想用于训练的 GPU 数量。

(iii) `local_rank`

local_rank 决定分布式训练的本地顺序，以及在训练过程中使用哪个 gpu。在上面的代码
中，我们使用 for 循环来运行训练函数，其中参数 local_rank 从 0 迭代到
world_size。在这种情况下，不同的进程可以使用不同的 GPU 进行训练。

创建`DistOpt`实例的参数应按照如下方式更新：

```python
sgd = opt.DistOpt(sgd, nccl_id=nccl_id, local_rank=local_rank, world_size=world_size)
```

3. 运行`mnist_multiprocess.py`：

```sh
python mnist_multiprocess.py 2
```

与单 GPU 训练相比，它最主要的意义是速度提升：

```
Starting Epoch 0:
Training loss = 408.909790, training accuracy = 0.880475
Evaluation accuracy = 0.956430
Starting Epoch 1:
Training loss = 102.396790, training accuracy = 0.967415
Evaluation accuracy = 0.977564
Starting Epoch 2:
Training loss = 69.217010, training accuracy = 0.977915
Evaluation accuracy = 0.981370
Starting Epoch 3:
Training loss = 54.248390, training accuracy = 0.982823
Evaluation accuracy = 0.984075
Starting Epoch 4:
Training loss = 45.213406, training accuracy = 0.985560
Evaluation accuracy = 0.985276
Starting Epoch 5:
Training loss = 38.868435, training accuracy = 0.987764
Evaluation accuracy = 0.986278
Starting Epoch 6:
Training loss = 34.078186, training accuracy = 0.989149
Evaluation accuracy = 0.987881
Starting Epoch 7:
Training loss = 30.138697, training accuracy = 0.990451
Evaluation accuracy = 0.988181
Starting Epoch 8:
Training loss = 26.854443, training accuracy = 0.991520
Evaluation accuracy = 0.988682
Starting Epoch 9:
Training loss = 24.039650, training accuracy = 0.992405
Evaluation accuracy = 0.989083
```

#### MPI

只要有多个 GPU，MPI 既适用于单节点，也适用于多节点。

1. 创建`mnist_dist.py`。

```python
if __name__ == '__main__':
    train_mnist_cnn()
```

2. 为 MPI 生成一个 hostfile，例如下面的 hostfile 在一个节点上使用了 2 个进程（即
   2 个 GPU）：

```txt
localhost:2
```

3. 通过`mpiexec`启动训练：

```sh
mpiexec --hostfile host_file python mnist_dist.py
```

与单 GPU 训练相比，它同样可以带来速度的提升：

```
Starting Epoch 0:
Training loss = 383.969543, training accuracy = 0.886402
Evaluation accuracy = 0.954327
Starting Epoch 1:
Training loss = 97.531479, training accuracy = 0.969451
Evaluation accuracy = 0.977163
Starting Epoch 2:
Training loss = 67.166870, training accuracy = 0.978516
Evaluation accuracy = 0.980769
Starting Epoch 3:
Training loss = 53.369656, training accuracy = 0.983040
Evaluation accuracy = 0.983974
Starting Epoch 4:
Training loss = 45.100403, training accuracy = 0.985777
Evaluation accuracy = 0.986078
Starting Epoch 5:
Training loss = 39.330826, training accuracy = 0.987447
Evaluation accuracy = 0.987179
Starting Epoch 6:
Training loss = 34.655270, training accuracy = 0.988799
Evaluation accuracy = 0.987780
Starting Epoch 7:
Training loss = 30.749735, training accuracy = 0.989984
Evaluation accuracy = 0.988281
Starting Epoch 8:
Training loss = 27.422146, training accuracy = 0.991319
Evaluation accuracy = 0.988582
Starting Epoch 9:
Training loss = 24.548153, training accuracy = 0.992171
Evaluation accuracy = 0.988682
```

## 针对分布式训练的优化

SINGA 为分布式训练提供了多种优化策略，以降低模块间的通信成本。参考`DistOpt`的
API，了解每种策略的配置。

当我们使用`model.Model`建立模型时，我们需要在`training_one_batch`方法中启用分布
式训练的选项，请参考本页顶部的示例代码。我们也可以直接复制这些选项的代码，然后在
其他模型中使用。

有了定义的选项，我们可以在使用`model(tx, ty, dist_option, spars)`开始训练时，设
置对应的参数`dist_option`和`spars`。

### 不采取优化手段

```python
out, loss = model(tx, ty)
```

`loss`是损失函数的输出张量，例如分类任务中的交叉熵。

### 半精度梯度（Half-precision Gradients）

```python
out, loss = model(tx, ty, dist_option = 'fp16')
```

在调用 all-reduce 之前，它将每个梯度值转换为 16-bit 表示（即半精度）。

### 部分同步（Partial Synchronization）

```python
out, loss = model(tx, ty, dist_option = 'partialUpdate')
```

在每一次迭代中，每个 rank 都做本地 SGD 更新。然后，只对一个部分的参数进行平均同
步，从而节省了通信成本。分块大小是在创建`DistOpt`实例时配置的。

### 梯度稀疏化（Gradient Sparsification）

该策略应用稀疏化方案来选择梯度的子集进行 all-reduce，有两种方式：

- 选择前 k 大的元素，spars 是被选择的元素的一部分（比例在 0 - 1 之间）。

```python
out, loss = model(tx, ty, dist_option = 'sparseTopK', spars = spars)
```

- 所有绝对值大于预定义阈值的梯度都会被选中。

```python
out, loss = model(tx, ty, dist_option = 'sparseThreshold', spars = spars)
```

超参数在创建`DistOpt`实例时配置。

## 实现

本节主要是让开发者了解分布训练模块的代码是如何实现的。

### NCCL communicator 的 C 接口

首先，通信层是用 C 语
言[communicator.cc](https://github.com/apache/singa/blob/master/src/io/communicator.cc)编
写的，它调用用 NCCL 库进行集体通信。

communicator 有两个构造器，一个是 MPI 的，另一个是 Multiprocess 的。

(i) MPI 构造器

构造器首先先获取全局 rank 和 world_size，计算出本地 rank，然后由 rank 0 生成
NCCL ID 并广播给每个 rank。之后，它调用 setup 函数来初始化 NCCL
communicator、cuda 流和缓冲区。

(ii) Python multiprocess 构造器

构造器首先从输入参数中获取 rank、world_size 和 NCCL ID。之后，调用 setup 函数来
初始化 NCCL communicator、cuda 流和缓冲区。

在初始化之后，它提供了 all-reduce 功能来同步模型参数或梯度。例如，synch 接收一个
输入张量，通过 NCCL 例程进行 all-reduce，在我们调用 synch 之后，需要调用 wait 函
数来等待 all-reduce 操作的完成。

### DistOpt 的 Python 接口

然后，python 接口提供了一
个[DistOpt](https://github.com/apache/singa/blob/master/python/singa/opt.py)类来
封装一
个[optimizer](https://github.com/apache/singa/blob/master/python/singa/opt.py)对
象，以执行基于 MPI 或 Multiprocess 的分布式训练。在初始化过程中，它创建了一个
NCCL communicator 对象（来自于上面小节提到的 C 接口），然后，`DistOpt`中的每一次
all-reduce 操作都会用到这个 communicator 对象。

在 MPI 或 Multiprocess 中，每个进程都有一个独立的 rank，它给出了各个进程使用的
GPU 的信息。训练数据是被拆分的，因此每个进程可以根据一部分训练数据来评估子梯度。
一旦每个进程的子梯度被计算出来，就可以将所有进程计算出的子梯度做 all-reduce，得
到总体随机梯度。
