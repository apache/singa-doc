---
id: onnx
title: ONNX
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

[ONNX](https://onnx.ai/) 是机器学习模型的开放表示格式，它使AI开发人员能够在不同的库和工具中使用模型。SINGA支持加载ONNX格式模型用于训练和inference，并将使用SINGA API（如[Module](./module)）定义的模型保存为ONNX格式。

SINGA在以下[版本](https://github.com/onnx/onnx/blob/master/docs/Versioning.md)中的ONNX中测试过。

| ONNX version | File format version | Opset version ai.onnx | Opset version ai.onnx.ml | Opset version ai.onnx.training |
| ------------ | ------------------- | --------------------- | ------------------------ | ------------------------------ |
| 1.6.0        | 6                   | 11                    | 2                        | -                              |

## 通常用法

### 从ONNX中读取一个Model到SINGA

在通过 `onnx.load` 从磁盘加载 ONNX 模型后，您需要更新模型的batch_size，因为对于大多数模型来说，它们使用一个占位符来表示其批处理量。我们在这里举一个例子，若要 `update_batch_size`，你只需要更新输入和输出的 batch_size，内部的 tensors 的形状会自动推断出来。


然后，您可以使用 `sonnx.prepare` 来准备 SINGA 模型。该函数将 ONNX 模型图中的所有节点迭代并翻译成 SINGA 运算符，加载所有存储的权重并推断每个中间张量的形状。

```python3
import onnx
from singa import device
from singa import sonnx

# if the input has multiple tensors? can put this function inside prepare()?
def update_batch_size(onnx_model, batch_size):
    model_input = onnx_model.graph.input[0]
    model_input.type.tensor_type.shape.dim[0].dim_value = batch_size
    model_output = onnx_model.graph.output[0]
    model_output.type.tensor_type.shape.dim[0].dim_value = batch_size
    return onnx_model


model_path = "PATH/To/ONNX/MODEL"
onnx_model = onnx.load(model_path)

# set batch size
onnx_model = update_batch_size(onnx_model, 1)

# convert onnx graph nodes into SINGA operators
dev = device.create_cuda_gpu()
sg_ir = sonnx.prepare(onnx_model, device=dev)
```

### Inference SINGA模型

一旦创建了模型，就可以通过调用`sg_ir.run`进行inference。输入和输出必须是SINGA Tensor实例，由于SINGA模型以列表形式返回输出，如果只有一个输出，你只需要从输出中取第一个元素即可。

```python3
# can warp the following code in prepare()
# and provide a flag training=True/False?

class Infer:


    def __init__(self, sg_ir):
        self.sg_ir = sg_ir

    def forward(self, x):
        return sg_ir.run([x])[0]


data = get_dataset()
x = tensor.Tensor(device=dev, data=data)

model = Infer(sg_ir)
y = model.forward(x)
```

### 将SINGA模型保存成ONNX格式

给定输入时序和输出时序，由运算符产生的模型，你可以追溯所有内部操作。因此，一个SINGA模型是由输入和输出张量定义的，要将 SINGA 模型导出为 ONNX 格式，您只需提供输入和输出张量列表。

```python3
# x is the input tensor, y is the output tensor
sonnx.to_onnx([x], [y])
```

### 在ONNX模型上重新训练

要使用 SINGA 训练（或改进）ONNX 模型，您需要设置内部的张量为可训练状态：

```python3
class Infer:

    def __init__(self, sg_ir):
        self.sg_ir = sg_ir
        ## can wrap these codes in sonnx?
        for idx, tens in sg_ir.tensor_map.items():
            # allow the tensors to be updated
            tens.requires_grad = True
            tens.stores_grad = True

    def forward(self, x):
        return sg_ir.run([x])[0]

autograd.training = False
model = Infer(sg_ir)

autograd.training = True
# then you training the model like normal
# give more details??
```

### 在ONNX模型上做迁移学习

您也可以在ONNX模型的最后附加一些图层来进行转移学习。`last_layers` 意味着您从 [0, last_layers] 切断 ONNX 层。然后您可以通过普通的SINGA模型附加更多的层。

```python3
class Trans:

    def __init__(self, sg_ir, last_layers):
        self.sg_ir = sg_ir
        self.last_layers = last_layers
        self.append_linear1 = autograd.Linear(500, 128, bias=False)
        self.append_linear2 = autograd.Linear(128, 32, bias=False)
        self.append_linear3 = autograd.Linear(32, 10, bias=False)

    def forward(self, x):
        y = sg_ir.run([x], last_layers=self.last_layers)[0]
        y = self.append_linear1(y)
        y = autograd.relu(y)
        y = self.append_linear2(y)
        y = autograd.relu(y)
        y = self.append_linear3(y)
        y = autograd.relu(y)
        return y

autograd.training = False
model = Trans(sg_ir, -1)

# then you training the model like normal
```

## 一个完整示例

本部分以mnist为例，介绍SINGA ONNX的使用方法。在这部分，将展示如何导出、加载、inference、再训练和迁移学习 mnist 模型的例子。您可以在[这里](https://colab.research.google.com/drive/1-YOfQqqw3HNhS8WpB8xjDQYutRdUdmCq)试用这部分内容。

### 读取数据集

首先，你需要导入一些必要的库，并定义一些辅助函数来下载和预处理数据集：

```python
import os
import urllib.request
import gzip
import numpy as np
import codecs

from singa import device
from singa import tensor
from singa import opt
from singa import autograd
from singa import sonnx
import onnx


def load_dataset():
    train_x_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    train_y_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    valid_x_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    valid_y_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    train_x = read_image_file(check_exist_or_download(train_x_url)).astype(
        np.float32)
    train_y = read_label_file(check_exist_or_download(train_y_url)).astype(
        np.float32)
    valid_x = read_image_file(check_exist_or_download(valid_x_url)).astype(
        np.float32)
    valid_y = read_label_file(check_exist_or_download(valid_y_url)).astype(
        np.float32)
    return train_x, train_y, valid_x, valid_y


def check_exist_or_download(url):

    download_dir = '/tmp/'

    name = url.rsplit('/', 1)[-1]
    filename = os.path.join(download_dir, name)
    if not os.path.isfile(filename):
        print("Downloading %s" % url)
        urllib.request.urlretrieve(url, filename)
    return filename


def read_label_file(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(
            (length))
        return parsed


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_image_file(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(
            (length, 1, num_rows, num_cols))
        return parsed


def to_categorical(y, num_classes):
    y = np.array(y, dtype="int")
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    categorical = categorical.astype(np.float32)
    return categorical
```

### MNIST模型

然后你可以定义一个叫做**CNN**的类来构造mnist模型，这个模型由几个卷积层、池化层、全连接层和relu层组成。你也可以定义一个函数来计算我们结果的**准确性**。最后，你可以定义一个**训练函数**和一个**测试函数**来处理训练和预测的过程。

```python
class CNN:
    def __init__(self):
        self.conv1 = autograd.Conv2d(1, 20, 5, padding=0)
        self.conv2 = autograd.Conv2d(20, 50, 5, padding=0)
        self.linear1 = autograd.Linear(4 * 4 * 50, 500, bias=False)
        self.linear2 = autograd.Linear(500, 10, bias=False)
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


def accuracy(pred, target):
    y = np.argmax(pred, axis=1)
    t = np.argmax(target, axis=1)
    a = y == t
    return np.array(a, "int").sum() / float(len(t))


def train(model,
          x,
          y,
          epochs=1,
          batch_size=64,
          dev=device.get_default_device()):
    batch_number = x.shape[0] // batch_size

    for i in range(epochs):
        for b in range(batch_number):
            l_idx = b * batch_size
            r_idx = (b + 1) * batch_size

            x_batch = tensor.Tensor(device=dev, data=x[l_idx:r_idx])
            target_batch = tensor.Tensor(device=dev, data=y[l_idx:r_idx])

            output_batch = model.forward(x_batch)
            # onnx_model = sonnx.to_onnx([x_batch], [y])
            # print('The model is:\n{}'.format(onnx_model))

            loss = autograd.softmax_cross_entropy(output_batch, target_batch)
            accuracy_rate = accuracy(tensor.to_numpy(output_batch),
                                     tensor.to_numpy(target_batch))

            sgd = opt.SGD(lr=0.001)
            for p, gp in autograd.backward(loss):
                sgd.update(p, gp)
            sgd.step()

            if b % 1e2 == 0:
                print("acc %6.2f loss, %6.2f" %
                      (accuracy_rate, tensor.to_numpy(loss)[0]))
    print("training completed")
    return x_batch, output_batch

def test(model, x, y, batch_size=64, dev=device.get_default_device()):
    batch_number = x.shape[0] // batch_size

    result = 0
    for b in range(batch_number):
        l_idx = b * batch_size
        r_idx = (b + 1) * batch_size

        x_batch = tensor.Tensor(device=dev, data=x[l_idx:r_idx])
        target_batch = tensor.Tensor(device=dev, data=y[l_idx:r_idx])

        output_batch = model.forward(x_batch)
        result += accuracy(tensor.to_numpy(output_batch),
                           tensor.to_numpy(target_batch))

    print("testing acc %6.2f" % (result / batch_number))
```

### 训练mnist模型并将其导出到onnx

现在，你可以通过调用 **soonx.to_onnx** 函数来训练 mnist 模型并导出其 onnx 模型。

```python
def make_onnx(x, y):
    return sonnx.to_onnx([x], [y])

# create device
dev = device.create_cuda_gpu()
#dev = device.get_default_device()
# create model
model = CNN()
# load data
train_x, train_y, valid_x, valid_y = load_dataset()
# normalization
train_x = train_x / 255
valid_x = valid_x / 255
train_y = to_categorical(train_y, 10)
valid_y = to_categorical(valid_y, 10)
# do training
autograd.training = True
x, y = train(model, train_x, train_y, dev=dev)
onnx_model = make_onnx(x, y)
# print('The model is:\n{}'.format(onnx_model))

# Save the ONNX model
model_path = os.path.join('/', 'tmp', 'mnist.onnx')
onnx.save(onnx_model, model_path)
print('The model is saved.')
```

### Inference

导出onnx模型后，可以在'/tmp'目录下找到一个名为**mnist.onnx**的文件，这个模型可以被其他库导入。现在，如果你想把这个onnx模型再次导入到singa中，并使用验证数据集进行推理，你可以定义一个叫做**Infer**的类，Infer的前向函数将被测试函数调用，对验证数据集进行推理。此外，你应该把训练的标签设置为**False**，以固定自变量算子的梯度。

在导入onnx模型时，需要先调用**onnx.load**来加载onnx模型。然后将onnx模型输入到 **soonx.prepare**中进行解析，并启动到一个singa模型(代码中的**sg_ir**)。sg_ir里面包含了一个singa图，然后就可以通过输入到它的run函数中运行一步推理。

```python
class Infer:
    def __init__(self, sg_ir):
        self.sg_ir = sg_ir
        for idx, tens in sg_ir.tensor_map.items():
            # allow the tensors to be updated
            tens.requires_grad = True
            tens.stores_grad= True
            sg_ir.tensor_map[idx] = tens

    def forward(self, x):
        return sg_ir.run([x])[0] # we can run one step of inference by feeding input

# load the ONNX model
onnx_model = onnx.load(model_path)
sg_ir = sonnx.prepare(onnx_model, device=dev) # parse and initiate to a singa model

# inference
autograd.training = False
print('The inference result is:')
test(Infer(sg_ir), valid_x, valid_y, dev=dev)
```

### 重训练

假设导入模型后，想再次对模型进行重新训练，我们可以定义一个名为**re_train**的函数。在调用这个re_train函数之前，我们应该将训练的标签设置为**True**，以使自变量运算符更新其梯度。而在完成训练后，我们再将其设置为**False**，以调用做推理的测试函数。

```python
def re_train(sg_ir,
             x,
             y,
             epochs=1,
             batch_size=64,
             dev=device.get_default_device()):
    batch_number = x.shape[0] // batch_size

    new_model = Infer(sg_ir)

    for i in range(epochs):
        for b in range(batch_number):
            l_idx = b * batch_size
            r_idx = (b + 1) * batch_size

            x_batch = tensor.Tensor(device=dev, data=x[l_idx:r_idx])
            target_batch = tensor.Tensor(device=dev, data=y[l_idx:r_idx])

            output_batch = new_model.forward(x_batch)

            loss = autograd.softmax_cross_entropy(output_batch, target_batch)
            accuracy_rate = accuracy(tensor.to_numpy(output_batch),
                                     tensor.to_numpy(target_batch))

            sgd = opt.SGD(lr=0.01)
            for p, gp in autograd.backward(loss):
                sgd.update(p, gp)
            sgd.step()

            if b % 1e2 == 0:
                print("acc %6.2f loss, %6.2f" %
                      (accuracy_rate, tensor.to_numpy(loss)[0]))
    print("re-training completed")
    return new_model

# load the ONNX model
onnx_model = onnx.load(model_path)
sg_ir = sonnx.prepare(onnx_model, device=dev)

# re-training
autograd.training = True
new_model = re_train(sg_ir, train_x, train_y, dev=dev)
autograd.training = False
test(new_model, valid_x, valid_y, dev=dev)
```

### 迁移学习

最后，如果我们想做迁移学习，我们可以定义一个名为**Trans**的函数，在onnx模型后追加一些层。为了演示，代码中只在onnx模型后追加了几个线性（全连接）和relu。可以定义一个transfer_learning函数来处理transfer-learning模型的训练过程，而训练的标签和前面一个的一样。

```python
class Trans:
    def __init__(self, sg_ir, last_layers):
        self.sg_ir = sg_ir
        self.last_layers = last_layers
        self.append_linear1 = autograd.Linear(500, 128, bias=False)
        self.append_linear2 = autograd.Linear(128, 32, bias=False)
        self.append_linear3 = autograd.Linear(32, 10, bias=False)

    def forward(self, x):
        y = sg_ir.run([x], last_layers=self.last_layers)[0]
        y = self.append_linear1(y)
        y = autograd.relu(y)
        y = self.append_linear2(y)
        y = autograd.relu(y)
        y = self.append_linear3(y)
        y = autograd.relu(y)
        return y

def transfer_learning(sg_ir,
             x,
             y,
             epochs=1,
             batch_size=64,
             dev=device.get_default_device()):
    batch_number = x.shape[0] // batch_size

    trans_model = Trans(sg_ir, -1)

    for i in range(epochs):
        for b in range(batch_number):
            l_idx = b * batch_size
            r_idx = (b + 1) * batch_size

            x_batch = tensor.Tensor(device=dev, data=x[l_idx:r_idx])
            target_batch = tensor.Tensor(device=dev, data=y[l_idx:r_idx])
            output_batch = trans_model.forward(x_batch)

            loss = autograd.softmax_cross_entropy(output_batch, target_batch)
            accuracy_rate = accuracy(tensor.to_numpy(output_batch),
                                     tensor.to_numpy(target_batch))

            sgd = opt.SGD(lr=0.07)
            for p, gp in autograd.backward(loss):
                sgd.update(p, gp)
            sgd.step()

            if b % 1e2 == 0:
                print("acc %6.2f loss, %6.2f" %
                      (accuracy_rate, tensor.to_numpy(loss)[0]))
    print("transfer-learning completed")
    return trans_mode

# load the ONNX model
onnx_model = onnx.load(model_path)
sg_ir = sonnx.prepare(onnx_model, device=dev)

# transfer-learning
autograd.training = True
new_model = transfer_learning(sg_ir, train_x, train_y, dev=dev)
autograd.training = False
test(new_model, valid_x, valid_y, dev=dev)
```

## ONNX模型库

[ONNX 模型库](https://github.com/onnx/models)是由社区成员贡献的 ONNX 格式的预先训练的最先进模型的集合。SINGA 现在已经支持了几个 CV 和 NLP 模型。将来会支持更多模型。

### 图像分类

这套模型以图像作为输入，然后将图像中的主要物体分为1000个物体类别，如键盘、鼠标、铅笔和许多动物。

| Model Class                                                                                         | Reference                                               | Description                                                                                                                                                                                                                               | Link                                                                                                                                                    |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <b>[MobileNet](https://github.com/onnx/models/tree/master/vision/classification/mobilenet)</b>      | [Sandler et al.](https://arxiv.org/abs/1801.04381)      | 最适合移动和嵌入式视觉应用的轻量级深度神经网络。 <br>Top-5 error from paper - ~10%                                                                                                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HsixqJMIpKyEPhkbB8jy7NwNEFEAUWAf) |
| <b>[ResNet18](https://github.com/onnx/models/tree/master/vision/classification/resnet)</b>          | [He et al.](https://arxiv.org/abs/1512.03385)           | 一个CNN模型（多达152层），在对图像进行分类时，使用shortcut来实现更高的准确性。 <br> Top-5 error from paper - ~3.6%                                                                                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1u1RYefSsVbiP4I-5wiBKHjsT9L0FxLm9) |
| <b>[VGG16](https://github.com/onnx/models/tree/master/vision/classification/vgg)</b>                | [Simonyan et al.](https://arxiv.org/abs/1409.1556)      | 深度CNN模型（多达19层）。类似于AlexNet，但使用多个较小的内核大小的滤波器，在分类图像时提供更高的准确性。 <br>Top-5 error from paper - ~8%                                                  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14kxgRKtbjPCKKsDJVNi3AvTev81Gp_Ds) |
| <b>[ShuffleNet_V2](https://github.com/onnx/models/tree/master/vision/classification/shufflenet)</b> | [Simonyan et al.](https://arxiv.org/pdf/1707.01083.pdf) | 专门为移动设备设计的计算效率极高的CNN模型。这种网络架构设计考虑了速度等直接指标，而不是FLOP等间接指标。 Top-1 error from paper - ~30.6% | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19HfRu3YHP_H2z3BcZujVFRp23_J5XsuA?usp=sharing)                                                |

### 目标检测

目标检测模型可以检测图像中是否存在多个对象，并将图像中检测到对象的区域分割出来。

| Model Class                                                                                                       | Reference                                             | Description                                                                                                                        | Link                                                                                                                                                    |
| ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <b>[Tiny YOLOv2](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny_yolov2)</b> | [Redmon et al.](https://arxiv.org/pdf/1612.08242.pdf) | 一个用于目标检测的实时CNN，可以检测20个不同的类。一个更复杂的完整YOLOv2网络的小版本。 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11V4I6cRjIJNUv5ZGsEGwqHuoQEie6b1T) |

### 面部识别

人脸检测模型可以识别和/或识别给定图像中的人脸和情绪。

| Model Class                                                                                               | Reference                                          | Description                                                                                                                         | Link                                                                                                                                                    |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <b>[ArcFace](https://github.com/onnx/models/tree/master/vision/body_analysis/arcface)</b>                 | [Deng et al.](https://arxiv.org/abs/1801.07698)    | 一种基于CNN的人脸识别模型，它可以学习人脸的判别特征，并对输入的人脸图像进行分析。 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qanaqUKGIDtifdzEzJOHjEj4kYzA9uJC) |
| <b>[Emotion FerPlus](https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus)</b> | [Barsoum et al.](https://arxiv.org/abs/1608.01041) | 基于人脸图像训练的情感识别深度CNN。                                                                        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XHtBQGRhe58PDi4LGYJzYueWBeWbO23r) |

### 机器理解

这个自然语言处理模型的子集，可以回答关于给定上下文段落的问题。

| Model Class                                                                                           | Reference                                                                                                                           | Description                                                                                                       | Link                                                                                                                                                                |
| ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <b>[BERT-Squad](https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad)</b> | [Devlin et al.](https://arxiv.org/pdf/1810.04805.pdf)                                                                               | 该模型根据给定输入段落的上下文回答问题。                                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kud-lUPjS_u-TkDAzihBTw0Vqr0FjCE-)             |
| <b>[RoBERTa](https://github.com/onnx/models/tree/master/text/machine_comprehension/roberta)</b>       | [Devlin et al.](https://arxiv.org/pdf/1907.11692.pdf)                                                                               | 一个基于大型变换器的模型，根据给定的输入文本预测情感。                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1F-c4LJSx3Cb2jW6tP7f8nAZDigyLH6iN?usp=sharing) |
| <b>[GPT-2](https://github.com/onnx/models/tree/master/text/machine_comprehension/gpt-2)</b>           | [Devlin et al.](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 一个基于大型变换器的语言模型，给定一些文本中的单词序列，预测下一个单词。 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZlXLSIMppPch6HgzKRillJiUcWn3PiK7?usp=sharing)                                                            |

## 支持的操作符

onnx支持下列运算:

- Acos
- Acosh
- Add
- And
- Asin
- Asinh
- Atan
- Atanh
- AveragePool
- BatchNormalization
- Cast
- Ceil
- Clip
- Concat
- ConstantOfShape
- Conv
- Cos
- Cosh
- Div
- Dropout
- Elu
- Equal
- Erf
- Expand
- Flatten
- Gather
- Gemm
- GlobalAveragePool
- Greater
- HardSigmoid
- Identity
- LeakyRelu
- Less
- Log
- MatMul
- Max
- MaxPool
- Mean
- Min
- Mul
- Neg
- NonZero
- Not
- OneHot
- Or
- Pad
- Pow
- PRelu
- Reciprocal
- ReduceMean
- ReduceSum
- Relu
- Reshape
- ScatterElements
- Selu
- Shape
- Sigmoid
- Sign
- Sin
- Sinh
- Slice
- Softmax
- Softplus
- Softsign
- Split
- Sqrt
- Squeeze
- Sub
- Sum
- Tan
- Tanh
- Tile
- Transpose
- Unsqueeze
- Upsample
- Where
- Xor

### 对ONNX后端的特别说明

- Conv, MaxPool 以及 AveragePool

  输入必须是1d`(N*C*H)`和2d`(N*C*H*W)`的形状，`dilation`必须是1。

- BatchNormalization

  `epsilon` 设定为1e-05，不能改变

- Cast

  只支持float32和int32，其他类型都会转向这两种类型。

- Squeeze and Unsqueeze

  如果你在`Tensor`和Scalar之间`Squeeze`或`Unsqueeze`时遇到错误，请向我们报告。

- Empty tensor 

  空张量在SINGA是非法的。

## 实现

SINGA ONNX的代码在`python/singa/soonx.py`中，主要有三个类，`SingaFrontend`、`SingaBackend`和`SingaRep`。`SingaFrontend`将SINGA模型翻译成ONNX模型；`SingaBackend`将ONNX模型翻译成`SingaRep`对象，其中存储了所有的SINGA运算符和张量（本文档中的张量指SINGA Tensor）；`SingaRep`可以像SINGA模型一样运行。

### SingaFrontend

`SingaFrontend`的入口函数是`singa_to_onnx_model`，它也被称为`to_onnx`，`singa_to_onnx_model`创建了ONNX模型，它还通过`singa_to_onnx_graph`创建了一个ONNX图。


`singa_to_onnx_graph`接受模型的输出，并从输出中递归迭代SINGA模型的图，得到所有的运算符，形成一个队列。SINGA模型的输入和中间张量，即可训练的权重，同时被获取。输入存储在`onnx_model.graph.input`中；输出存储在`onnx_model.graph.output`中；可训练权重存储在`onnx_model.graph.initializer`中。

然后将队列中的SINGA运算符逐一翻译成ONNX运算符。`_rename_operators` 定义了 SINGA 和 ONNX 之间的运算符名称映射。`_special_operators` 定义了翻译运算符时要使用的函数。

此外，SINGA 中的某些运算符与 ONNX 的定义不同，即 ONNX 将 SINGA 运算符的某些属性视为输入，因此 `_unhandled_operators` 定义了处理特殊运算符的函数。

由于SINGA中的布尔类型被视为int32，所以`_bool_operators`定义了要改变的操作符为布尔类型。

### SingaBackend

`SingaBackend`的入口函数是`prepare`，它检查ONNX模型的版本，然后调用`_onnx_model_to_singa_net`。

`_onnx_model_to_singa_net`的目的是获取SINGA的时序和运算符。tensors在ONNX中以其名称存储在字典中，而操作符则以`namedtuple('SingaOps', ['name', 'op', 'handle', 'forward'])`的形式存储在队列中。对于每个运算符，`name`是它的ONNX节点名称；`op`是ONNX节点；`forward`是SINGA运算符的转发函数；`handle`是为一些特殊的运算符准备的，如Conv和Pooling，它们有`handle`对象。

`_onnx_model_to_singa_net`的第一步是调用`_init_graph_parameter`来获取模型内的所有tensors。对于可训练的权重，可以从`onnx_model.graph.initializer`中初始化`SINGA Tensor`。请注意，权重也可能存储在图的输入或称为`Constant`的ONNX节点中，SINGA也可以处理这些。

虽然所有的权重都存储在ONNX模型中，但模型的输入是未知的，只有它的形状和类型。所以SINGA支持两种方式来初始化输入，1、根据其形状和类型生成随机张量，2、允许用户分配输入。第一种方法对大多数模型都很好，但是对于一些模型，比如BERT，矩阵的指数不能随机生成，否则会产生错误。

然后，`_onnx_model_to_singa_net`迭代ONNX图中的所有节点，将其翻译成SIGNA运算符。另外，`_rename_operators` 定义了 SINGA 和 ONNX 之间的运算符名称映射。`_special_operators` 定义翻译运算符时要使用的函数。`_run_node`通过输入时序来运行生成的 SINGA 模型，并存储其输出时序，供以后的运算符使用。

该类最后返回一个`SingaRep`对象，并在其中存储所有SINGA时序和运算符。

### SingaRep

`SingaBackend`存储所有的SINGA tensors和运算符。`run`接受模型的输入，并按照运算符队列逐个运行SINGA运算符。用户可以使用`last_layers`来决定是否将模型运行到最后几层。将 `all_outputs` 设置为 `False` 表示只得到最后的输出，设置为 `True` 表示也得到所有的中间输出。
