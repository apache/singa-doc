---
id: version-2.0.0-onnx
title: ONNX
original_id: onnx
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

ONNX is an open format built to represent machine learning models, which enables an ability to transfer trained models between different deep learning frameworks. We have integrated the main functionality of ONNX into Singa, and several basic operators have been supported. More operators are being developing.

## Example: ONNX mnist on singa

We will introduce the onnx of singa by using the mnist example. In this section, the examples of how to export, load, inference, re-training, and transfer-learning the minist model will be displayed.

### Load dataset

Firstly, we import some necessary libraries and define some auxiliary functions for downloading and preprocessing the dataset:

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

### MNIST model

We define a class called **CNN** to construct the mnist model which consists of several convolution, pooling, fully connection and relu layers. We also define a function to calculate the **accuracy** of our result. Finally, we define a **train** and a **test** function to handle the training and prediction process.

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

### Train mnist model and export it to onnx

Now, we can train the mnist model and export its onnx model by calling the **soonx.to_onnx** function.

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

After we export the onnx model, we can find a file called **mnist.onnx** in the '/tmp' directory, this model, therefore, can be imported by other libraries. Now, if we want to import this onnx model into singa again and do the inference using the validation dataset, we can define a class called **Infer**, the forward function of Infer will be called by the test function to do inference for validation dataset. By the way, we should set the label of training to **False** to fix the gradient of autograd operators.

When import the onnx model, we firstly call **onnx.load** to load the onnx model. Then the onnx model will be fed into the **soonx.prepare** to parse and initiate to a singa model(**sg_ir** in the code). The sg_ir contains a singa graph within it, and we can run an step of inference by feeding input to its run function.

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

### Re-training

Assume after import the model, we want to re-train the model again, we can define a function called **re_train**. Before we call this re_train function, we should set the label of training to **True** to make the autograde operators update their gradient. And after we finish the training, we set it as **False** again to call the test function doing inference.

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

### Transfer learning

Finally, if we want to do transfer-learning, we can define a function called **Trans** to append some layers after the onnx model. For demonstration, we only append several linear(fully connection) and relu after the onnx model. We also define a transfer_learning function to handle the training process of the transfer-learning model. And the label of training is the same as the previous one.

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

## Supported operators

The following operators are supported:

- Conv
- Relu
- Constant
- MaxPool
- AveragePool
- Softmax
- Sigmoid
- Add
- MatMul
- BatchNormalization
- Concat
- Flatten
- Add
- Gemm
- Reshape
- Sum
- Cos
- Cosh
- Sin
- Sinh
- Tan
- Tanh
- Acos
- Acosh
- Asin
- Asinh
- Atan
- Atanh
- Selu
- Elu
- Equal
- Less
- Sign
- Div
- Sub
- Sqrt
- Log
- Greater
- HardSigmoid
- Identity
- Softplus
- Softsign
- Mean
- Pow
- Clip
- PRelu
- Mul
- Transpose
- Max
- Min
- Shape
- And
- Or
- Xor
- Not
- Neg
- Reciprocal
