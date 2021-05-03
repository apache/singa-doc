---
id: autograd
title: Autograd
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Có hai cách thường dùng để sử dụng autograd, qua symbolic differentiation như là [Theano](http://deeplearning.net/software/theano/index.html) hoặc reverse
differentiation như là
[Pytorch](https://pytorch.org/docs/stable/notes/autograd.html). SINGA dùng cách Pytorch, lưu trữ computation graph rồi áp dụng backward
propagation tự động sau khi forward propagation. Thuật toán autograd được giải thích cụ thể ở
 [đây](https://pytorch.org/docs/stable/notes/autograd.html). Chúng tôi giải thích các modules liên quan trong Singa và đưa ra ví dụ để minh hoạ cách sử dụng.

## Các Module liên quan

Autograd gồm ba classes với tên gọi `singa.tensor.Tensor`,
`singa.autograd.Operation`, và `singa.autograd.Layer`. Trong phần tiếp theo của văn kiện này, chúng tôi dùng tensor, operation và layer để chỉ một chương trình (instance) trong class tương ứng. 

### Tensor

Ba tính năng của Tensor được sử dụng bởi autograd,

- `.creator` là một chương trình `Operation`. Chương trình này lưu trữ tác vụ tạo ra Tensor instance.
- `.requires_grad` là một biến kiểu bool. Biến được sử dụng để chỉ rằng thuật toán autograd cần tính ra độ dốc (gradient) của tensor. (như owner). Ví dụ, khi chạy backpropagation, thì cần phải tính ra độ dốc của tensor cho ma trận trọng lượng (weight matrix) của lớp tuyến tính (linear layer) và bản đồ tính năng (feature map) của convolution
  layer (không phải lớp cuối).
- `.stores_grad` là một biến kiểu bool. Biến được sử dụng để chỉ rằng độ dốc của owner tensor cần được lưu và tạo ra bởi hàm backward. Ví dụ, độ dốc của feature maps được tính thông qua backpropagation, nhưng không được bao gồm trong kết quả của hàm backward.

Lập trình viên có thể thay đổi `requires_grad` và `stores_grad` của chương trình Tensor. Ví dụ nếu hàm sau để là True, độ dốc tương ứng sẽ được bao gồm trong kết quả của hàm backward. Cần lưu ý rằng nếu `stores_grad` để là True, thì `requires_grad` cũng phải là True, và ngược lại. 

### Operation

Hàm chạy một hoặc một vài chương trình `Tensor` instances ở đầu vào, sau đó đầu ra là một hoặc một vài chương trình `Tensor` instances. Ví dụ, hàm ReLU có thể được sử dụng như một subclass của một hàm Operation cụ thể. Khi gọi một chương trình `Operation` (sau cài đặt), cần thực hiện hai bước sau: 

1. Ghi lại hàm operations nguồn, vd. biến `creator`của tensor đầu vào.
2. làm tính toán bằng cách gọi hàm thành viên `.forward()`

Có hai hàm thành viên cho forwarding và backwarding, vd.
`.forward()` và `.backward()`. Đầu vào là `Tensor.data` (thuộc loại
`CTensor`), và đầu ra là `Ctensor`. Nếu muốn thêm một hàm operation thì subclass `operation` cần chạy riêng `.forward()` và `.backward()`. Hàm 
`backward()` được tự động gọi bởi hàm `backward()` của autograd trong quá trình chạy backward để thực hiện độ dốc của đầu vào
(theo mục `require_grad`).

### Layer

Với các hàm yêu cầu tham số (parameter), chúng tôi gói chúng lại thành một class mới,
`Layer`. Ví dụ hàm convolution operation thì được nhóm vào trong convolution layer.
`Layer` quản lý (hoặc lưu trữ) các tham số và sẽ gọi các hàm `Operation` tương ứng để thực hiện việc chuyển đổi.

## Ví dụ

Chúng tôi cung cấp nhiều ví dụ trong 
[mục ví dụ](https://github.com/apache/singa/tree/master/examples/autograd).
Chúng tôi đưa ra giải thích cụ thể trong hai ví dụ tiêu biểu ở đây. 

### Dùng hàm Operation

Code dưới đây áp dụng model MLP, chỉ dùng hàm Operation (không dùng hàm Layer).

#### Thêm packages

```python
from singa.tensor import Tensor
from singa import autograd
from singa import opt
```

#### Tạo ma trận trọng lượng (weight matrix) và bias vector

Tham số tensors được tạo bởi cả `requires_grad` và `stores_grad`
ở giá trị `True`.

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

#### Training

```python
inputs = Tensor(data=data)  # data matrix
target = Tensor(data=label) # label vector
autograd.training = True    # cho training
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

### Hàm Operation + Layer

[Ví dụ](https://github.com/apache/singa/blob/master/examples/autograd/mnist_cnn.py) sau đây áp dụng CNN model sử dụng các lớp (layers) tạo từ autograd module.

#### Tạo layers

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

#### Định nghĩa hàm forward

Hàm trong forward pass sẽ được tự đông lưu cho backward propagation.

```python
def forward(x, t):
    # x là input data (batch hình ảnh)
    # t là label vector (batch số nguyên)
    y = conv1(x)           # Conv layer
    y = autograd.relu(y)   # ReLU operation
    y = bn1(y)             # BN layer
    y = pooling1(y)        # Pooling Layer

    # hai convolution layers song song
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

#### Training

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

### Sử dụng Model API

[Ví dụ](https://github.com/apache/singa/blob/master/examples/cnn/model/cnn.py) sau áp dụng CNN model sử dụng [Model API](./graph).

#### Định nghiã subclass của Model

Model class được định nghĩa là subclass của Model. Theo đó, tất cả các hàm operations được sử dụng trong bước training sẽ tạo thành một computational graph và được phân tích. Hàm operation trong graph sẽ được lên lịch trình và thực hiện hiệu quả. Layers cũng có thể được bao gồm trong model class.

```python
class MLP(model.Model):  # model là subclass của Model

    def __init__(self, data_size=10, perceptron_size=100, num_classes=10):
        super(MLP, self).__init__()

        # taọ operators, layers và các object khác
        self.relu = layer.ReLU()
        self.linear1 = layer.Linear(perceptron_size)
        self.linear2 = layer.Linear(num_classes)
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()

    def forward(self, inputs):  # định nghĩa forward function
        y = self.linear1(inputs)
        y = self.relu(y)
        y = self.linear2(y)
        return y

    def train_one_batch(self, x, y):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)
        self.optimizer(loss)
        return out, loss

    def set_optimizer(self, optimizer):  # đính kèm optimizer
        self.optimizer = optimizer
```

#### Training

```python
# tạo hàm model instance
model = MLP()
# tạo optimizer và đính vào model
sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)
model.set_optimizer(sgd)
# input và target placeholders cho model
tx = tensor.Tensor((batch_size, 1, IMG_SIZE, IMG_SIZE), dev, tensor.float32)
ty = tensor.Tensor((batch_size, num_classes), dev, tensor.int32)
# tổng hợp model trước khi training
model.compile([tx], is_train=True, use_graph=True, sequential=False)

# train model theo bước lặp (iterative)
for b in range(num_train_batch):
    # generate the next mini-batch
    x, y = ...

    # Copy the data into input tensors
    tx.copy_from_numpy(x)
    ty.copy_from_numpy(y)

    # Training with one batch
    out, loss = model(tx, ty)
```

#### Lưu model checkpoint

```python
# xác định đường dẫn để lưu checkpoint
checkpointpath="checkpoint.zip"

# lưu checkpoint
model.save_states(fpath=checkpointpath)
```

#### Tải model checkpoint

```python
# xác định đường dẫn để lưu checkpoint
checkpointpath="checkpoint.zip"

# lưu checkpoint
import os
if os.path.exists(checkpointpath):
    model.load_states(fpath=checkpointpath)
```

### Python API

Xem
[tại đây](https://singa.readthedocs.io/en/latest/autograd.html#module-singa.autograd)
để thêm thông tin chi tiết về Python API.
