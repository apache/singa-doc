---
id: version-3.2.0_Viet-onnx
title: ONNX
original_id: onnx
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

[ONNX](https://onnx.ai/) là một định dạng đại diện mở dùng trong các model của
machine learning, cho phép nhà phát triển AI sử dụng các models trên các
libraries và công cụ khác nhau. SINGA hỗ trợ tải các models dạng ONNX trong
training và inference, và lưu các models ở dạng ONNX với SINGA APIs (e.g.,
[Module](./module)).

SINGA đã được thử nghiệm với
[phiên bản sau](https://github.com/onnx/onnx/blob/master/docs/Versioning.md) của
ONNX.

| Phiên bản ONNX | Phiên bản định dạng tệp tin | Opset phiên bản ai.onnx | Opset phiên bản ai.onnx.ml | Opset phiên bản ai.onnx.training |
| -------------- | --------------------------- | ----------------------- | -------------------------- | -------------------------------- |
| 1.6.0          | 6                           | 11                      | 2                          | -                                |

## Sử dụng chung

### Tải một ONNX Model trong SINGA

Sau khi tải một ONNX model từ disk qua `onnx.load`, bạn chỉ cần cập nhật
batch-size của input sử dụng `tensor.PlaceHolder` sau SINGA v3.0, shape của
internal tensors sẽ tự động được tạo ra.

Sau đó, bạn định nghĩa một class thừa hưởng từ `sonnx.SONNXModel` và thực hiện
hai phương pháp `forward` cho quá trình forward và `train_one_batch` cho quá
trình training. Sau khi gọi hàm `model.compile`, hàm SONNX sẽ lặp lại và dịch
tất cả các nodes trong phạm vi graph của ONNX model sang các hàm SINGA, tải tất
cả weights đã lưu trữ và tạo ra shape của từng tensor trung gian.

```python3
import onnx
from singa import device
from singa import sonnx

class MyModel(sonnx.SONNXModel):

    def __init__(self, onnx_model):
        super(MyModel, self).__init__(onnx_model)

    def forward(self, *x):
        y = super(MyModel, self).forward(*x)
        # Since SINGA model returns the output as a list,
        # if there is only one output,
        # you just need to take the first element.
        return y[0]

    def train_one_batch(self, x, y):
        pass

model_path = "PATH/To/ONNX/MODEL"
onnx_model = onnx.load(model_path)

# convert onnx model into SINGA model
dev = device.create_cuda_gpu()
x = tensor.PlaceHolder(INPUT.shape, device=dev)
model = MyModel(onnx_model)
model.compile([x], is_train=False, use_graph=True, sequential=True)
```

### Inference với SINGA model

Sau khi tạo models, bạn có thể tiến hành inference bằng cách gọi hàm
`model.forward`. Đầu vào và đầu ra phải ở dạng phiên bản của SINGA `Tensor`.

```python3
x = tensor.Tensor(device=dev, data=INPUT)
y = model.forward(x)
```

### Lưu model của SINGA dưới dạng ONNX

Với hàm tensors đầu vào và đầu ra được tạo ra bởi các hàm của model, bạn có thể
truy nguyên đến tất cả các hàm nội bộ. Bởi vậy, một model SINGA được xác định
bởi tensors đầu vào và đầu ra. Để biến một model SINGA sang dạng ONNX, bạn chỉ
cần cung cấp danh sách tensor đầu vào và đầu ra.

```python3
# x is the input tensor, y is the output tensor
sonnx.to_onnx([x], [y])
```

### Training lại với model ONNX

Để train (hay luyện) một model ONNX sử dụng SINGA, bạn cần thực hiện
`train_one_batch` từ `sonnx.SONNXModel` và đánh dấu `is_train=True` khi gọi hàm
`model.compile`.

```python3
from singa import opt
from singa import autograd

class MyModel(sonnx.SONNXModel):

    def __init__(self, onnx_model):
        super(MyModel, self).__init__(onnx_model)

    def forward(self, *x):
        y = super(MyModel, self).forward(*x)
        return y[0]

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = autograd.softmax_cross_entropy(out, y)
        if dist_option == 'fp32':
            self.optimizer.backward_and_update(loss)
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

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)
model.set_optimizer(sgd)
model.compile([tx], is_train=True, use_graph=graph, sequential=True)
```

### Transfer-learning một model ONNX

Bạn cũng có thể thêm một vài layers vào phần cuối của ONNX model để làm
transfer-learning. Hàm `last_layers` chấp nhận một số nguyên âm để chỉ layer bị
cắt ra. Ví dụ, `-1` nghĩa là bị cắt ra sau kết quả cuối cùng (không xoá bớt
layer nào) `-2` nghĩa là bị cắt ra sau hai layer cuối cùng.

```python3
from singa import opt
from singa import autograd

class MyModel(sonnx.SONNXModel):

    def __init__(self, onnx_model):
        super(MyModel, self).__init__(onnx_model)
        self.linear = layer.Linear(1000, 3)

    def forward(self, *x):
        # cut off after the last third layer
        # and append a linear layer
        y = super(MyModel, self).forward(*x, last_layers=-3)[0]
        y = self.linear(y)
        return y

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = autograd.softmax_cross_entropy(out, y)
        if dist_option == 'fp32':
            self.optimizer.backward_and_update(loss)
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

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)
model.set_optimizer(sgd)
model.compile([tx], is_train=True, use_graph=graph, sequential=True)
```

## ONNX model zoo

[ONNX Model Zoo](https://github.com/onnx/models) là tổ hợp các models ở dạng
ONNX, đã được train có kết quả tốt nhất, đóng góp bởi cộng đồng thành viên.
SINGA giờ đây đã hỗ trợ một số models CV và NLP. Chúng tôi dự định sẽ sớm hỗ trợ
thêm các models khác.

### Phân loại hình ảnh (Image Classification)

Tổ hợp models này có đầu vào là hình ảnh, sau đó phân loại các đối tượng chính
trong hình ảnh thành 1000 mục đối tượng như bàn phím, chuột, bút chì, và các
động vật.

| Model Class                                                                                         | Tham khảo                                               | Mô tả                                                                                                                                                                                                                                              | Đường dẫn                                                                                                                                               |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <b>[MobileNet](https://github.com/onnx/models/tree/master/vision/classification/mobilenet)</b>      | [Sandler et al.](https://arxiv.org/abs/1801.04381)      | deep neural network nhỏ, nhẹ phù hợp nhất cho điện thoại và ứng dụng hình ảnh đính kèm. <br>Top-5 error từ báo cáo - ~10%                                                                                                                          | [![Mở trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HsixqJMIpKyEPhkbB8jy7NwNEFEAUWAf) |
| <b>[ResNet18](https://github.com/onnx/models/tree/master/vision/classification/resnet)</b>          | [He et al.](https://arxiv.org/abs/1512.03385)           | Mô hình CNN (lên tới 152 layers). Sử dụng liên kết ngắn gọn để đạt độ chính xác cao hơn khi phân loại hình ảnh. <br> Top-5 error từ báo cáo - ~3.6%                                                                                                | [![Mở trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1u1RYefSsVbiP4I-5wiBKHjsT9L0FxLm9) |
| <b>[VGG16](https://github.com/onnx/models/tree/master/vision/classification/vgg)</b>                | [Simonyan et al.](https://arxiv.org/abs/1409.1556)      | Mô hình CNN chuyên sâu (lên tới 19 layers). Tương tự như AlexNet nhưng sử dụng nhiều loại filters cỡ kernel nhỏ hơn mang lại độ chính xác cao hơn khi phân loại hình ảnh. <br>Top-5 từ báo cáo - ~8%                                               | [![Mở trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14kxgRKtbjPCKKsDJVNi3AvTev81Gp_Ds) |
| <b>[ShuffleNet_V2](https://github.com/onnx/models/tree/master/vision/classification/shufflenet)</b> | [Simonyan et al.](https://arxiv.org/pdf/1707.01083.pdf) | Mô hình CNN cực kỳ hiệu quả trong sử dụng tài nguyên, được thiết kế đặc biệt cho các thiết bị di động. Mạng lưới thiết kế hệ mô hình sử dụng số liệu trực tiếp như tốc độ, thay vì các số liệu gián tiếp như FLOP. Top-1 error từ báo cáo - ~30.6% | [![Mở trên Colab](https://colab.research.google.com/drive/19HfRu3YHP_H2z3BcZujVFRp23_J5XsuA?usp=sharing)                                                |

Chúng tôi cung cấp ví dụ re-training sử dụng VGG và ResNet, vui lòng xem tại
`examples/onnx/training`.

### Nhận Diện Đối Tượng (Object Detection)

Các models Object detection nhận diện sự hiện diện của các đối tượng trong một
hình ảnh và phân đoạn ra các khu vực của bức ảnh mà đối tượng được nhận diện.

| Model Class                                                                                                       | Tham khảo                                             | Mô tả                                                                                                                                       | Đường dẫn                                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <b>[Tiny YOLOv2](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny_yolov2)</b> | [Redmon et al.](https://arxiv.org/pdf/1612.08242.pdf) | Mô hình CNN thời gian thực cho Nhận diện đối tượng có thể nhận diện 20 loại đối tượng khác nhau. Phiên bản nhỏ của mô hình phức tạp Yolov2. | [![Mở trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11V4I6cRjIJNUv5ZGsEGwqHuoQEie6b1T) |

### Phân tích Khuôn Mặt (Face Analysis)

Các mô hình Nhận Diện Khuôn Mặt xác định và/hoặc nhận diện khuôn mặt người và
các trạng thái cảm xúc trong bức ảnh.

| Model Class                                                                                               | Tham khảo                                          | Mô tả                                                                                                                                              | Đường dẫn                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <b>[ArcFace](https://github.com/onnx/models/tree/master/vision/body_analysis/arcface)</b>                 | [Deng et al.](https://arxiv.org/abs/1801.07698)    | Mô hình dựa trên CNN để nhận diện khuôn mặt, học từ các đặc tính khác nhau trên khuôn mặt và tạo ra các embeddings cho hình ảnh khuôn mặt đầu vào. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qanaqUKGIDtifdzEzJOHjEj4kYzA9uJC) |
| <b>[Emotion FerPlus](https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus)</b> | [Barsoum et al.](https://arxiv.org/abs/1608.01041) | Mô hình CNN chuyên sâu nhận diện cảm xúc được train trên các hình ảnh khuôn mặt.                                                                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XHtBQGRhe58PDi4LGYJzYueWBeWbO23r) |

### Máy Hiểu (Machine Comprehension)

Một dạng của mô hình xử lý ngôn ngữ tự nhiên giúp trả lời câu hỏi trên một đoạn
ngôn ngữ cung cấp.

| Model Class                                                                                           | Tham khảo                                                                                                                           | Mô tả                                                                                             | Đường dẫn                                                                                                                                                           |
| ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <b>[BERT-Squad](https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad)</b> | [Devlin et al.](https://arxiv.org/pdf/1810.04805.pdf)                                                                               | Mô hình này trả lời câu hỏi dựa trên ngữ cảnh của đoạn văn đầu vào.                               | [![Mở trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kud-lUPjS_u-TkDAzihBTw0Vqr0FjCE-)             |
| <b>[RoBERTa](https://github.com/onnx/models/tree/master/text/machine_comprehension/roberta)</b>       | [Devlin et al.](https://arxiv.org/pdf/1907.11692.pdf)                                                                               | Mô hình transformer-based kích thước lớn, dự đoán ngữ nghĩa dựa trên đoạn văn đầu vào.            | [![Mở trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1F-c4LJSx3Cb2jW6tP7f8nAZDigyLH6iN?usp=sharing) |
| <b>[GPT-2](https://github.com/onnx/models/tree/master/text/machine_comprehension/gpt-2)</b>           | [Devlin et al.](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Mô hình ngôn ngữ transformer-based kích thước lớn, đưa ra một đoạn chữ, rồi dự đoán từ tiếp theo. | [![Mở trên Colab](https://colab.research.google.com/drive/1ZlXLSIMppPch6HgzKRillJiUcWn3PiK7?usp=sharing)                                                            |

## Các toán tử (Operators) được hỗ trợ

Chúng tôi hỗ trợ các toán tử sau:

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

### Các lưu ý đặc biệt cho ONNX backend

- Conv, MaxPool và AveragePool

  Đầu vào phải có shape 1d`(N*C*H)` và 2d(`N*C*H*W`) trong khi `dilation` phải
  là 1.

- BatchNormalization

  `epsilon` là 1e-05 và không được đổi.

- Cast

  Chỉ hỗ trợ float32 và int32, các dạng khác phải được cast thành hai dạng này.

- Squeeze và Unsqueeze

  Nếu gặp lỗi khi dùng `Squeeze` hay `Unsqueeze` giữa `Tensor` và Scalar, vui
  lòng báo cho chúng tôi.

- Empty tensor Empty tensor không được chấp nhận trong SINGA.

## Thực hiện

Mã code của SINGA ONNX được đặt trong `python/singa/soonx.py`. Có bốn loại
chính, `SingaFrontend`, `SingaBackend`, `SingaRep` và `SONNXModel`.
`SingaFrontend` qui đổi mô hình SINGA model sang mô hình ONNX model;
`SingaBackend` biến mô hình ONNX model sang đối tượng `SingaRep` giúp lưu trữ
tất cả các toán tử SINGA operators và tensors(tensor trong văn bản này nghĩa là
SINGA `Tensor`); `SingaRep` có thẻ chạy giống như mô hình SINGA model.
`SONNXModel` tạo ra từ `model.Model` xác định thống nhất API cho SINGA.

### SingaFrontend

Hàm function đầu vào của `SingaFrontend` là `singa_to_onnx_model` cũng được gọi
là `to_onnx`. `singa_to_onnx_model` tạo ra mô hình ONNX model, và nó cũng tạo ra
một ONNX graph bằng việc sử dụng `singa_to_onnx_graph`.

`singa_to_onnx_graph` chấp nhận đầu ra của mô hình, và lặp lại đệ quy graph của
mô hình SINGA model từ đầu ra để gom tất cả toán tử tạo thành một hàng. Tensors
đầu vào và trực tiếp, v.d, weights để train, của mô hình SINGA model được chọn
cùng một lúc. Đầu vào được lưu trong `onnx_model.graph.input`; đầu ra được lưu
trong `onnx_model.graph.output`; và weights để train được lưu trong
`onnx_model.graph.initializer`.

Sau đó toán tử SINGA operator trong hàng được đổi sang từng toán tử ONNX
operators. `_rename_operators` xác định tên toán tử giữa SINGA và ONNX.
`_special_operators` xác định function sử dụng để biến đổi toán tử.

Thêm vào đó, một vài toán tử trong SINGA có các định nghĩa khác với ONNX, chẳng
hạn như, ONNX coi một vài thuộc tính của toán tử SINGA operators là đầu vào, vì
thế `_unhandled_operators` xác định function nào dùng để xử lý toán tử đặc biệt.

Do dạng bool được coi là dạng int32 trong SINGA, `_bool_operators` địng nghĩa
toán tử có thể chuyển sang dạng bool.

### SingaBackend

Function đầu vào của `SingaBackend` là `prepare` kiểm tra phiên bản nào của mô
hình ONNX model rồi gọi `_onnx_model_to_singa_ops`.

Chức năng của `_onnx_model_to_singa_ops` là để lấy SINGA tensors và operators.
Các tensors được lưu trong một thư viện, theo tên trong ONNX, và operators được
lưu trong hàng ở dạng `namedtuple('SingaOps', ['node', 'operator'])`. Với mỗi
toán tử operator, `node` là một ví dụ từ OnnxNode được dùng để lưu các thông tin
cơ bản của ONNX node; `operator` là forward function cho toán tử SINGA;

Bước đầu tiên của `_onnx_model_to_singa_ops` có bốn bước, đầu tiên là gọi
`_parse_graph_params` để lấy tất các các tensors lưu trong `params`. Sau đó gọi
hàm `_parse_graph_inputs_outputs` để lấy tất cả thông tin đầu vào đầu ra lưu
trong `inputs` và `outputs`. Cuối cùng nó lặp lại tất cả các nodes trong ONNX
graph và đẩy sang `_onnx_node_to_singa_op` như SINGA operators hoặc layers và
lưu chúng thành `outputs`. Một vài weights được lưu trong ONNX node gọi là
`Constant`, SONNX có thể xử lý chúng bằng `_onnx_constant_to_np` để lưu trong
`params`.

Cuối cùng class này trả lại một đối tượng `SingaRep` và lưu trên `params`,
`inputs`, `outputs`, `layers`.

### SingaRep

`SingaBackend` lưu tất cả SINGA tensors và operators. `run` chấp nhận đầu vào
của mô hình và chạy từng toán tử SINGA operators một, theo hàng của toán tử.
Người dùng có thể sử dụng `last_layers` để xoá mô hình model sau vài layers cuối
cùng.

### SONNXModel

`SONNXModel` được tạo từ `sonnx.SONNXModel` và thực hiện phương pháp `forward`
để cung cấp một API đồng bộ với các mô hình SINGA.
