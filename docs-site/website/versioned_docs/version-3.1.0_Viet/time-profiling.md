---
id: version-3.1.0_Viet-time-profiling
title: Time Profiling
original_id: time-profiling
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA hỗ trợ hồ sơ thời gian (time profilier) của mỗi toán tử được lưu tạm thời
trong graph. Để tận dụng chức năng hồ sơ thời gian, chúng tôi trước tiên gọi
method `device.SetVerbosity` để tạo độ dài cho hồ sơ thời gian, sau đó gọi hàm
`device.PrintTimeProfiling` để in ra kết quả của hồ sơ thời gian.

## Thiết lập Độ dài cho Hồ sơ thời gian

Để sử dụng chức năng hồ sơ thời gian, bạn cần tạo độ dài cho nó. Có ba mức độ.
Giá trị ban đầu đặt sẵn `verbosity == 0`, là không áp dụng hồ sơ thời gian. Khi
bạn để `verbosity == 1`, nó sẽ lên hồ sơ thời gian chạy forward và backward. Khi
`verbosity == 2`, nó sẽ lên hồ sơ thời gian cho mỗi buffered operation trong
graph.

Sau đây là mã code ví dụ để thiết lập chức năng hồ sơ thời gian:

```python
# tạo thiết bị
from singa import device
dev = device.create_cuda_gpu()
# đặt độ dài
verbosity = 2
dev.SetVerbosity(verbosity)
# không bắt buộc: bỏ qua 5 vòng lặp đầu tiên khi lên hồ sơ thời gian
dev.SetSkipIteration(5)
```

Tiếp theo, sau khi kết thúc training ở cuối mỗi chương trình, chúng ta có thể in
kết quả hồ sơ thời gian bằng cách gọi method `device.PrintTimeProfiling`:

```python
dev.PrintTimeProfiling()
```

## Ví dụ đầu ra cho các độ dài khác nhau

Có thể chạy
[ví dụ](https://github.com/apache/singa/blob/master/examples/cnn/benchmark.py)
ResNet để xem kết quả với cách đặt độ dài khác nhau:

1. `verbosity == 1`

```
Time Profiling:
Forward Propagation Time : 0.0409127 sec
Backward Propagation Time : 0.114813 sec
```

2. `verbosity == 2`

```
Time Profiling:
OP_ID0. SetValue : 1.73722e-05 sec
OP_ID1. cudnnConvForward : 0.000612724 sec
OP_ID2. GpuBatchNormForwardTraining : 0.000559449 sec
OP_ID3. ReLU : 0.000375004 sec
OP_ID4. GpuPoolingForward : 0.000240041 sec
OP_ID5. SetValue : 3.4176e-06 sec
OP_ID6. cudnnConvForward : 0.000115619 sec
OP_ID7. GpuBatchNormForwardTraining : 0.000150415 sec
OP_ID8. ReLU : 9.95494e-05 sec
OP_ID9. SetValue : 3.22432e-06 sec
OP_ID10. cudnnConvForward : 0.000648668 sec
OP_ID11. GpuBatchNormForwardTraining : 0.000149793 sec
OP_ID12. ReLU : 9.92118e-05 sec
OP_ID13. SetValue : 3.37728e-06 sec
OP_ID14. cudnnConvForward : 0.000400953 sec
OP_ID15. GpuBatchNormForwardTraining : 0.000572181 sec
OP_ID16. SetValue : 3.21312e-06 sec
OP_ID17. cudnnConvForward : 0.000398698 sec
OP_ID18. GpuBatchNormForwardTraining : 0.00056836 sec
OP_ID19. Add : 0.000542246 sec
OP_ID20. ReLU : 0.000372783 sec
OP_ID21. SetValue : 3.25312e-06 sec
OP_ID22. cudnnConvForward : 0.000260731 sec
OP_ID23. GpuBatchNormForwardTraining : 0.000149041 sec
OP_ID24. ReLU : 9.9072e-05 sec
OP_ID25. SetValue : 3.10592e-06 sec
OP_ID26. cudnnConvForward : 0.000637481 sec
OP_ID27. GpuBatchNormForwardTraining : 0.000152577 sec
OP_ID28. ReLU : 9.90518e-05 sec
OP_ID29. SetValue : 3.28224e-06 sec
OP_ID30. cudnnConvForward : 0.000404586 sec
OP_ID31. GpuBatchNormForwardTraining : 0.000569679 sec
OP_ID32. Add : 0.000542291 sec
OP_ID33. ReLU : 0.00037211 sec
OP_ID34. SetValue : 3.13696e-06 sec
OP_ID35. cudnnConvForward : 0.000261219 sec
OP_ID36. GpuBatchNormForwardTraining : 0.000148281 sec
OP_ID37. ReLU : 9.89299e-05 sec
OP_ID38. SetValue : 3.25216e-06 sec
OP_ID39. cudnnConvForward : 0.000633644 sec
OP_ID40. GpuBatchNormForwardTraining : 0.000150711 sec
OP_ID41. ReLU : 9.84902e-05 sec
OP_ID42. SetValue : 3.18176e-06 sec
OP_ID43. cudnnConvForward : 0.000402752 sec
OP_ID44. GpuBatchNormForwardTraining : 0.000571523 sec
OP_ID45. Add : 0.000542435 sec
OP_ID46. ReLU : 0.000372539 sec
OP_ID47. SetValue : 3.24672e-06 sec
OP_ID48. cudnnConvForward : 0.000493054 sec
OP_ID49. GpuBatchNormForwardTraining : 0.000293142 sec
OP_ID50. ReLU : 0.000190047 sec
OP_ID51. SetValue : 3.14784e-06 sec
OP_ID52. cudnnConvForward : 0.00148837 sec
OP_ID53. GpuBatchNormForwardTraining : 8.34794e-05 sec
OP_ID54. ReLU : 5.23254e-05 sec
OP_ID55. SetValue : 3.40096e-06 sec
OP_ID56. cudnnConvForward : 0.000292971 sec
OP_ID57. GpuBatchNormForwardTraining : 0.00029174 sec
OP_ID58. SetValue : 3.3248e-06 sec
OP_ID59. cudnnConvForward : 0.000590154 sec
OP_ID60. GpuBatchNormForwardTraining : 0.000294149 sec
OP_ID61. Add : 0.000275119 sec
OP_ID62. ReLU : 0.000189268 sec
OP_ID63. SetValue : 3.2704e-06 sec
OP_ID64. cudnnConvForward : 0.000341232 sec
OP_ID65. GpuBatchNormForwardTraining : 8.3304e-05 sec
OP_ID66. ReLU : 5.23667e-05 sec
OP_ID67. SetValue : 3.19936e-06 sec
OP_ID68. cudnnConvForward : 0.000542484 sec
OP_ID69. GpuBatchNormForwardTraining : 8.60537e-05 sec
OP_ID70. ReLU : 5.2479e-05 sec
OP_ID71. SetValue : 3.41824e-06 sec
OP_ID72. cudnnConvForward : 0.000291295 sec
OP_ID73. GpuBatchNormForwardTraining : 0.000292795 sec
OP_ID74. Add : 0.000274438 sec
OP_ID75. ReLU : 0.000189689 sec
OP_ID76. SetValue : 3.21984e-06 sec
OP_ID77. cudnnConvForward : 0.000338776 sec
OP_ID78. GpuBatchNormForwardTraining : 8.484e-05 sec
OP_ID79. ReLU : 5.29408e-05 sec
OP_ID80. SetValue : 3.18208e-06 sec
OP_ID81. cudnnConvForward : 0.000545542 sec
OP_ID82. GpuBatchNormForwardTraining : 8.40976e-05 sec
OP_ID83. ReLU : 5.2256e-05 sec
OP_ID84. SetValue : 3.36256e-06 sec
OP_ID85. cudnnConvForward : 0.000293003 sec
OP_ID86. GpuBatchNormForwardTraining : 0.0002989 sec
OP_ID87. Add : 0.000275041 sec
OP_ID88. ReLU : 0.000189867 sec
OP_ID89. SetValue : 3.1184e-06 sec
OP_ID90. cudnnConvForward : 0.000340417 sec
OP_ID91. GpuBatchNormForwardTraining : 8.39395e-05 sec
OP_ID92. ReLU : 5.26544e-05 sec
OP_ID93. SetValue : 3.2336e-06 sec
OP_ID94. cudnnConvForward : 0.000539787 sec
OP_ID95. GpuBatchNormForwardTraining : 8.2753e-05 sec
OP_ID96. ReLU : 4.86758e-05 sec
OP_ID97. SetValue : 3.24384e-06 sec
OP_ID98. cudnnConvForward : 0.000287108 sec
OP_ID99. GpuBatchNormForwardTraining : 0.000293127 sec
OP_ID100. Add : 0.000269478 sec
.
.
.
```
