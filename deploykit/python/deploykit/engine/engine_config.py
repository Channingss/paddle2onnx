# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class PaddleInferenceConfig(object):
    def __init__(self):
        # Whether to use mkdnn accelerator library when deploying on CPU
        self.use_mkl = True
        # The number of threads set when using mkldnn accelerator
        self.mkl_thread_num = 1
        # Whether to use GPU
        self.use_gpu = False
        # Set GPU ID, default is 0
        self.gpu_id = 0
        # Enable IR optimization
        self.use_ir_optim = True
        # Whether to use TensorRT
        self.use_trt = False
        # Set batch size
        self.batch_size = 1
        # Set TensorRT min_subgraph_size
        self.min_subgraph_size = 1
        # Set TensorRT data precision
        # 0: FP32
        # 1: FP16
        # 2: Int8
        self.precision = 0
        # When tensorrt is used, whether to serialize tensorrt engine to disk
        self.use_static = False
        # Is offline calibration required, when tensorrt is used
        self.use_calib_mode = False
        self.use_glog = False
        self.memory_optimize = True
