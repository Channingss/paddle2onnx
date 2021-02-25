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

import os
import os.path as osp

from deploykit.common import DataBlob
from .engine_config import PaddleInferenceConfig


class PaddleInferenceEngine(object):
    def __init__(self, model_dir, engine_config):
        import paddle
        from paddle import fluid
        try:
            paddle.enable_static()
        except:
            pass

        if not osp.exists(model_dir):
            raise ValueError(
                "model_dir {} does not exist, please set a right path".format(
                    model_dir))
        self.model_dir = model_dir
        if not isinstance(engine_config, PaddleInferenceConfig):
            raise TypeError(
                "Type of engine_config should be PaddleInferenceConfig, but recieved is {}".
                format(type(engine_config)))
        self.engine_config = engine_config
        self.precision_dict = {
            0: fluid.core.AnalysisConfig.Precision.Float32,
            1: fluid.core.AnalysisConfig.Precision.Half,
            2: fluid.core.AnalysisConfig.Precision.Int8
        }
        self.create_predictor()

    def create_predictor(self):
        import paddle
        from paddle import fluid
        try:
            paddle.enable_static()
        except:
            pass

        config = fluid.core.AnalysisConfig(
            os.path.join(self.model_dir, '__model__'),
            os.path.join(self.model_dir, '__params__'))

        if self.engine_config.use_gpu:
            # 设置GPU初始显存(单位M)和Device ID
            config.enable_use_gpu(100, self.engine_config.gpu_id)
            if self.engine_config.use_trt:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 10,
                    max_batch_size=self.engine_config.batch_size,
                    min_subgraph_size=self.engine_config.min_subgraph_size,
                    precision_mode=self.precision_dict[
                        self.engine_config.precision],
                    use_static=self.engine_config.use_static,
                    use_calib_mode=self.engine_config.use_calib_mode)
        else:
            config.disable_gpu()
        if not self.engine_config.use_gpu:
            config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(
                self.engine_config.mkl_thread_num)
        if self.engine_config.use_glog:
            config.enable_glog_info()
        else:
            config.disable_glog_info()
        if self.engine_config.memory_optimize:
            config.enable_memory_optim()
        else:
            config.disable_memory_optim()
        config.switch_ir_optim(self.engine_config.use_ir_optim)
        config.switch_use_feed_fetch_ops(False)
        config.switch_specify_input_names(True)
        self.predictor = fluid.core.create_paddle_predictor(config)

    def infer(self, inputs):
        for data_blob in inputs:
            try:
                tensor = self.predictor.get_input_tensor(data_blob.name)
            except:
                continue
            tensor.copy_from_cpu(data_blob.data)
        self.predictor.zero_copy_run()
        output_names = self.predictor.get_output_names()
        outputs = list()
        for name in output_names:
            output_tensor = self.predictor.get_output_tensor(name)
            output_data_blob = DataBlob()
            output_data_blob.name = name
            output_data_blob.data = output_tensor.copy_to_cpu()
            output_data_blob.lod = output_tensor.lod()
            outputs.append(output_data_blob)
        return outputs
