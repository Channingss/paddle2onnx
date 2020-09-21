# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from six import text_type as _text_type
import argparse
import sys
import os
import paddle.fluid as fluid
from paddle2onnx.graph import Graph
from paddle2onnx.convert import convert
from paddle2onnx.optimizer import GraphOptimizer


def export_dygraph(model,
                   save_dir,
                   input_spec=None,
                   configs=None,
                   opset_version=9):
    output_spec = None
    if configs is not None:
        output_spec = configs.output_spec

    graph, param, input, output, block = Graph.parse_graph(model, input_spec,
                                                           output_spec)

    onnx_model = convert(graph, param, input, output, block, opset_version)

    #optimizer = GraphOptimizer()
    #onnx_model = optimizer.optimize(onnx_model)

    path, file_name = os.path.split(save_dir)
    if path != '' and not os.path.isdir(path):
        os.makedirs(path)
    with open(save_dir, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print("\nONNX model saved in {}".format(save_dir))
