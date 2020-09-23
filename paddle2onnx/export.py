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
import onnx
from paddle2onnx.graph import Graph
from paddle2onnx.convert import convert_inputs, convert_outputs, convert_weights, convert_nodes
from paddle2onnx.optimizer import GraphOptimizer


def make_model(op_nodes, input_nodes, output_nodes, weight_nodes,
               opset_version):
    onnx_graph = onnx.helper.make_graph(
        nodes=weight_nodes + op_nodes,
        name='paddle-onnx',
        initializer=[],
        inputs=input_nodes,
        outputs=output_nodes)
    opset_imports = [onnx.helper.make_opsetid("", opset_version)]
    onnx_model = onnx.helper.make_model(
        onnx_graph, producer_name='PaddlePaddle', opset_imports=opset_imports)
    onnx.checker.check_model(onnx_model)

    return onnx_model


def export_dygraph(model,
                   save_dir,
                   input_spec=None,
                   configs=None,
                   opset_version=9):
    output_spec = None
    if configs is not None:
        output_spec = configs.output_spec

    graph = Graph.parse_graph(model, input_spec, output_spec)

    print("Converting PaddlePaddle to ONNX...\n")
    input_nodes = convert_inputs(graph.input_nodes)
    output_nodes = convert_outputs(graph.output_nodes)
    weight_nodes = convert_weights(graph.parameters)
    op_nodes = convert_nodes(graph.topo_sort, opset_version)

    onnx_model = make_model(op_nodes, input_nodes, output_nodes, weight_nodes,
                            opset_version)

    #optimizer = GraphOptimizer()
    #onnx_model = optimizer.optimize(onnx_model)

    path, file_name = os.path.split(save_dir)
    if path != '' and not os.path.isdir(path):
        os.makedirs(path)
    with open(save_dir, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print("\nONNX model saved in {}".format(save_dir))


def export_inference_program(program,
                             save_dir,
                             input_spec=None,
                             scope=None,
                             configs=None,
                             opset_version=9):
    output_spec = None
    if configs is not None:
        output_spec = configs.output_spec

    graph = Graph.parse_program(program, input_spec, output_spec, scope)

    print("Converting PaddlePaddle to ONNX...\n")
    input_nodes = convert_inputs(graph.input_nodes)
    output_nodes = convert_outputs(graph.output_nodes)
    weight_nodes = convert_weights(graph.parameters)
    op_nodes = convert_nodes(graph.topo_sort, opset_version)

    onnx_model = make_model(op_nodes, input_nodes, output_nodes, weight_nodes,
                            opset_version)

    #optimizer = GraphOptimizer()
    #onnx_model = optimizer.optimize(onnx_model)

    path, file_name = os.path.split(save_dir)
    if path != '' and not os.path.isdir(path):
        os.makedirs(path)
    with open(save_dir, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print("\nONNX model saved in {}".format(save_dir))
