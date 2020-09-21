#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
import math
import sys
import os
import inspect
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
from onnx import helper, onnx_pb
from paddle.fluid.dygraph.base import program_desc_tracing_guard, switch_to_static_graph
from .utils import DTYPE_PADDLE_ONNX_MAP


def make_tensor(var):
    tensor_info = helper.make_tensor_value_info(
        name=var.name,
        shape=var.shape,
        elem_type=DTYPE_PADDLE_ONNX_MAP[var.dtype])
    return tensor_info


def convert_inputs(inputs=None):
    input_nodes = []
    for ipt in inputs:
        if isinstance(ipt, fluid.Variable):
            input_nodes.append(make_tensor(ipt))
        if isinstance(ipt, dict):
            for key, var in ipt.items():
                input_nodes.append(make_tensor(ipt))
    return input_nodes


def convert_outputs(outputs=None):
    output_nodes = []
    for opt in outputs:
        if isinstance(opt, fluid.Variable):
            output_nodes.append(make_tensor(opt))
    return output_nodes


def convert_weights(parameters=None):
    nodes = list()
    if parameters is None:
        return nodes
    for param in parameters:
        if param.name.endswith('feed') or param.name.endswith('fetch'):
            continue
        if not param.persistable:
            continue
        weight = np.array(param.value().get_tensor())
        tensor = helper.make_tensor(
            name=param.name,
            dims=param.shape,
            data_type=DTYPE_PADDLE_ONNX_MAP[param.dtype],
            vals=weight.flatten().tolist())
        node = helper.make_node(
            'Constant', inputs=[], outputs=[param.name], value=tensor)
        nodes.append(node)
    return nodes


class OpMapper(object):
    OPSETS = {}

    def __init__(self, name, **kwargs):
        if not isinstance(name, list):
            name = [name]
        self.name = name
        self.kwargs = kwargs

    def __call__(self, cls):
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k.startswith("opset_"):
                version = int(k.replace("opset_", ""))
                if version not in OpMapper.OPSETS:
                    OpMapper.OPSETS[version] = {}
                opset_dict = OpMapper.OPSETS[version]
                for op in self.name:
                    opset_dict[op] = (v, self.kwargs)

    @staticmethod
    def convert_ops(graph, opset_version):
        op_nodes = list()
        input_nodes = list()
        output_nodes = list()
        unsupported_ops = set()
        opsets = OpMapper.OPSETS[opset_version]
        for i, op in enumerate(graph.topo_sort):
            sys.stdout.write("\rTotal:{}, Current:{} : {} ".format(
                len(graph.topo_sort), i + 1, op.type))
            sys.stdout.flush()
            if op.type == 'feed':
                #input_nodes.append(node)
                continue
            if op.type == 'fetch':
                #output_nodes.append(node)
                continue
            if op.type not in opsets:
                unsupported_ops.add(op.type)
                continue
            if len(unsupported_ops) > 0:
                continue
            mapper_func, kw = opsets[op.type]
            node = mapper_func(op, **kw)
            if isinstance(node, list):
                op_nodes = op_nodes + node
            else:
                op_nodes.append(node)

        if len(unsupported_ops) > 0:
            unsupported_ops_string = "\nThere's {} ops are not supported yet\n".format(
                len(unsupported_ops))
            for op in unsupported_ops:
                unsupported_ops_string += "=========== {} ===========\n".format(
                    op)
            raise ValueError(unsupported_ops_string)
        return op_nodes


def convert(graph, parameters, inputs, outputs, block, opset_version):
    print("Converting PaddlePaddle to ONNX...\n")
    input_nodes = convert_inputs(inputs)
    output_nodes = convert_outputs(outputs)
    weight_nodes = convert_weights(parameters)
    op_nodes = OpMapper.convert_ops(graph, opset_version)

    onnx_graph = helper.make_graph(
        nodes=weight_nodes + op_nodes,
        name='paddle-onnx',
        initializer=[],
        inputs=input_nodes,
        outputs=output_nodes)
    opset_imports = [helper.make_opsetid("", opset_version)]
    onnx_model = helper.make_model(
        onnx_graph, producer_name='PaddlePaddle', opset_imports=opset_imports)
    onnx.checker.check_model(onnx_model)

    return onnx_model
