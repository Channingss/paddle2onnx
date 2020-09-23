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


def get_max_support_version(versions, opset_version):
    max_version = -1
    for vs in sorted(versions):
        if vs < opset_version:
            max_version = vs
    return max_version


def convert_nodes(node_list, opset_version):
    onnx_nodes = list()
    unsupported_op_type = set()
    for i, node in enumerate(node_list):
        sys.stdout.write("\rTotal:{}, Current:{} : {} ".format(
            len(node_list), i + 1, node.type))
        sys.stdout.flush()
        if node.type == 'feed':
            #input_nodes.append(node)
            continue
        if node.type == 'fetch':
            #output_nodes.append(node)
            continue

        onnx_node = OpMapper.mapping(node, opset_version)
        if isinstance(onnx_node, list):
            onnx_nodes = onnx_nodes + onnx_node
        elif onnx_node is None:
            unsupported_op_type.add(node.type)
        else:
            onnx_nodes.append(node)

    if len(unsupported_op_type) > 0:
        unsupported_op_type_string = "\nThere's {} ops are not supported yet\n".format(
            len(unsupported_op_type))
        for op_type in unsupported_op_type:
            unsupported_op_type_string += "=========== {} ===========\n".format(
                op_type)
        raise ValueError(unsupported_op_type_string)
    return onnx_nodes


def convert(graph, parameters, inputs, outputs, block, opset_version):
    print("Converting PaddlePaddle to ONNX...\n")
    input_nodes = convert_inputs(inputs)
    output_nodes = convert_outputs(outputs)
    weight_nodes = convert_weights(parameters)
    onnx_nodes = convert_nodes(graph, opset_version)

    onnx_graph = helper.make_graph(
        nodes=weight_nodes + onnx_nodes,
        name='paddle-onnx',
        initializer=[],
        inputs=input_nodes,
        outputs=output_nodes)
    opset_imports = [helper.make_opsetid("", opset_version)]
    onnx_model = helper.make_model(
        onnx_graph, producer_name='PaddlePaddle', opset_imports=opset_imports)
    onnx.checker.check_model(onnx_model)

    return onnx_model


class OpMapper(object):
    OPSETS = {}

    def __init__(self, pd_op, **kwargs):
        if not isinstance(pd_op, list):
            pd_op = [pd_op]
        self.pd_op = pd_op
        self.kwargs = kwargs

    def __call__(self, cls):
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k.startswith("opset_"):
                version = int(k.replace("opset_", ""))
                opset_dict = OpMapper.OPSETS[self.name]
                for pd_op in self.name:
                    if pd_op not in OpMapper.OPSETS:
                        OpMapper.OPSETS[self.pd_op] = {}
                    opset_dict[version] = (v, self.kwargs)

    @staticmethod
    def mapping(node, opset_version):
        if node.type not in OpMapper.OPSETS:
            return None
        opsets_mapping = OpMapper.OPSETS[node.type]
        versions = opsets_mapping.keys()
        convert_version = get_max_support_version(versions, opset_version)
        mapper_func, kw = opsets[convert_version]
        onnx_node = mapper_func(op, **kw)
        return onnx_node
