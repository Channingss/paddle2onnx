# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
import sys
import inspect
import numpy as np
import paddle.fluid as fluid
import onnx
from onnx import helper, onnx_pb
from .utils import DTYPE_PADDLE_ONNX_MAP
from paddle2onnx.op_mapper import OpMapper


def make_value_info(name, shape, dtype):
    tensor_info = helper.make_tensor_value_info(
        name=name, shape=shape, elem_type=DTYPE_PADDLE_ONNX_MAP[dtype])
    return tensor_info


def convert_inputs(inputs=None):
    input_nodes = []
    for ipt in inputs:
        vi = make_value_info(ipt.layer_name,
                             ipt.attr('shape'), ipt.attr('dtype'))
        input_nodes.append(vi)
    return input_nodes


def convert_outputs(outputs=None):
    output_nodes = []
    for opt in outputs:
        vi = make_value_info(opt.layer_name,
                             opt.attr('shape'), opt.attr('dtype'))
        output_nodes.append(vi)
    return output_nodes


def convert_weights(parameters=None):
    nodes = list()
    if parameters is None:
        return nodes
    for name, param in parameters.items():
        weight = param['data']
        if weight is not np.ndarray:
            weight = np.array(weight)
        tensor = helper.make_tensor(
            name=name,
            dims=param['shape'],
            data_type=DTYPE_PADDLE_ONNX_MAP[param['dtype']],
            vals=weight.flatten().tolist())
        node = helper.make_node(
            'Constant', inputs=[], outputs=[name], value=tensor)
        nodes.append(node)
    return nodes


def convert_nodes(node_list, opset_version):
    onnx_nodes = list()
    unsupported_op_type = set()
    for i, node in enumerate(node_list):
        sys.stdout.write("\rTotal:{}, Current:{} : {} ".format(
            len(node_list), i + 1, node.type))
        sys.stdout.flush()
        onnx_node = OpMapper.mapping(node, opset_version)
        if isinstance(onnx_node, list):
            onnx_nodes = onnx_nodes + onnx_node
        elif onnx_node is None:
            unsupported_op_type.add(node.type)
        else:
            onnx_nodes.append(onnx_node)

    if len(unsupported_op_type) > 0:
        unsupported_op_type_string = "\nThere's {} ops are not supported yet\n".format(
            len(unsupported_op_type))
        for op_type in unsupported_op_type:
            unsupported_op_type_string += "=========== {} ===========\n".format(
                op_type)
        raise ValueError(unsupported_op_type_string)
    return onnx_nodes
