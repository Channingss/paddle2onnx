#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import numpy as np
import copy
from paddle2onnx.constant import dtypes
from paddle2onnx.onnx_helper import helper
from paddle2onnx.mapper.op_mapper import OpMapper
from paddle2onnx.constant import PRODUCER
from paddle2onnx.constant.op_mapping_status import *


def make_value_info(name, shape, dtype):
    tensor_info = helper.make_tensor_value_info(
        name=name, shape=shape, elem_type=dtypes.DTYPE_PADDLE_ONNX_MAP[dtype])
    return tensor_info


def make_onnx_constant_node(node):
    dtype = node.attr('dtype')
    value = node.attr('value')
    if isinstance(value, list):
        dims = (len(value), )
    elif value is None:
        dims = ()
        value = []
    else:
        dims = ()
        value = [value]

    if 'dims' in node.attrs:
        dims = node.attrs['dims']

    tensor = helper.make_tensor(
        name=node.layer_name, data_type=dtype, dims=dims, vals=value)

    onnx_node = helper.make_node(
        node.type, inputs=node.inputs, outputs=node.outputs, value=tensor)

    return onnx_node


def mapping_node_to_onnx(node):
    if node.type in ['Constant', 'ConstantOfShape']:
        return make_onnx_constant_node(node)
    else:
        onnx_node = helper.make_node(
            node.type,
            inputs=node.inputs,
            outputs=node.outputs,
            name=node.layer_name,
            **node.attrs)
        return onnx_node


def mapping_inputs_to_onnx(inputs=None):
    input_nodes = []
    for ipt in inputs:
        vi = make_value_info(ipt.layer_name,
                             ipt.attr('shape'), ipt.attr('dtype'))
        input_nodes.append(vi)
    return input_nodes


def mapping_outputs_to_onnx(outputs=None):
    output_nodes = []
    for opt in outputs:
        vi = make_value_info(opt.layer_name,
                             opt.attr('shape'), opt.attr('dtype'))
        output_nodes.append(vi)
    return output_nodes


def mapping_weights_to_onnx(parameters=None):
    nodes = []
    if parameters is None:
        return nodes
    for name, param in parameters.items():
        weight = param['data']
        if weight is not np.ndarray:
            weight = np.array(weight)
        tensor = helper.make_tensor(
            name=name,
            dims=param['shape'],
            data_type=dtypes.DTYPE_PADDLE_ONNX_MAP[param['dtype']],
            vals=weight.flatten().tolist())
        node = helper.make_node(
            'Constant', inputs=[], outputs=[name], value=tensor)
        nodes.append(node)
    return nodes


def check_op_mapping_status(op_mapping_status, opset_version):
    if len(op_mapping_status[OP_MAPPING_NO_REGISTER]) > 0:
        unsupported_op_types = set(
            [node.type for node in mapping_status[OP_MAPPING_NO_REGISTER]])
        error_info = "\nThere's {} ops are not supported yet\n".format(
            len(unsupported_op_types))
        for op_type in unsupported_op_types:
            error_info += "=========== {} ===========\n".format(op_type)
        raise NotImplementedError(error_info)

    if len(op_mapping_status[OP_MAPPING_NO_VERSION]) > 0:
        unsupported_op_types = set(
            [node.type for node in op_mapping_status[OP_MAPPING_NO_VERSION]])
        error_info = "\nThere's {} ops are not supported in opset_version {}, please try other opset versions\n".format(
            len(unsupported_op_types), opset_version)

        for op_type in unsupported_op_types:
            error_info += "=========== {} ===========\n".format(op_type)
        raise NotImplementedError(error_info)


def mapping_nodes_to_onnx(graph, verbose=False):
    op_mapping_status = {
        OP_MAPPING_NO_REGISTER: [],
        OP_MAPPING_NO_VERSION: [],
        OP_MAPPING_SUCCESSED: [],
    }
    graph = copy.copy(graph)

    for name, node in list(graph.node_map.items()):
        status = OpMapper.mapping(graph, node, graph.opset_version)
        op_mapping_status[status].append(node)

    check_op_mapping_status(op_mapping_status, graph.opset_version)
    return [node.node for node in graph.node_map.values()]


def mapping_graph_to_onnx_proto(graph, verbose=False):
    input_nodes = mapping_inputs_to_onnx(graph.input_nodes)
    output_nodes = mapping_outputs_to_onnx(graph.output_nodes)
    weight_nodes = mapping_weights_to_onnx(graph.parameters)
    op_nodes = mapping_nodes_to_onnx(graph)

    onnx_graph = helper.make_graph(
        nodes=weight_nodes + op_nodes,
        name='paddle-onnx',
        initializer=[],
        inputs=input_nodes,
        outputs=output_nodes)

    opset_imports = [helper.make_opsetid("", graph.opset_version)]
    onnx_proto = helper.make_model(
        onnx_graph, producer_name=PRODUCER, opset_imports=opset_imports)

    return onnx_proto
