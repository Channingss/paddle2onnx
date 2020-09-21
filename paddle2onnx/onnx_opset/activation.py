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

import math
import sys
import os
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
from onnx import helper, onnx_pb
from paddle2onnx.utils import DTYPE_PADDLE_ONNX_MAP, DTYPE_ONNX_NUMPY_MAP, get_name, make_constant_node
from paddle2onnx.convert import OpMapper as op_mapper


@op_mapper(
    ['relu', 'tanh', 'log', 'sigmoid', 'leaky_relu'],
    mapper_dict={
        'relu': 'Relu',
        'tanh': 'Tanh',
        'log': 'Log',
        'sigmoid': 'Sigmoid',
    })
class ActivationOps():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_type = kw['mapper_dict'][node.type]
        onnx_node = helper.make_node(
            onnx_type, inputs=node.input('X'), outputs=node.output('Out'))
        return onnx_node


@op_mapper('leaky_relu')
class LeakyRelu():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'LeakyRelu',
            inputs=[node.input('X')[0]],
            outputs=node.output('Out'),
            alpha=node.attr('alpha'))
        return onnx_node


@op_mapper('relu6')
class Relu6():
    @classmethod
    def opset_9(cls, node, **kw):
        threshold = node.attr('threshold')
        onnx_node = helper.make_node(
            'Clip',
            inputs=[node.input('X')[0]],
            outputs=node.output('Out'),
            max=threshold,
            min=0.0)
        return onnx_node

    @classmethod
    def opset_11(cls, node, **kw):
        min_name = get_name(node.type, 'min')
        max_name = get_name(node.type, 'max')
        min_node = make_constant_node(min_name, onnx_pb.TensorProto.FLOAT, 0)
        max_node = make_constant_node(max_name, onnx_pb.TensorProto.FLOAT,
                                      node.attr('threshold'))
        node = helper.make_node(
            'Clip',
            inputs=[node.input('X')[0], min_name, max_name],
            outputs=node.output('Out'), )
        return [min_node, max_node, node]


@op_mapper('hard_sigmoid')
class Relu6():
    @classmethod
    def opset_9(cls, node, **kw):
        slope = node.attr('slope')
        offset = node.attr('offset')
        node = helper.make_node(
            'HardSigmoid',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            alpha=slope,
            beta=offset)
        return node


@op_mapper('swish')
class Swish():
    @classmethod
    def opset_9(cls, node, **kw):
        beta = node.attr('beta')
        beta_name = get_name(node.type, 'beta')
        beta_node = onnx.helper.make_node(
            'Constant',
            name=beta_name,
            inputs=[],
            outputs=[beta_name],
            value=onnx.helper.make_tensor(
                name=beta_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=(),
                vals=[beta]))

        beta_x_name = get_name(node.type, 'beta_x')
        beta_x_node = onnx.helper.make_node(
            'Mul',
            name=beta_x_name,
            inputs=[node.input('X')[0], beta_name],
            outputs=[beta_x_name])
        sigmoid_name = get_name(node.type, 'sigmoid')
        sigmoid_node = onnx.helper.make_node(
            'Sigmoid',
            name=sigmoid_name,
            inputs=[beta_x_name],
            outputs=[sigmoid_name])
        swish_node = onnx.helper.make_node(
            'Mul',
            inputs=[node.input('X')[0], sigmoid_name],
            outputs=node.output('Out'))
        return [beta_node, beta_x_node, sigmoid_node, swish_node]


@op_mapper('swish')
class Swish():
    @classmethod
    def opset_9(cls, node, **kw):
        scale_name = get_name(node.type, 'scale')
        offset_name = get_name(node.type, 'offset')
        scale_node = make_constant_node(scale_name, onnx_pb.TensorProto.FLOAT,
                                        node.attr('scale'))
        offset_node = make_constant_node(offset_name, onnx_pb.TensorProto.FLOAT,
                                         node.attr('offset'))

        name0 = get_name(node.type, 'add')
        node0 = helper.make_node(
            'Add', inputs=[node.input('X')[0], offset_name], outputs=[name0])
        name1 = get_name(node.type, 'relu')
        min_value = 0.0
        max_value = node.attr('threshold')
        node1 = helper.make_node(
            'Clip',
            inputs=[name0],
            outputs=[name1],
            max=max_value,
            min=min_value)
        name2 = get_name(node.type, 'mul')
        node2 = helper.make_node(
            'Mul', inputs=[node.input('X')[0], name1], outputs=[name2])
        node3 = helper.make_node(
            'Div', inputs=[name2, scale_name], outputs=node.output('Out'))
        return [scale_node, offset_node, node0, node1, node2, node3]
