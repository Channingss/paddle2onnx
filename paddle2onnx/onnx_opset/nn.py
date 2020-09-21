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


@op_mapper(['conv2d', 'depthwise_conv2d'])
class Conv():
    @classmethod
    def opset_9(cls, node, **kw):
        kernel_shape = node.input_shape('Filter', 0)
        onnx_node = helper.make_node(
            'Conv',
            inputs=node.input('Input') + node.input('Filter'),
            outputs=node.output('Output'),
            dilations=node.attr('dilations'),
            kernel_shape=kernel_shape[-2:],
            strides=node.attr('strides'),
            group=node.attr('groups'),
            pads=node.attr('paddings') + node.attr('paddings'))
        return onnx_node


@op_mapper('conv2d_transpose')
class ConvTranspose():
    @classmethod
    def opset_9(cls, node, **kw):
        kernel_shape = node.input_shape('Filter', 0)
        node = helper.make_node(
            'ConvTranspose',
            inputs=node.input('Input') + node.input('Filter'),
            outputs=node.output('Output'),
            dilations=node.attr('dilations'),
            kernel_shape=kernel_shape[-2:],
            strides=node.attr('strides'),
            group=1,
            pads=node.attr('paddings') + node.attr('paddings'))
        return node


@op_mapper('pool2d')
class Pool():
    @classmethod
    def opset_9(cls, node, **kw):
        pool_type = {
            'max': ('MaxPool', 'GlobalMaxPool'),
            'avg': ('AveragePool', 'GlobalAveragePool')
        }
        if node.attr('global_pooling'):
            onnx_node = helper.make_node(
                pool_type[node.attr('pooling_type')][1],
                inputs=node.input('X'),
                outputs=node.output('Out'), )
            return onnx_node
        elif node.attr('adaptive'):
            raise Excpetion("ONNX cannot support adaptive pool")
        else:
            input_shape = node.input_shape('X', 0)
            k_size = node.attr('ksize')
            paddings = node.attr('paddings')
            if input_shape[2] > 0 and input_shape[2] + paddings[0] < k_size[0]:
                k_size[0] = input_shape[2] + paddings[0]
            if input_shape[3] > 0 and input_shape[3] + paddings[1] < k_size[1]:
                k_size[1] = input_shape[3] + paddings[1]
            onnx_node = helper.make_node(
                pool_type[node.attr('pooling_type')][0],
                inputs=node.input('X'),
                outputs=node.output('Out'),
                kernel_shape=k_size,
                strides=node.attr('strides'),
                pads=node.attr('paddings') + node.attr('paddings'))
            return onnx_node


@op_mapper('batch_norm')
class BatchNorm():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_attr = {
            'epsilon': node.attr('epsilon'),
            'momentum': node.attr('momentum')
        }
        inputs = node.input('X') + node.input('Scale') + node.input(
            'Bias') + node.input('Mean') + node.input('Variance')
        onnx_node = helper.make_node(
            'BatchNormalization',
            inputs=inputs,
            outputs=node.output('Y'),
            **onnx_attr)
        return onnx_node


@op_mapper('instance_norm')
class InstanceNorm():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_attr = {'epsilon': node.attr('epsilon'), }
        inputs = node.input('X') + node.input('Scale') + node.input('Bias')
        onnx_node = helper.make_node(
            'InstanceNormalization',
            inputs=inputs,
            outputs=node.output('Y'),
            **onnx_attr)
        return onnx_node


@op_mapper('dropout')
class Dropout():
    @classmethod
    def opset_9(cls, node, **kw):
        dropout_mode = node.attr('dropout_implementation')
        dropout_prob = node.attr('dropout_prob')
        if dropout_mode == 'upscale_in_train':
            onnx_node = helper.make_node(
                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
            return onnx_node
        elif dropout_mode == 'downgrade_in_infer':
            scale_name = get_name(node.type, 'scale')
            scale_node = make_constant_node(
                scale_name, onnx_pb.TensorProto.FLOAT, 1 - dropout_prob)
            onnx_node = helper.make_node(
                "Mul",
                inputs=[node.input('X')[0], scale_name],
                outputs=node.output('Out'))
            return [scale_node, onnx_node]
        else:
            raise Exception("Unexpected situation happend")
