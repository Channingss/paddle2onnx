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
from paddle2onnx.op_mapper import OpMapper as op_mapper


@op_mapper('matmul')
class MatMul():
    @classmethod
    def opset_9(cls, node, **kw):
        x = node.input('X', idx=0)
        y = node.input('Y', idx=0)
        onnx_node = helper.make_node(
            'MatMul', inputs=[x, y], outputs=node.output('Out'))
        return onnx_node


@op_mapper('exp')
class Exp():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'Exp', inputs=node.input('X'), outputs=node.output('Out'))
        return onnx_node


@op_mapper('abs')
class Abs:
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'Abs', inputs=op.input('X'), outputs=node.output('Out'))
        return onnx_node


@op_mapper(
    [
        'elementwise_add', 'elementwise_sub', 'elementwise_div',
        'elementwise_mul'
    ],
    mapper_dict={
        'elementwise_add': 'Add',
        'elementwise_sub': 'Sub',
        'elementwise_div': 'Div',
        'elementwise_mul': 'Mul',
    })
class ElementwiseOps():
    @classmethod
    def opset_9(cls, node, **kw):
        op_type = kw['mapper_dict'][node.type]
        axis = node.attr('axis')
        x = node.input('X', 0)
        y = node.input('Y', 0)
        x_shape = node.input_shape('X', 0)
        y_shape = node.input_shape('Y', 0)
        if len(y_shape) == 1 and axis == 1:
            shape_name = get_name(node.type, 'shape')
            shape_value = [1] * len(x_shape)
            shape_value[axis] = y_shape[0]
            shape_node = make_constant_node(
                shape_name, onnx_pb.TensorProto.INT64, shape_value)
            temp_value = get_name(node.type, 'temp')
            y_node = helper.make_node(
                'Reshape', inputs=[y, shape_name], outputs=[temp_value])
            onnx_node = helper.make_node(
                op_type, inputs=[x, temp_value], outputs=node.outputs('Out'))
            return [shape_node, y_node, onnx_node]
        elif axis == -1 or axis == (len(x_shape) - 1
                                    ) or len(x_shape) == len(y_shape):
            onnx_node = helper.make_node(
                op_type, inputs=[x, y], outputs=node.outputs('Out'))
            return onnx_node
        else:
            raise Exception("Unexpected situation happend in elementwise_{}".
                            format(op_type.lower()))


@op_mapper('mul')
class Mul():
    @classmethod
    def opset_9(cls, node, **kw):
        x = node.input('X', 0)
        y = node.input('Y', 0)
        out = node.output('Out', 0)
        x_shape = node.input_shape('X', 0)
        y_shape = node.input_shape('Y', 0)
        out_shape = list(node.output_shape('Out', 0))
        x_num_col_dims = node.attr('x_num_col_dims')
        y_num_col_dims = node.attr('y_num_col_dims')
        flatten_x_name = 'flatten_{}'.format(x)
        flatten_y_name = 'flatten_{}'.format(y)
        shape_name = 'temp_shape_{}'.format(out)
        temp_out_name = 'temp_{}'.format(out)
        flatten_x = helper.make_node(
            'Flatten',
            inputs=node.input('X'),
            outputs=[flatten_x_name],
            axis=x_num_col_dims)
        flatten_y = helper.make_node(
            'Flatten',
            inputs=node.input('Y'),
            outputs=[flatten_y_name],
            axis=y_num_col_dims)
        shape_node = make_constant_node(shape_name, onnx_pb.TensorProto.INT64,
                                        out_shape)
        onnx_node = helper.make_node(
            'MatMul',
            inputs=[flatten_x_name, flatten_y_name],
            outputs=[temp_out_name])
        reshape_out = helper.make_node(
            'Reshape',
            inputs=[temp_out_name, shape_name],
            outputs=node.output('Out'))
        return [flatten_x, flatten_y, shape_node, onnx_node, reshape_out]


@op_mapper('sum')
class Sum():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'Sum', inputs=node.input('X'), outputs=node.output('Out'))
        return onnx_node


@op_mapper('floor')
class Floor():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'Floor', inputs=node.input('X'), outputs=node.output('Out'))
        return onnx_node


@op_mapper('reduce_mean')
class ReduceMean():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'ReduceMean',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            axes=node.attr('dim'),
            keepdims=node.attr('keep_dim'))
        return onnx_node


@op_mapper('arg_max')
class ArgMax():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'ArgMax',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            axis=node.attr('axis'),
            keepdims=0)
        return onnx_node


@op_mapper('scale')
class Linear():
    @classmethod
    def opset_9(cls, node, **kw):
        scale = node.attr('scale')
        bias = node.attr('bias')
        if math.fabs(scale - 1.0) < 1e-06 and math.fabs(bias - 0.0) < 1e-06:
            onnx_node = helper.make_node(
                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
            return onnx_node
        else:
            scale_name = get_name(node.type, 'scale')
            bias_name = get_name(node.type, 'bias')
            scale_node = make_constant_node(scale_name,
                                            onnx_pb.TensorProto.FLOAT, scale)
            bias_node = make_constant_node(bias_name, onnx_pb.TensorProto.FLOAT,
                                           bias)
            temp_tensor_name = get_name(node.type, 'temporary')
            node_cast_name = get_name(node.type, 'cast')
            node_cast = helper.make_node(
                'Cast',
                inputs=node.input('X'),
                outputs=[node_cast_name],
                to=onnx_pb.TensorProto.FLOAT)
            if node.attr('bias_after_scale'):
                node1 = helper.make_node(
                    'Mul',
                    inputs=[scale_name, node_cast_name],
                    outputs=[temp_tensor_name])
                node2 = helper.make_node(
                    'Add',
                    inputs=[bias_name, temp_tensor_name],
                    outputs=node.output('Out'))
            else:
                node1 = helper.make_node(
                    'Add',
                    inputs=[bias_name, node_cast_name],
                    outputs=temp_tensor_name)
                node2 = helper.make_node(
                    'Mul',
                    inputs=[scale_name, temp_tensor_name],
                    outputs=[node.output('Out')])
            return [scale_node, bias_node, node_cast, node1, node2]


@op_mapper('softmax')
class Softmax():
    @classmethod
    def opset_9(cls, node, **kw):
        axis = node.attr('axis')
        shape = node.output_shape('Out', 0)
        if axis < 0:
            axis += len(shape)
        if axis == len(shape) - 1:
            node = helper.make_node(
                'Softmax',
                inputs=node.input('X'),
                outputs=node.output('Out'),
                axis=node.attr('axis'))
            return node
        else:
            perm = [i for i in range(len(shape))]
            perm[-1] = axis
            perm[axis] = len(shape) - 1
            transpose_name0 = get_name(node.type, 'transpose')
            transpose_node0 = helper.make_node(
                'Transpose',
                inputs=node.input('X'),
                outputs=[transpose_name0],
                perm=perm)
            softmax_name = get_name(node.type, 'softmax')
            softmax_node = helper.make_node(
                'Softmax',
                inputs=[transpose_name0],
                outputs=[softmax_name],
                axis=-1)
            transpose_name1 = get_name(node.type, 'transpose')
            transpose_node1 = helper.make_node(
                'Transpose',
                inputs=[softmax_name],
                outputs=node.output('Out'),
                perm=perm)
            return [transpose_node0, softmax_node, transpose_node1]
