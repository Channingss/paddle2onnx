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

import math
import sys
import os
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
from onnx import helper, onnx_pb
from ..utils import DTYPE_PADDLE_ONNX_MAP, DTYPE_ONNX_NUMPY_MAP, get_name, make_constant_node


def conv2d(op, block):
    kernel_shape = block.var(op.input('Filter')[0]).shape
    node = helper.make_node(
        'Conv',
        inputs=op.input('Input') + op.input('Filter'),
        outputs=op.output('Output'),
        dilations=op.attr('dilations'),
        kernel_shape=kernel_shape[-2:],
        strides=op.attr('strides'),
        group=op.attr('groups'),
        pads=op.attr('paddings') + op.attr('paddings'))
    return node


def conv2d_transpose(op, block):
    kernel_shape = block.var(op.input('Filter')[0]).shape
    node = helper.make_node(
        'ConvTranspose',
        inputs=op.input('Input') + op.input('Filter'),
        outputs=op.output('Output'),
        dilations=op.attr('dilations'),
        kernel_shape=kernel_shape[-2:],
        strides=op.attr('strides'),
        group=1,
        pads=op.attr('paddings') + op.attr('paddings'))
    return node


def relu(op, block):
    node = helper.make_node(
        'Relu', inputs=op.input('X'), outputs=op.output('Out'))
    return node


def tanh(op, block):
    node = helper.make_node(
        'Tanh', inputs=op.input('X'), outputs=op.output('Out'))
    return node


def log(op, block):
    node = helper.make_node(
        'Log', inputs=op.input('X'), outputs=op.output('Out'))
    return node


def sigmoid(op, block):
    node = helper.make_node(
        'Sigmoid', inputs=op.input('X'), outputs=op.output('Out'))
    return node


def clip(op, block):
    min_value = op.attr('min')
    max_value = op.attr('max')
    node = helper.make_node(
        'Clip',
        inputs=[op.input('X')[0]],
        outputs=op.output('Out'),
        max=max_value,
        min=min_value)
    return node


def exp(op, block):
    node = helper.make_node(
        'Exp', inputs=op.input('X'), outputs=op.output('Out'))
    return node


def abs(op, block):
    node = helper.make_node(
        'Abs', inputs=op.input('X'), outputs=op.output('Out'))
    return node


def leaky_relu(op, block):
    node = helper.make_node(
        'LeakyRelu',
        inputs=op.input('X'),
        outputs=op.output('Out'),
        alpha=op.attr('alpha'))
    return node


def _elementwise_ops(op, block, op_type):
    axis = op.attr('axis')
    x_shape = block.var(op.input('X')[0]).shape
    y_shape = block.var(op.input('Y')[0]).shape
    if len(y_shape) == 1 and axis == 1:
        shape_name = get_name(op.type, 'shape')
        shape_value = [1] * len(x_shape)
        shape_value[axis] = y_shape[0]
        shape_node = make_constant_node(shape_name, onnx_pb.TensorProto.INT64,
                                        shape_value)
        temp_value = get_name(op.type, 'temp')
        y_node = helper.make_node(
            'Reshape',
            inputs=[op.input('Y')[0], shape_name],
            outputs=[temp_value])
        node = helper.make_node(
            op_type,
            inputs=[op.input('X')[0], temp_value],
            outputs=op.output('Out'))
        return [shape_node, y_node, node]
    elif axis == -1 or axis == (len(x_shape) - 1
                                ) or len(x_shape) == len(y_shape):
        node = helper.make_node(
            op_type,
            inputs=[op.input('X')[0], op.input('Y')[0]],
            outputs=op.output('Out'))
        return node
    else:
        raise Exception("Unexpected situation happend in elementwise_{}".format(
            op_type.lower()))


def elementwise_add(op, block):
    return _elementwise_ops(op, block, 'Add')


def elementwise_sub(op, block):
    return _elementwise_ops(op, block, 'Sub')


def elementwise_div(op, block):
    return _elementwise_ops(op, block, 'Div')


def elementwise_mul(op, block):
    return _elementwise_ops(op, block, 'Mul')


def pool2d(op, block):
    pool_type = {
        'max': ('MaxPool', 'GlobalMaxPool'),
        'avg': ('AveragePool', 'GlobalAveragePool')
    }
    if op.attr('global_pooling'):
        node = helper.make_node(
            pool_type[op.attr('pooling_type')][1],
            inputs=op.input('X'),
            outputs=op.output('Out'), )
    elif op.attr('adaptive'):
        raise Excpetion("ONNX cannot support adaptive pool")
    else:
        input_shape = block.var(op.input('X')[0]).shape
        k_size = op.attr('ksize')
        paddings = op.attr('paddings')
        if input_shape[2] > 0 and input_shape[2] + paddings[0] < k_size[0]:
            k_size[0] = input_shape[2] + paddings[0]
        if input_shape[3] > 0 and input_shape[3] + paddings[1] < k_size[1]:
            k_size[1] = input_shape[3] + paddings[1]
        node = helper.make_node(
            pool_type[op.attr('pooling_type')][0],
            inputs=op.input('X'),
            outputs=op.output('Out'),
            kernel_shape=k_size,
            strides=op.attr('strides'),
            pads=op.attr('paddings') + op.attr('paddings'))
    return node


def pad2d(op, block):
    x_shape = block.var(op.input('X')[0]).shape
    paddings = op.attr('paddings')
    onnx_pads = []
    if op.attr('data_format') == 'NCHW':
        pads = [0, 0, paddings[0], paddings[2], 0, 0, paddings[1], paddings[3]]
    else:
        pads = [0, paddings[0], paddings[2], 0, 0, paddings[1], paddings[3], 0]
    #TODO support pads is Variable
    node = helper.make_node(
        'Pad',
        inputs=op.input('X'),
        outputs=op.output('Out'),
        mode=op.attr('mode'),
        value=op.attr('pad_value'),
        pads=pads)
    return node


def softmax(op, block):
    axis = op.attr('axis')
    shape = block.var(op.output('Out')[0]).shape
    if axis < 0:
        axis += len(shape)
    if axis == len(shape) - 1:
        node = helper.make_node(
            'Softmax',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            axis=op.attr('axis'))
        return node
    else:
        perm = [i for i in range(len(shape))]
        perm[-1] = axis
        perm[axis] = len(shape) - 1
        transpose_name0 = get_name(op.type, 'transpose')
        transpose_node0 = helper.make_node(
            'Transpose',
            inputs=op.input('X'),
            outputs=[transpose_name0],
            perm=perm)
        softmax_name = get_name(op.type, 'softmax')
        softmax_node = helper.make_node(
            'Softmax',
            inputs=[transpose_name0],
            outputs=[softmax_name],
            axis=-1)
        transpose_name1 = get_name(op.type, 'transpose')
        transpose_node1 = helper.make_node(
            'Transpose',
            inputs=[softmax_name],
            outputs=op.output('Out'),
            perm=perm)
        return [transpose_node0, softmax_node, transpose_node1]


def scale(op, block):
    scale = op.attr('scale')
    bias = op.attr('bias')
    if math.fabs(scale - 1.0) < 1e-06 and math.fabs(bias - 0.0) < 1e-06:
        node = helper.make_node(
            'Identity', inputs=op.input('X'), outputs=op.output('Out'))
        return node
    else:
        scale_name = get_name(op.type, 'scale')
        bias_name = get_name(op.type, 'bias')
        scale_node = make_constant_node(scale_name, onnx_pb.TensorProto.FLOAT,
                                        scale)
        bias_node = make_constant_node(bias_name, onnx_pb.TensorProto.FLOAT,
                                       bias)
        temp_tensor_name = get_name(op.type, 'temporary')
        node_cast_name = get_name(op.type, 'cast')
        node_cast = helper.make_node(
            'Cast',
            inputs=op.input('X'),
            outputs=[node_cast_name],
            to=onnx_pb.TensorProto.FLOAT)
        if op.attr('bias_after_scale'):
            node1 = helper.make_node(
                'Mul',
                inputs=[scale_name, node_cast_name],
                outputs=[temp_tensor_name])
            node2 = helper.make_node(
                'Add',
                inputs=[bias_name, temp_tensor_name],
                outputs=op.output('Out'))
        else:
            node1 = helper.make_node(
                'Add',
                inputs=[bias_name, node_cast_name],
                outputs=temp_tensor_name)
            node2 = helper.make_node(
                'Mul',
                inputs=[scale_name, temp_tensor_name],
                outputs=[op.output('Out')])
        return [scale_node, bias_node, node_cast, node1, node2]


def mul(op, block):
    x_shape = block.var(op.input('X')[0]).shape
    y_shape = block.var(op.input('Y')[0]).shape
    out_shape = list(block.var(op.output('Out')[0]).shape)
    x_num_col_dims = op.attr('x_num_col_dims')
    y_num_col_dims = op.attr('y_num_col_dims')
    flatten_x_name = 'flatten_{}'.format(op.input('X')[0])
    flatten_y_name = 'flatten_{}'.format(op.input('Y')[0])
    shape_name = 'temp_shape_{}'.format(op.output('Out')[0])
    temp_out_name = 'temp_{}'.format(op.output('Out')[0])
    flatten_x = helper.make_node(
        'Flatten',
        inputs=op.input('X'),
        outputs=[flatten_x_name],
        axis=x_num_col_dims)
    flatten_y = helper.make_node(
        'Flatten',
        inputs=op.input('Y'),
        outputs=[flatten_y_name],
        axis=y_num_col_dims)
    shape_node = make_constant_node(shape_name, onnx_pb.TensorProto.INT64,
                                    out_shape)
    node = helper.make_node(
        'MatMul',
        inputs=[flatten_x_name, flatten_y_name],
        outputs=[temp_out_name])
    reshape_out = helper.make_node(
        'Reshape', inputs=[temp_out_name, shape_name], outputs=op.output('Out'))
    return [flatten_x, flatten_y, shape_node, node, reshape_out]


def matmul(op, block):
    x_shape = block.var(op.input('X')[0]).shape
    y_shape = block.var(op.input('Y')[0]).shape
    out_shape = list(block.var(op.output('Out')[0]).shape)
    node = helper.make_node(
        'MatMul',
        inputs=[op.input('X')[0], op.input('Y')[0]],
        outputs=op.output('Out'))
    return [node]


def batch_norm(op, block):
    kwargs = {'epsilon': op.attr('epsilon'), 'momentum': op.attr('momentum')}
    inputs = op.input('X') + op.input('Scale') + op.input('Bias') + op.input(
        'Mean') + op.input('Variance')
    node = helper.make_node(
        'BatchNormalization', inputs=inputs, outputs=op.output('Y'), **kwargs)
    return node


def instance_norm(op, block):
    kwargs = {'epsilon': op.attr('epsilon'), }
    inputs = op.input('X') + op.input('Scale') + op.input('Bias')
    node = helper.make_node(
        'InstanceNormalization',
        inputs=inputs,
        outputs=op.output('Y'),
        **kwargs)
    return node


def concat(op, block):
    node = helper.make_node(
        'Concat',
        inputs=op.input('X'),
        outputs=op.output('Out'),
        axis=op.attr('axis'))
    return node


def sum(op, block):
    node = helper.make_node(
        'Sum', inputs=op.input('X'), outputs=op.output('Out'))
    return node


def floor(op, block):
    node = helper.make_node(
        'Floor', inputs=op.input('X'), outputs=op.output('Out'))
    return node


def uniform_random_batch_size_like(op, block):
    node = helper.make_node(
        'RandomUniformLike',
        inputs=op.input('Input'),
        outputs=op.output('Out'),
        high=op.attr('max'),
        dtype=DTYPE_PADDLE_ONNX_MAP[op.attr('dtype')],
        low=op.attr('min'),
        seed=float(op.attr('seed')), )
    return node


def depthwise_conv2d(op, block):
    return conv2d(op, block)


def relu6(op, block):
    threshold = op.attr('threshold')
    node = helper.make_node(
        'Clip',
        inputs=[op.input('X')[0]],
        outputs=op.output('Out'),
        max=threshold,
        min=0.0)
    return [node]


def shape(op, block):
    node = helper.make_node(
        'Shape', inputs=op.input('Input'), outputs=op.output('Out'))
    return node


def split(op, block):
    sections = op.attr('sections')
    if len(sections) > 0:
        node = helper.make_node(
            'Split',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            axis=op.attr('axis'),
            split=sections)
    else:
        node = helper.make_node(
            'Split',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            axis=op.attr('axis'))
    return node


def slice(op, block):
    axes = op.attr('axes')
    starts = op.attr('starts')
    ends = op.attr('ends')
    node = helper.make_node(
        "Slice",
        inputs=[op.input('Input')[0]],
        outputs=op.output('Out'),
        axes=axes,
        starts=starts,
        ends=ends)
    return [node]


def fill_constant(op, block):
    value = op.attr('value')
    dtype = op.attr('dtype')
    shape = op.attr('shape')
    value = np.ones(shape) * value
    value = value.astype(DTYPE_ONNX_NUMPY_MAP[dtype])
    node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=op.output('Out'),
        value=helper.make_tensor(
            name=op.output('Out')[0],
            data_type=DTYPE_PADDLE_ONNX_MAP[dtype],
            dims=shape,
            vals=value.tolist()))
    return node


def transpose2(op, block):
    node = helper.make_node(
        'Transpose',
        inputs=op.input('X'),
        outputs=op.output('Out'),
        perm=op.attr('axis'))
    return node


def flatten2(op, block):
    node = helper.make_node(
        'Flatten',
        inputs=op.input('X'),
        outputs=op.output('Out'),
        axis=op.attr('axis'))
    return node


def reshape2(op, block):
    if len(op.input('ShapeTensor')) > 1:
        cast_shape_nodes = list()
        cast_shape_names = list()
        for i in range(len(op.input('ShapeTensor'))):
            dim = op.input('ShapeTensor')[i]
            temp_name = get_name(op.type, 'shape.cast')
            node = helper.make_node(
                'Cast',
                inputs=[dim],
                outputs=[temp_name],
                to=onnx_pb.TensorProto.INT64)
            cast_shape_nodes.append(node)
            cast_shape_names.append(temp_name)

        temp_name = get_name(op.type, 'shape.concat')
        shape_node = helper.make_node(
            'Concat', inputs=cast_shape_names, outputs=[temp_name], axis=-1)
        node = helper.make_node(
            'Reshape',
            inputs=[op.input('X')[0], temp_name],
            outputs=op.output('Out'))
        return cast_shape_nodes + [shape_node, node]
    elif len(op.input('ShapeTensor')) == 1:
        temp_name = get_name(op.type, 'shape.cast')
        cast_shape_node = helper.make_node(
            'Cast',
            inputs=op.input('ShapeTensor'),
            outputs=[temp_name],
            to=onnx_pb.TensorProto.INT64)
        node = helper.make_node(
            'Reshape',
            inputs=[op.input('X')[0], temp_name],
            outputs=op.output('Out'))
        return [cast_shape_node, node]
    elif op.attr('shape') is not None and len(op.attr('shape')) > 0:
        shape_name = get_name(op.type, 'shape')
        shape_node = make_constant_node(shape_name, onnx_pb.TensorProto.INT64,
                                        op.attr('shape'))
        reshape_node = helper.make_node(
            'Reshape',
            inputs=[op.input('X')[0], shape_name],
            outputs=op.output('Out'))
        return [shape_node, reshape_node]


def dropout(op, block):
    dropout_mode = op.attr('dropout_implementation')
    dropout_prob = op.attr('dropout_prob')
    if dropout_mode == 'upscale_in_train':
        node = helper.make_node(
            'Identity', inputs=op.input('X'), outputs=op.output('Out'))
        return node
    elif dropout_mode == 'downgrade_in_infer':
        scale_name = get_name(op.type, 'scale')
        scale_node = make_constant_node(scale_name, onnx_pb.TensorProto.FLOAT,
                                        1 - dropout_prob)
        node = helper.make_node(
            "Mul",
            inputs=[op.input('X')[0], scale_name],
            outputs=op.output('Out'))
        return [scale_node, node]
    else:
        raise Exception("Unexpected situation happend")


def reduce_mean(op, block):
    node = helper.make_node(
        'ReduceMean',
        inputs=op.input('X'),
        outputs=op.output('Out'),
        axes=op.attr('dim'),
        keepdims=op.attr('keep_dim'))
    return node


def hard_sigmoid(op, block):
    slope = op.attr('slope')
    offset = op.attr('offset')
    node = helper.make_node(
        'HardSigmoid',
        inputs=op.input('X'),
        outputs=op.output('Out'),
        alpha=slope,
        beta=offset)
    return node


def swish(op, block):
    beta = op.attr('beta')
    beta_name = get_name(op.type, 'beta')
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

    beta_x_name = get_name(op.type, 'beta_x')
    beta_x_node = onnx.helper.make_node(
        'Mul',
        name=beta_x_name,
        inputs=[op.input('X')[0], beta_name],
        outputs=[beta_x_name])
    sigmoid_name = get_name(op.type, 'sigmoid')
    sigmoid_node = onnx.helper.make_node(
        'Sigmoid',
        name=sigmoid_name,
        inputs=[beta_x_name],
        outputs=[sigmoid_name])
    swish_node = onnx.helper.make_node(
        'Mul',
        inputs=[op.input('X')[0], sigmoid_name],
        outputs=op.output('Out'))
    return [beta_node, beta_x_node, sigmoid_node, swish_node]


def hard_swish(op, block):
    scale_name = get_name(op.type, 'scale')
    offset_name = get_name(op.type, 'offset')
    scale_node = make_constant_node(scale_name, onnx_pb.TensorProto.FLOAT,
                                    op.attr('scale'))
    offset_node = make_constant_node(offset_name, onnx_pb.TensorProto.FLOAT,
                                     op.attr('offset'))

    name0 = get_name(op.type, 'add')
    node0 = helper.make_node(
        'Add', inputs=[op.input('X')[0], offset_name], outputs=[name0])
    name1 = get_name(op.type, 'relu')
    min_value = 0.0
    max_value = op.attr('threshold')
    node1 = helper.make_node(
        'Clip', inputs=[name0], outputs=[name1], max=max_value, min=min_value)
    name2 = get_name(op.type, 'mul')
    node2 = helper.make_node(
        'Mul', inputs=[op.input('X')[0], name1], outputs=[name2])
    node3 = helper.make_node(
        'Div', inputs=[name2, scale_name], outputs=op.output('Out'))
    return [scale_node, offset_node, node0, node1, node2, node3]


#def feed(op, block):
#    name = op.output('Out')[0]
#    var = block.var(name)
#    tensor_info = helper.make_tensor_value_info(
#        name=name,
#        shape=var.shape,
#        elem_type=DTYPE_PADDLE_ONNX_MAP[var.dtype])
#    return tensor_info
#
#def fetch(op, block):
#    name = op.input('X')[0]
#    var = block.var(name)
#    tensor_info = helper.make_tensor_value_info(
#        name=name,
#        shape=var.shape,
#        elem_type=DTYPE_PADDLE_ONNX_MAP[var.dtype])
#    return tensor_info


def feed(var, block=None):
    tensor_info = helper.make_tensor_value_info(
        name=var.name,
        shape=var.shape,
        elem_type=DTYPE_PADDLE_ONNX_MAP[var.dtype])
    return tensor_info


def fetch(var, block=None):
    tensor_info = helper.make_tensor_value_info(
        name=var.name,
        shape=var.shape,
        elem_type=DTYPE_PADDLE_ONNX_MAP[var.dtype])
    return tensor_info


def unsqueeze2(op, block):
    node = helper.make_node(
        'Unsqueeze',
        inputs=op.input('X'),
        outputs=op.output('Out'),
        axes=op.attr('axes'))
    return node


def cast(op, block):
    node = helper.make_node(
        'Cast',
        inputs=op.input('X'),
        outputs=op.output('Out'),
        to=DTYPE_PADDLE_ONNX_MAP[op.attr('out_dtype')])
    return node


def arg_max(op, block):
    node = helper.make_node(
        'ArgMax',
        inputs=op.input('X'),
        outputs=op.output('Out'),
        axis=op.attr('axis'),
        keepdims=0)
    return node


def reciprocal(op, block):
    inputs = op.input(op.input_names[0])
    outputs = op.output(op.output_names[0])
    node = helper.make_node('Reciprocal', inputs=inputs, outputs=outputs)
    return node


def im2sequence(op, block):
    from .paddle_custom_layer.im2sequence import im2sequence
    return im2sequence(op, block)


def yolo_box(op, block):
    from .paddle_custom_layer.yolo_box import yolo_box
    return yolo_box(op, block)


def multiclass_nms(op, block):
    from .paddle_custom_layer.multiclass_nms import multiclass_nms
    return multiclass_nms(op, block)


def box_coder(op, block):
    from .paddle_custom_layer.box_coder import box_coder
    return box_coder(op, block)


def prior_box(op, block):
    from .paddle_custom_layer.prior_box import prior_box
    return prior_box(op, block)
