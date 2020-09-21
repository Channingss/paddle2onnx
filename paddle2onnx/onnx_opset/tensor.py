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


@op_mapper('concat')
class Concat():
    @classmethod
    def opset_9(cls, node, **kw):
        node = helper.make_node(
            'Concat',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            axis=node.attr('axis'))
        return node


@op_mapper('shape')
class Shape():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'Shape', inputs=node.input('Input'), outputs=node.output('Out'))
        return onnx_node


@op_mapper('split')
class Split():
    @classmethod
    def opset_9(cls, node, **kw):
        sections = node.attr('sections')
        if len(sections) > 0:
            onnx_node = helper.make_node(
                'Split',
                inputs=node.input('X'),
                outputs=node.output('Out'),
                axis=node.attr('axis'),
                split=sections)
        else:
            onnx_node = helper.make_node(
                'Split',
                inputs=node.input('X'),
                outputs=node.output('Out'),
                axis=node.attr('axis'))
        return onnx_node


@op_mapper('slice')
class Slice():
    @classmethod
    def opset_9(cls, node, **kw):
        axes = node.attr('axes')
        starts = node.attr('starts')
        ends = node.attr('ends')
        onnx_node = helper.make_node(
            "Slice",
            inputs=[node.input('Input')[0]],
            outputs=node.output('Out'),
            axes=axes,
            starts=starts,
            ends=ends)
        return onnx_node

    @classmethod
    def opset_10(cls, node, **kw):
        axes = node.attr('axes')
        starts = node.attr('starts')
        ends = node.attr('ends')
        axes_name = get_name(node.type, 'axes')
        starts_name = get_name(node.type, 'starts')
        ends_name = get_name(node.type, 'ends')

        axes_node = make_constant_node(axes_name, onnx_pb.TensorProto.INT64,
                                       axes)
        starts_node = make_constant_node(starts_name, onnx_pb.TensorProto.INT64,
                                         starts)
        ends_node = make_constant_node(ends_name, onnx_pb.TensorProto.INT64,
                                       ends)
        onnx_node = helper.make_node(
            "Slice",
            inputs=[node.input('Input')[0], starts_name, ends_name, axes_name],
            outputs=node.output('Out'), )
        return [starts_node, ends_node, axes_node, onnx_node]


@op_mapper('fill_constant')
class Constant():
    @classmethod
    def opset_9(cls, node, **kw):
        value = node.attr('value')
        print(value)
        dtype = node.attr('dtype')
        shape = node.attr('shape')
        value = np.ones(shape) * value
        value = value.astype(DTYPE_ONNX_NUMPY_MAP[dtype])
        onnx_node = helper.make_node(
            'Constant',
            inputs=[],
            outputs=node.output('Out'),
            value=helper.make_tensor(
                name=node.output('Out')[0],
                data_type=DTYPE_PADDLE_ONNX_MAP[dtype],
                dims=shape,
                vals=value.tolist()))
        return onnx_node


@op_mapper('transpose2')
class Transpose():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'Transpose',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            perm=node.attr('axis'))
        return onnx_node


@op_mapper('flatten2')
class Flatten():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'Flatten',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            axis=node.attr('axis'))
        return onnx_node


@op_mapper('reshape2')
class Reshape():
    @classmethod
    def opset_9(cls, node, **kw):
        if len(node.input('ShapeTensor')) > 1:
            cast_shape_nodes = list()
            cast_shape_names = list()
            for i in range(len(node.input('ShapeTensor'))):
                dim = node.input('ShapeTensor')[i]
                temp_name = get_name(node.type, 'shape.cast')
                cast_node = helper.make_node(
                    'Cast',
                    inputs=[dim],
                    outputs=[temp_name],
                    to=onnx_pb.TensorProto.INT64)
                cast_shape_nodes.append(cast_node)
                cast_shape_names.append(temp_name)

            temp_name = get_name(node.type, 'shape.concat')
            shape_node = helper.make_node(
                'Concat', inputs=cast_shape_names, outputs=[temp_name], axis=-1)
            onnx_node = helper.make_node(
                'Reshape',
                inputs=[node.input('X')[0], temp_name],
                outputs=node.output('Out'))
            return cast_shape_nodes + [shape_node, onnx_node]
        elif len(node.input('ShapeTensor')) == 1:
            temp_name = get_name(onnx.type, 'shape.cast')
            cast_shape_node = helper.make_node(
                'Cast',
                inputs=onnx.input('ShapeTensor'),
                outputs=[temp_name],
                to=onnx_pb.TensorProto.INT64)
            onnx_node = helper.make_node(
                'Reshape',
                inputs=[node.input('X')[0], temp_name],
                outputs=node.output('Out'))
            return [cast_shape_node, onnx_node]
        elif node.attr('shape') is not None and len(node.attr('shape')) > 0:
            shape_name = get_name(node.type, 'shape')
            shape_node = make_constant_node(shape_name,
                                            onnx_pb.TensorProto.INT64,
                                            node.attr('shape'))
            reshape_node = helper.make_node(
                'Reshape',
                inputs=[node.input('X')[0], shape_name],
                outputs=node.output('Out'))
            return [shape_node, reshape_node]


@op_mapper('unsqueeze2')
class Unsqueeze():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'Unsqueeze',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            axes=node.attr('axes'))
        return onnx_node


@op_mapper('reciprocal')
class Reciprocal():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'Reciprocal', inputs=node.input('X'), outputs=node.output('Out'))
        return onnx_node


@op_mapper('cast')
class Cast():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'Cast',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            to=DTYPE_PADDLE_ONNX_MAP[node.attr('out_dtype')])
        return onnx_node


@op_mapper('clip')
class Clip():
    @classmethod
    def opset_9(cls, node, **kw):
        min_value = node.attr('min')
        max_value = node.attr('max')
        onnx_node = helper.make_node(
            'Clip',
            inputs=[node.input('X')[0]],
            outputs=node.output('Out'),
            max=max_value,
            min=min_value)
        return onnx_node

    @classmethod
    def opset_9(cls, node, **kw):
        min_name = get_name(node.type, 'min')
        max_name = get_name(node.type, 'max')
        min_node = make_constant_node(min_name, onnx_pb.TensorProto.FLOAT,
                                      node.attr('min'))
        max_node = make_constant_node(max_name, onnx_pb.TensorProto.FLOAT,
                                      node.attr('max'))
        node = helper.make_node(
            'Clip',
            inputs=[node.input('X')[0], min_name, max_name],
            outputs=node.output('Out'))
        return [min_node, max_node, node]


@op_mapper('pad2d')
class Pad():
    @classmethod
    def opset_9(cls, node, **kw):
        pads = convert_padding(cls, node, **kw)
        onnx_node = helper.make_node(
            'Pad',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            mode=node.attr('mode'),
            value=node.attr('pad_value'),
            pads=pads)
        return onnx_node

    @classmethod
    def opset_11(cls, node, **kw):
        pads = convert_padding(cls, node, **kw)
        pads_name = get_name(node.type, 'pads')
        pads_node = make_constant_node(pads_name, onnx_pb.TensorProto.INT64,
                                       pads)
        constant_value_name = get_name(node.type, 'constant_value')
        constant_value_node = make_constant_node(constant_value_name,
                                                 onnx_pb.TensorProto.FLOAT,
                                                 node.attr('pad_value'))
        onnx_node = helper.make_node(
            'Pad',
            inputs=node.input('X') + [pads_name, constant_value_name],
            outputs=node.output('Out'),
            mode=node.attr('mode'))
        return [pads_node, constant_value_node, onnx_node]

    @classmethod
    def convert_padding(cls, node, **kw):
        x_shape = node.input_shape('X', 0)
        paddings = node.attr('paddings')
        #TODO support pads is Variable
        if node.attr('data_format') == 'NCHW':
            return [
                0, 0, paddings[0], paddings[2], 0, 0, paddings[1], paddings[3]
            ]
        else:
            return [
                0, paddings[0], paddings[2], 0, 0, paddings[1], paddings[3], 0
            ]


@op_mapper('uniform_random_batch_size_like')
class UniformRandom():
    @classmethod
    def opset_9(cls, node, **kw):
        onnx_node = helper.make_node(
            'RandomUniformLike',
            inputs=node.input('Input'),
            outputs=node.output('Out'),
            high=node.attr('max'),
            dtype=DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')],
            low=node.attr('min'),
            seed=float(node.attr('seed')), )
        return onnx_node


@op_mapper(
    ['bilinear_interp', 'nearest_interp'],
    mapper_dict={'bilinear_interp': 'linear',
                 'nearest_interp': 'nearest'})
class Resize():
    node_lists = []

    @classmethod
    def opset_10(cls, node, **kw):
        if node.attr('align_corners') or node.attr('align_mode') == 0:
            raise Exception(
                "Resize in onnx(opset<=10) only support coordinate_transformation_mode: 'asymmetric', Try converting with --onnx_opset 11"
            )
        scale = cls.convert_scale(cls, node, **kw)
        onnx_node = helper.make_node(
            'Resize',
            inputs=[node.input('X')[0], scale],
            outputs=node.output('Out'),
            mode=resize_type)
        cls.node_lists.append(onnx_node)
        return cls.node_lists

    def opset_11(cls, node, **kw):
        coordinate_transformation_mode = ''
        if node.attr('align_corners'):
            coordinate_transformation_mode = 'align_corners'
        elif node.type == 'nearest_interp':
            coordinate_transformation_mode = 'half_pixel'
        else:
            if node.type('align_mode') == 1:
                coordinate_transformation_mode = 'asymmetric'
            else:
                coordinate_transformation_mode = 'half_pixel'
        roi_name = get_name(node.type, 'roi')
        roi_node = make_constant_node(roi_name, onnx_pb.TensorProto.FLOAT,
                                      [1, 1, 1, 1, 1, 1, 1, 1])
        inputs = [node.input('X')[0], roi_name]
        if ('OutSize' in node.input_names and
                len(node.input('OutSize')) > 0) or (
                    'SizeTensor' in node.input_names and
                    len(node.input('SizeTensor')) > 0):
            empty_name = get_name(node.type, 'empty')
            empty_tensor = helper.make_tensor(
                empty_name,
                onnx_pb.TensorProto.FLOAT, (0, ),
                np.array([]).astype('float32'),
                raw=False)
            empty_node = helper.make_node(
                'Constant', [], outputs=[empty_name], value=empty_tensor)
            inputs.append(empty_name)

        inputs.append(scale)
        onnx_node = helper.make_node(
            'Resize',
            inputs=inputs,
            outputs=node.output('Out'),
            mode=resize_type,
            coordinate_transformation_mode=coordinate_transformation_mode)
        cls.node_lists.append(onnx_node)
        return cls.node_lists

    @classmethod
    def convert_scale(cls, node, **kw):
        resize_type = kw['mapper_dict'][node.type]
        input_shape = node.input_shape('X', 0)
        if ('OutSize' in node.input_names and
                len(node.input('OutSize')) > 0) or (
                    'SizeTensor' in node.input_names and
                    len(node.input('SizeTensor')) > 0):
            shape_name0 = get_name(node.type, 'shape')
            shape_node0 = helper.make_node(
                'Shape', inputs=node.input('X'), outputs=[shape_name0])
            starts_name = get_name(node.type, 'slice.starts')
            starts_node = make_constant_node(starts_name,
                                             onnx_pb.TensorProto.INT64, [0])
            ends_name = get_name(node.type, 'slice.ends')
            ends_node = make_constant_node(ends_name, onnx_pb.TensorProto.INT64,
                                           [2])
            shape_name1 = get_name(node.type, 'shape')
            shape_node1 = helper.make_node(
                'Slice',
                inputs=[shape_name0, starts_name, ends_name],
                outputs=[shape_name1])
            cls.node_list.extend(
                [shape_node0, starts_node, ends_node, shape_node1])
            if 'OutSize' in node.input_names and len(node.input('OutSize')) > 0:
                cast_shape_name = get_name(node.type, "shape.cast")
                cast_shape_node = helper.make_node(
                    'Cast',
                    inputs=node.input('OutSize'),
                    outputs=[cast_shape_name],
                    to=onnx_pb.TensorProto.INT64)
                cls.node_list.append(cast_shape_node)
            else:
                concat_shape_name = get_name(
                    node.type, node.output('Out')[0] + "shape.concat")
                concat_shape_node = helper.make_node(
                    "Concat",
                    inputs=node.input('SizeTensor'),
                    outputs=[concat_shape_name],
                    axis=0)
                cast_shape_name = get_name(node.type, "shape.cast")
                cast_shape_node = helper.make_node(
                    'Cast',
                    inputs=[concat_shape_name],
                    outputs=[cast_shape_name],
                    to=onnx_pb.TensorProto.INT64)
                cls.node_list.extend([concat_shape_node, cast_shape_node])
            shape_name2 = get_name(node.type, "shape.concat")
            shape_node2 = helper.make_node(
                'Concat',
                inputs=[shape_name1, cast_shape_name],
                outputs=[shape_name2],
                axis=0)
            cls.node_list.append(shape_node2)
            cast_shape_name2 = get_name(node.type, "shape.cast")
            cast_shape_node2 = helper.make_node(
                'Cast',
                inputs=[shape_name2],
                outputs=[cast_shape_name2],
                to=onnx_pb.TensorProto.FLOAT)
            cls.node_list.append(cast_shape_node2)
            cast_shape_name0 = get_name(node.type, "shape.cast")
            cast_shape_node0 = helper.make_node(
                'Cast',
                inputs=[shape_name0],
                outputs=[cast_shape_name0],
                to=onnx_pb.TensorProto.FLOAT)
            cls.node_list.append(cast_shape_node0)
            outputs_h_w_scales = node.output('Out')[0] + "@out_hw_scales"
            node_h_w_scales = helper.make_node(
                'Div',
                inputs=[cast_shape_name2, cast_shape_name0],
                outputs=[outputs_h_w_scales])
            cls.node_list.append(node_h_w_scales)
            return outputs_h_w_scales
        elif 'Scale' in input_names and len(node.input('Scale')) > 0:
            return node.input('Scale')[0]
        else:
            out_shape = [node.attr('out_h'), node.attr('out_w')]
            scale = node.attr('scale')
            if out_shape.count(-1) > 0:
                scale_name = get_name(node.type, 'scale')
                scale_node = make_constant_node(scale_name,
                                                onnx_pb.TensorProto.FLOAT,
                                                [1, 1, scale, scale])
                cls.node_list.append(scale_node)
                return scale_name
            else:
                raise Exception("Unexpected situation happend")
