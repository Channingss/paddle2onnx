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

import math
import sys
import paddle2onnx
import os
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
from onnx import helper, onnx_pb
from paddle.fluid.dygraph.base import program_desc_tracing_guard, switch_to_static_graph
from .utils import DTYPE_MAP

class Converter(object):
    def __init__(self):
        self.support_opsets = [9, 10, 11]
        self.default_opset = 9
        self.name_counter = dict()
        self.op_set = None

    def convert_weights(self, concrete_program):
        nodes = list()
        for param in concrete_program.parameters:
            if param.name.endswith('feed') or param.name.endswith('fetch'):
                continue
            if not param.persistable:
                continue
            weight = np.array(param.value().get_tensor())
            tensor = helper.make_tensor(
                name=param.name,
                dims=param.shape,
                data_type=DTYPE_MAP[param.dtype],
                vals=weight.flatten().tolist())
            node = helper.make_node(
                'Constant', inputs=[], outputs=[param.name], value=tensor)
            nodes.append(node)
        return nodes

    def convert_inputs(self, concrete_program):
        input_nodes = []
        for ipt in concrete_program.inputs:
            if isinstance(ipt, fluid.Variable):
                input_nodes.append(getattr(self.op_set, 'feed')(ipt))
            if isinstance(ipt, dict):
                for key, var in ipt.items():
                    input_nodes.append(getattr(self.op_set, 'feed')(var))
        return input_nodes 

    def convert_outputs(self, concrete_program):
        output_nodes = []
        for opt in concrete_program.outputs:
            if isinstance(opt, fluid.Variable):
                output_nodes.append(getattr(self.op_set, 'fetch')(opt))
        return output_nodes

    def convert_ops(self, concrete_program):
        op_nodes = list()
        unsupported_ops = set()
        for block in concrete_program.main_program.blocks:
            for i, op in enumerate(block.ops):
                sys.stdout.write("\rTotal:{}, Current:{} : {} ".format(
                    len(block.ops), i + 1, op.type))
                sys.stdout.flush()
                if not hasattr(self.op_set, op.type):
                    unsupported_ops.add(op.type)
                    continue
                if len(unsupported_ops) > 0:
                    continue
                if op.type in ['feed', 'fetch']:
                    continue
                node = getattr(self.op_set, op.type)(op, block)
                if isinstance(node, list):
                    op_nodes = op_nodes + node
                else:
                    op_nodes.append(node)
        if len(unsupported_ops) > 0:
            unsupported_ops_string = "\nThere's {} ops are not supported yet\n".format(
                len(unsupported_ops))
            for op in unsupported_ops:
                unsupported_ops_string += "=========== {} ===========\n".format(op)
            raise ValueError(unsupported_ops_string)
        return op_nodes

    @switch_to_static_graph
    def convert(self, concrete_program, save_dir, opset_version=9):
        program = concrete_program.main_program.clone()
        self.op_set = self.import_ops_with_opset_version(opset_version)

        print("Translating PaddlePaddle to ONNX...\n")
        input_nodes = self.convert_inputs(concrete_program)
        output_nodes = self.convert_outputs(concrete_program)
        weight_nodes = self.convert_weights(concrete_program)
        op_nodes = self.convert_ops(concrete_program)

        graph = helper.make_graph(
            nodes=weight_nodes + op_nodes,
            name='onnx_model_from_paddle',
            initializer=[],
            inputs=input_nodes,
            outputs=output_nodes)
        opset_imports = [helper.make_opsetid("", opset_version)]
        model = helper.make_model(
            graph, producer_name='X2Paddle', opset_imports=opset_imports)
        onnx.checker.check_model(model)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'paddle2onnx_model.onnx'), 'wb') as f:
            f.write(model.SerializeToString())
        print("\nTranslated model saved in {}".format(
            os.path.join(save_dir, 'paddle2onnx_model.onnx')))

    def import_ops_with_opset_version(self, opset_version=9):
        run_opset = self.default_opset
        opset = ''
        if opset_version in self.support_opsets:
            run_opset = opset_version
        else:
            for support_opset_version in self.support_opsets:
                if support_opset_version < opset_version:
                    run_opset = support_opset_version
                else:
                    break
        print(
            'Now, onnpaddle2onnx support convert onnx model opset_verison {},'
            'opset_verison of your onnx model is {}, automatically treated as op_set: {}.'
            .format(self.support_opsets, opset_version, run_opset))
        opset = 'opset' + str(run_opset)
        import importlib
        ops_module = importlib.import_module('.opset', package='paddle2onnx.paddle2onnx.converter.'+opset)
        return ops_module
