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
import os
import pickle
import warnings
import inspect
import copy
import collections

import numpy as np
import paddle
import paddle.jit as jit
import paddle.fluid.core as core
import paddle.fluid.dygraph.base as base
import paddle.fluid.dygraph.dygraph_to_static.program_translator as program_translator
import paddle.fluid.dygraph.layers as layers
import paddle.fluid.dygraph.io as io
from paddle.fluid.framework import Variable
import paddle.fluid as fluid
from paddle.fluid.executor import global_scope


def prepend_feed_ops(inference_program,
                     feed_target_names,
                     feed_holder_name='feed'):
    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True)

    for i, name in enumerate(feed_target_names):
        if not global_block.has_var(name):
            raise ValueError(
                "The feeded_var_names[{i}]: '{name}' doesn't exist in pruned inference program. "
                "Please check whether '{name}' is a valid feed_var name, or remove it from feeded_var_names "
                "if '{name}' is not involved in the target_vars calculation.".
                format(
                    i=i, name=name))
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i})


def append_fetch_ops(inference_program,
                     fetch_target_names,
                     fetch_holder_name='fetch'):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)

    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})


def prune_input_output(concrete_program, input_spec, output_spec):
    feeded_vars, feeded_var_names = get_inout_spec(concrete_program.inputs,
                                                   input_spec, True)
    target_vars = get_inout_spec(concrete_program.outputs, output_spec)
    main_program = concrete_program.main_program.clone()
    global_block = main_program.global_block()
    need_to_remove_op_index = []

    for i, op in enumerate(global_block.ops):
        op.desc.set_is_target(False)
        if op.type == "feed" or op.type == "fetch":
            need_to_remove_op_index.append(i)

    for index in need_to_remove_op_index[::-1]:
        global_block._remove_op(index)

    main_program.desc.flush()

    main_program = main_program._prune_with_input(
        feeded_var_names=feeded_var_names, targets=target_vars)
    main_program = main_program._inference_optimize(prune_read_op=True)
    fetch_var_names = [v.name for v in target_vars]

    prepend_feed_ops(main_program, feeded_var_names)
    append_fetch_ops(main_program, fetch_var_names)

    concrete_program.outputs = tuple(target_vars)
    concrete_program.inputs = tuple(feeded_vars)
    concrete_program.main_program = main_program

    return concrete_program


@base.switch_to_static_graph
def get_concrete_program(layer):
    jit.set_verbosity(10)
    if isinstance(layer, layers.Layer):
        if isinstance(layer.forward, program_translator.StaticLayer):
            return layer.forward.concrete_program
        else:
            raise TypeError(
                "The foward of layer should be StaticLayer, but received forward type is %s."
                % type(layer.forward))
    elif isinstance(layer, program_translator.StaticLayer):
        return layer.concrete_program
    else:
        raise TypeError(
            "The input Layer should be 'Layer' or 'StaticLayer', but received  type is %s."
            % type(layer))


def get_inout_spec(all_vars, target_vars, return_name=False):
    result_list = []
    valid_var_dict = {}
    valid_vars = [var for var in all_vars if isinstance(var, Variable)]
    for var in valid_vars:
        valid_var_dict[var.name] = var
    if target_vars is not None:
        for i, var in enumerate(target_vars):
            # check target var whether exists
            if var.name not in valid_var_dict:
                raise RuntimeError("The variable to feed/fetch are not exist.")
            result_list.append(valid_var_dict[var.name])
    else:
        result_list = valid_vars
    if return_name:
        return result_list, [var.name for var in result_list]
    return result_list


class Node(object):
    def __init__(self, op_type, layer_name, inputs, outputs, attrs, block):
        self.block = block
        self.inputs = inputs
        self.outputs = outputs
        self.type = op_type
        self.attrs = attrs
        self.layer_name = layer_name

    def __hash__(self):
        return hash(self.layer_name)

    def __eq__(self, other):
        if self.layer_name == other.layer_name:
            return True
        return False

    def input(self, name, idx=None):
        if idx is None:
            return self.inputs(name)
        return self.inputs(name)[idx]

    @property
    def input_names(self):
        return [name for name in self.inputs]

    @property
    def output_names(self):
        return [name for name in self.outputs]

    def output(self, name, idx=None):
        if idx is None:
            return self.outputs(name)
        return self.outputs(name)[idx]

    def output_shape(self, arg_name, idx):
        return self.block.var(self.output(arg_name)[idx]).shape

    def input_shape(self, arg_name, idx):
        return self.block.var(self.input(arg_name)[idx]).shape

    def input_var(self, arg_name, idx):
        return self.block.var(self.input(arg_name)[idx])

    def attr(self, name):
        return self.attrs[name]


class Graph(object):
    def __init__(self, program, parameters):
        self.parameters = parameters
        self.node_map = collections.OrderedDict()
        self.input_nodes = list()
        self.output_nodes = list()
        self.topo_sort = list()
        self.op_type_count = {}
        self.build(program)

    def make_name(self, op_type):
        if op_type in self.op_type_count:
            self.op_type_count[op_type] += 1
        else:
            self.op_type_count[op_type] = 1
        layer_name = op_type + '@' + str(self.op_type_count[op_type])
        return layer_name

    def make_node(self, op_type, layer_name, inputs, outputs, attr, block):
        node = Node(op_type, layer_name, inputs, outputs, attr, block)
        self.node_map[layer_name] = node
        return node

    def build(self, program):
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type == 'feed':
                    layer_name = op.output('Out')[0]
                    var = block.var(layer_name)
                    attrs = {}
                    attrs['shape'] = var.shape
                    attrs['dtype'] = var.dtype
                    node = self.make_node(op.type, layer_name, None, None,
                                          attrs, block)
                    self.input_nodes.append(node)
                elif op.type == 'fetch':
                    layer_name = op.input('X')[0]
                    var = block.var(layer_name)
                    attrs = {}
                    attrs['shape'] = var.shape
                    attrs['dtype'] = var.dtype
                    node = self.make_node(op.type, layer_name, None, None,
                                          attrs, block)
                    self.output_nodes.append(node)
                else:
                    layer_name = self.make_name(op.type)
                    node = self.make_node(op.type, layer_name, op.input,
                                          op.output, op.all_attrs(), block)
                    self.topo_sort.append(node)

    def get_node(self, name, copy=False):
        if copy:
            node = cp.copy(self.node_map[name])
        else:
            node = self.node_map[name]
        return node

    @staticmethod
    @base.switch_to_static_graph
    def parse_graph(layer, input_spec=None, output_spec=None):
        jit.set_verbosity(2)
        if isinstance(layer, io.TranslatedLayer):
            program = layer.program()
            parameters = layer.parameters()
            return StaticGraph(None, None, None, None)
        elif isinstance(layer, layers.Layer) or isinstance(
                layer, program_translator.StaticLayer):
            concrete_program = get_concrete_program(layer)
            concrete_program = prune_input_output(concrete_program, input_spec,
                                                  output_spec)
            program = concrete_program.main_program
            parameters = {}

            #for param in concrete_program.parameters:
            #    if param.name.endswith('feed') or param.name.endswith('fetch'):
            #        continue
            #    if not param.persistable:
            #        continue
            #    parameters[param.name] = {
            #        'tensor': param.value().get_tensor(),
            #        'dtype': param.dtype,
            #        'shape': param.shape
            #    }

            var_names = program.global_block().vars
            for name in var_names:
                var = program.global_block().var(name)
                if name.endswith('feed') or name.endswith('fetch'):
                    continue
                if not var.persistable:
                    continue
                parameters[name] = {
                    'tensor': core.Scope().var(name).get_tensor(),
                    'dtype': var.dtype,
                    'shape': var.shape
                }
            graph = Graph(program, parameters)
            return graph
        else:
            raise TypeError(
                "The input Layer should be 'Layer' or 'StaticLayer', 'TranslatedLayer', but received  type is %s."
                % type(layer))

    @staticmethod
    def parse_program(program, feed=None, fetch=None, scope=None):
        parameters = {}
        var_names = program.global_block().vars

        #for var in program.list_vars():
        #    print(var.name)
        #    if var.name.endswith('feed') or var.name.endswith('fetch'):
        #        continue
        #    if not var.persistable:
        #        continue
        #    parameters[var.name] = {
        #        'tensor': var, #scope.find_var(name).get_tensor(),
        #        'dtype': var.dtype,
        #        'shape': var.shape
        #    }

        for name in var_names:
            var = program.global_block().var(name)
            #print(var)
            if name.endswith('feed') or name.endswith('fetch'):
                continue
            if not var.persistable:
                continue
            parameters[name] = {
                'tensor': scope.var(name).get_tensor(),
                'dtype': var.dtype,
                'shape': var.shape
            }

        graph = Graph(program, parameters)
        return graph
