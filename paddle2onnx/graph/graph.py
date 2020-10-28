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
import copy
import collections
from paddle2onnx.constant import NodeDomain
from paddle2onnx.graph.node import PaddleNode, ONNXNode, Node


class Graph(object):
    """ Graph IR between Paddle and ONNX.  
    Args:
        id (int): the id of graph.
    """

    def __init__(self, id, opset_version=None, block=None):
        self.id = id
        self.opset_version = opset_version
        self.parameters = {}
        self.node_map = collections.OrderedDict()
        self.topo_sort = list()
        self.input_nodes = list()
        self.output_nodes = list()
        self.edge_map = dict()
        self.op_type_count = dict()
        self.sub_graphs = list()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False

    def __str__(self):
        graph_str = 'graph { \n'
        for node in self.input_nodes:
            graph_str += " input: {} \n".format(node.layer_name)
        for node in self.output_nodes:
            graph_str += " output: {} \n \n".format(node.layer_name)
        for name, node in self.node_map.items():
            graph_str += node.__str__()
        graph_str += ' }'
        return graph_str

    def set_output_nodes(self, node_list):
        if isinstance(node_list, list):
            self.output_nodes = node_list
        else:
            raise TypeError(
                'output_nodes of Graph must be type: list, but got {}'.format(
                    type(node_list)))

    def set_node_map(self, node_map):
        if isinstance(node_map, dict):
            self.node_map = node_map
            self.generate_topo_sort()
        else:
            raise TypeError('node_map of Graph must be type: list, but got {}'.
                            format(type(node_map)))

    def set_input_nodes(self, node_list):
        if isinstance(node_list, list):
            self.input_nodes = node_list
        else:
            raise TypeError(
                'input_nodes of Graph must be type: list, but got {}'.format(
                    type(node_list)))

    def set_parameters(self, parameters):
        if isinstance(parameters, dict):
            self.parameters = parameters
        else:
            raise TypeError(
                'parameters of Graph must be type: dict, but got {}'.format(
                    type(parameters)))

    def generate_node_name(self, op_type):
        if op_type in self.op_type_count:
            self.op_type_count[op_type] += 1
        else:
            self.op_type_count[op_type] = 1
        # layer_name need follow https://github.com/onnx/onnx/blob/master/docs/OpConventions.md
        layer_name = op_type + '@block_' + str(self.id) + '@' + str(
            self.op_type_count[op_type] - 1)
        return layer_name

    def generate_edge_map(self):
        for layer_name, node in self.node_map.items():
            inputs = None
            if isinstance(node.inputs, dict):
                inputs = node.inputs.values()
                inputs = [x for j in inputs for x in j]
            elif isinstance(node.inputs, list):
                inputs = node.inputs
            for ipt in inputs:
                for layer_name, ipt_node in self.node_map.items():
                    if isinstance(ipt_node.outputs, dict):
                        outputs = ipt_node.outputs.values()
                        outputs = [x for j in outputs for x in j]
                    elif isinstance(ipt_node.outputs, list):
                        outputs = ipt_node.outputs
                    if ipt in outputs:
                        if ipt_node not in self.edge_map:
                            self.edge_map[ipt_node] = [node]
                        else:
                            self.edge_map[ipt_node].append(node)

    def generate_topo_sort(self):
        self.generate_edge_map()
        input_node_names = [node.layer_name for node in self.input_nodes]
        for layer_name, node in self.node_map.items():
            if node not in self.edge_map:
                self.topo_sort.append(node)
        for current_node in self.topo_sort:
            if isinstance(current_node.outputs, dict):
                outputs = current_node.outputs.values()
                outputs = [x for j in outputs for x in j]
            elif isinstance(current_node.outputs, list):
                outputs = current.outputs
            for layer_name, node in self.node_map.items():
                if isinstance(node.inputs, dict):
                    inputs = node.inputs.values()
                    inputs = [x for j in inputs for x in j]
                elif isinstance(node.inputs, list):
                    inputs = node.inputs
                for ipt in inputs:
                    if ipt in outputs:
                        self.topo_sort.append(node)

    def insert_node(self, node):
        if node.type not in ['feed', 'fetch']:
            self.node_map[node.layer_name] = node

    def make_node(self,
                  op_type,
                  inputs=None,
                  outputs=None,
                  attrs=None,
                  layer_name=None,
                  domain=None,
                  **kw):
        if layer_name is None:
            layer_name = self.generate_node_name(op_type)

        if attrs is None:
            attrs = kw
        attrs.update(kw)

        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = [layer_name]
        node = Node(op_type, layer_name, inputs, outputs, attrs, domain)
        self.insert_node(node)
        return node

    def make_onnx_node(self,
                       op_type,
                       inputs=None,
                       outputs=None,
                       attrs=None,
                       layer_name=None,
                       **kw):
        domain = NodeDomain.ONNX
        if layer_name is None:
            layer_name = self.generate_node_name(op_type)

        if attrs is None:
            attrs = kw
        attrs.update(kw)

        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = [layer_name]
        node = ONNXNode(op_type, layer_name, inputs, outputs, attrs, domain)
        self.insert_node(node)
        return node

    def make_paddle_node(self,
                         op,
                         inputs=None,
                         outputs=None,
                         attrs=None,
                         block=None,
                         layer_name=None,
                         **kw):
        domain = NodeDomain.PADDLE
        if layer_name is None:
            layer_name = self.generate_node_name(op.type)

        if attrs is None:
            attrs = kw
        attrs.update(kw)

        if inputs is None:
            inputs = {}
        if outputs is None:
            outputs = {'Out': layer_name}
        node = PaddleNode(op, layer_name, inputs, outputs, attrs, block, domain)
        self.insert_node(node)
        return node

    def update_node(self,
                    node,
                    op_type=None,
                    inputs=None,
                    outputs=None,
                    attrs=None,
                    block=None,
                    move_to_end=True,
                    domain=None,
                    **kw):
        node.type = op_type
        if inputs is not None:
            node.set_inputs(inputs)
        if outputs is not None:
            node.set_outputs(outputs)
        if attrs is None:
            attrs = kw
        attrs.update(kw)
        node.attrs = attrs
        if domain is not None:
            node.domain = domain
        if move_to_end:
            self.node_map.pop(node.layer_name)
            self.node_map[node.layer_name] = node
        return node

    def get_node(self, name, copy=False):
        if name not in self.node_map:
            raise TypeError('Node with name:{} not in graph'.format(name))
        if copy:
            node = copy.copy(self.node_map[name])
        else:
            node = self.node_map[name]
        return node

    def remove_node_by_name(self, name):
        if name in self.node_map:
            node = self.node_map.pop(name)
            return node
        raise TypeError('Node with name:{} not in graph'.format(name))

    def remove_node(self, node):
        if isinstance(node, Node):
            node = self.remove_node_by_name(node.layer_name)
            return node
        elif isinstance(node, str):
            node = self.remove_node_by_name(node)
            return node
        else:
            raise TypeError(
                'Remove node by str or Node, but got type: {}'.format(node))

    def get_output_nodes_of_node(self, node):
        if node in self.edge_map:
            return self.edge_map[node]
        elif self.get_node(node.layer_name, copy=False):
            return []
        else:
            raise KeyError('Node with layer_name {} not in graph.egde_map'.
                           format(node.layer_name))
