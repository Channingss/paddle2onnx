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
from paddle2onnx.onnx_helper import helper


class Node(object):
    """
    Args:
        op_type (str): Operator type.
        layer_name (str): Name of node, the name of node in graph is unique. 
        inputs (list): Inputs of node with domain=NodeDomain.ONNX, which stored by list.
        outputs (list): Outputs of node with domain=NodeDomain.ONNX, which stored by list 
        attrs (dict): Attributes of node.
        domain (str):  Domain of node.  
    """

    def __init__(self, op_type, layer_name, inputs, outputs, attrs,
                 domain=None):
        self.domain = domain
        self.type = op_type
        self.attrs = attrs
        self.layer_name = layer_name
        self.set_inputs(inputs)
        self.set_outputs(outputs)

    def __hash__(self):
        return hash(self.layer_name)

    def __eq__(self, other):
        if self.layer_name == other.layer_name:
            return True
        return False

    def __str__(self):
        node_str = ''
        if self.domain == NodeDomain.PADDLE:
            self.attrs.pop('op_callstack')
        attrs = ''
        for key, value in self.attrs.items():
            attrs += ', ' + key + '=' + str(value)
        node_str += "  {} = {}::{}(inputs={}{}) \n".format(
            self.outputs, self.domain, self.type, self.inputs, attrs)
        return node_str

    def input(self, idx=None):
        return self.inputs[idx]

    def output(self, name=None, idx=None):
        return self.outputs[idx]

    def attr(self, name):
        if name in self.attrs:
            return self.attrs[name]
        return None

    def set_inputs(self, inputs):
        if isinstance(inputs, list):
            self.inputs = [
                ipt.layer_name if isinstance(ipt, Node) else ipt
                for ipt in inputs
            ]
        else:
            raise TypeError('Inputs of node must be type: list, but got {}'.
                            format(type(inputs)))

    def set_outputs(self, outputs):
        if isinstance(outputs, list):
            self.outputs = [
                opt.layer_name if isinstance(opt, Node) else opt
                for opt in outputs
            ]
        else:
            raise TypeError('Outputs of node must be type: list, but got {}'.
                            format(type(outputs)))


class ONNXNode(Node):
    def __init__(self, op_type, layer_name, inputs, outputs, attrs,
                 domain=None):
        super(ONNXNode, self).__init__(op_type, layer_name, inputs, outputs,
                                       attrs, domain)
        self.node = None
        if domain == NodeDomain.ONNX:
            self.node = self.make_onnx_node()

    def make_onnx_constant_node(self):
        dtype = self.attr('dtype')
        value = self.attr('value')
        if isinstance(value, list):
            dims = (len(value), )
        elif value is None:
            dims = ()
            value = []
        else:
            dims = ()
            value = [value]

        if 'dims' in self.attrs:
            dims = self.attrs['dims']

        tensor = helper.make_tensor(
            name=self.layer_name, data_type=dtype, dims=dims, vals=value)

        onnx_node = helper.make_node(
            self.type, inputs=self.inputs, outputs=self.outputs, value=tensor)

        return onnx_node

    def make_onnx_node(self):
        if self.type in ['Constant', 'ConstantOfShape']:
            onnx_node = self.make_onnx_constant_node()
        else:
            onnx_node = helper.make_node(
                self.type,
                inputs=self.inputs,
                outputs=self.outputs,
                name=self.layer_name,
                **self.attrs)
        return onnx_node


class PaddleNode(Node):
    """
    Args:
        op_type (str): Operator type.
        layer_name (str): Name of node, the name of node in graph is unique. 
        inputs (dict): Inputs of node in node with domain=NodeDomain.PADDLE, which stored by dict. 
        outputs (dict): Outputs of node with domain=NodeDomain.PADDLE, which stored by dict. 
        attrs (dict): Attributes of node.
        block (paddle.fluid.framework.Block): The block that node belongs to. 
        domain (str):  Domain of node.  
    """

    def __init__(self,
                 node,
                 layer_name,
                 inputs,
                 outputs,
                 attrs,
                 block,
                 domain=NodeDomain.PADDLE):
        super(PaddleNode, self).__init__(node.type, layer_name, inputs, outputs,
                                         attrs, domain)
        self.node = node
        self.block = block

    @property
    def input_names(self):
        return [name for name in self.inputs.keys()]

    @property
    def output_names(self):
        return [name for name in self.outputs.keys()]

    def input(self, name, idx=None):
        if idx is None:
            return self.inputs[name]
        return self.inputs[name][idx]

    def output(self, name, idx=None):
        if idx is None:
            return self.outputs[name]
        return self.outputs[name][idx]

    def output_shape(self, name, idx):
        return self.block.var(self.output(name, idx)).shape

    def input_shape(self, name, idx):
        return self.block.var(self.input(name, idx)).shape

    def input_var(self, name, idx):
        return self.block.var(self.input(name, idx))

    def attr(self, name):
        if name in self.attrs:
            return self.attrs[name]
        return None

    def set_inputs(self, inputs):
        if isinstance(inputs, dict):
            # input of node in paddle, which stored by dict 
            self.inputs = inputs
        else:
            raise TypeError('Inputs of node must be type: dict, but got {}'.
                            format(type(inputs)))

    def set_outputs(self, outputs):
        if isinstance(outputs, dict):
            # output of node in paddle, which stored by dict 
            self.outputs = outputs
        else:
            raise TypeError('Outputs of node must be type: dict, but got {}'.
                            format(type(outputs)))
