# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import

import os
import six
import numpy as np
from paddle.fluid.framework import Variable
import paddle2onnx.onnx_helper as onnx
from paddle2onnx import utils
from paddle2onnx.constant import PRODUCER
from paddle2onnx.graph import build_graph
from paddle2onnx.mapper.graph_mapper import mapping_graph_to_onnx_proto


def build_graph_from_program(program,
                             feed=None,
                             fetch=None,
                             scope=None,
                             opset_version=None):
    parameters_dict = {}
    vars = program.global_block().vars

    for name in vars:
        var = program.global_block().var(name)
        if name.endswith('feed') or name.endswith('fetch'):
            continue
        if not var.persistable:
            continue
        parameters_dict[name] = {
            'data': np.array(scope.var(name).get_tensor()),
            'dtype': var.dtype,
            'shape': var.shape
        }

    graph = build_graph(program, parameters_dict, opset_version=opset_version)
    return graph


def convert_program_to_onnx(program,
                            scope,
                            save_dir,
                            feeded_var_names=None,
                            target_vars=None,
                            opset_version=9,
                            enable_onnx_checker=False):

    if feeded_var_names is not None:
        if isinstance(feeded_var_names, six.string_types):
            feeded_var_names = [feeded_var_names]
        else:
            if not (bool(feeded_var_names) and all(
                    isinstance(name, six.string_types)
                    for name in feeded_var_names)):
                raise TypeError("'feeded_var_names' should be a list of str.")

    if target_vars is not None:
        if isinstance(target_vars, Variable):
            target_vars = [target_vars]
        else:
            if not (bool(target_vars) and
                    all(isinstance(var, Variable) for var in target_vars)):
                raise TypeError("'target_vars' should be a list of variable.")

    graph = build_graph_from_program(program, feeded_var_names, target_vars,
                                     scope, opset_version)

    onnx_proto = mapping_graph_to_onnx_proto(graph)

    if enable_onnx_checker:
        utils.check_model(onnx_proto)

    path, _ = os.path.split(save_dir)
    if path != '' and not os.path.isdir(path):
        os.makedirs(path)
    with open(save_dir, 'wb') as f:
        f.write(onnx_proto.SerializeToString())

    print("ONNX model saved in {}".format(save_dir))