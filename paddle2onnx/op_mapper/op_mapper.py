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

from __future__ import absolute_import
import inspect


class OpMapper(object):
    OPSETS = {}

    def __init__(self, pd_op, **kwargs):
        if not isinstance(pd_op, list):
            pd_op = [pd_op]
        self.pd_op = pd_op
        self.kwargs = kwargs

    def __call__(self, cls):
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k.startswith("opset_"):
                version = int(k.replace("opset_", ""))
                opset_dict = OpMapper.OPSETS[self.name]
                for pd_op in self.name:
                    if pd_op not in OpMapper.OPSETS:
                        OpMapper.OPSETS[self.pd_op] = {}
                    opset_dict[version] = (v, self.kwargs)

    @staticmethod
    def mapping(node, opset_version):
        if node.type not in OpMapper.OPSETS:
            return None
        opsets_mapping = OpMapper.OPSETS[node.type]
        versions = opsets_mapping.keys()
        convert_version = get_max_support_version(versions, opset_version)
        mapper_func, kw = opsets[convert_version]
        onnx_node = mapper_func(op, **kw)
        return onnx_node
