# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict


class DataBlob(object):
    def __init__(self):
        self.data = None
        self.name = None
        self.shape = None
        self.lod = None


class ShapeInfo(object):
    def __init__(self):
        self.shape = OrderedDict()

    def set_origin_shape(self, height, width):
        self.shape['Origin'] = (width, height)

    def get_origin_shape(self):
        return self.shape['Origin']

    def get_last_shape(self):
        if len(self.shape) > 0:
            last_op_name = list(self.shape.keys())[-1]
            last_shape = self.shape[last_op_name]
            return last_shape
        else:
            raise Exception(
                "Cannot get the last shape, because any shape information is not recorded."
            )
