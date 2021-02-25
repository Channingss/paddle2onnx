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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
import numpy as np

from deploykit.common.config import ConfigParser
import deploykit.common.transform as T
from deploykit.common.transform import Padding


class BasePreprocessor(object):
    def __init__(self, config):
        if not isinstance(config, ConfigParser):
            raise TypeError(
                "Type of config must be ConfigParser, but recieved is {}".
                format(type(config)))

        self.config = config.config
        self.padding = Padding(height=1, width=1)

    def build_transforms(self, op_info_list):
        if not isinstance(op_info_list, Sequence):
            raise TypeError(
                "op_info_list should be list or tuple, but recieved is {}".
                format(type(op_info_list)))

        if len(op_info_list) < 1:
            raise ValueError(
                "The length of op_info_list should not be less than 1, but recieved is {}".
                format(len(op_info_list)))

        self.transforms = list()
        for op_info in op_info_list:
            op_name = list(op_info.keys())[0]
            op_attr = op_info[op_name]
            if not hasattr(T, op_name):
                raise Exception(
                    "There's no implementation for the transform operator named '{}'".
                    format(op_name))
            self.transforms.append(getattr(T, op_name)(**op_attr))
        return self.transforms

    def transform_image(self, img_list):
        if len(img_list) > 1:
            batch_img = list()
            for img in img_list:
                img = img.astype(np.float32)
                for op in self.transforms:
                    img = op(img)
                batch_img.append(img)

            batch_img = self.pad_batch_img(batch_img)

            return batch_img
        else:
            img_list[0] = img_list[0].astype(np.float32)
            for op in self.transforms:
                img_list[0] = op(img_list[0])
            return img_list

    def infer_shape(self, shape_info_list):
        if len(shape_info_list) > 1:
            batch_shape_info = list()
            for shape_info in shape_info_list:
                for op in self.transforms:
                    shape_info = op.infer_shape(shape_info)
                batch_shape_info.append(shape_info)
            batch_shape_info = self.pad_shape_info(batch_shape_info)
            return batch_shape_info

        else:
            for op in self.transforms:
                shape_info_list[0] = op.infer_shape(shape_info_list[0])
            return shape_info_list

    def pad_batch_image(self, img_list):
        height_list = [img.shape[0] for img in img_list]
        width_list = [img.shape[1] for img in img_list]
        max_height = max(height_list)
        max_width = max(width_list)
        self.padding.height = max_height
        self.padding.width = max_width
        batch_img = list()
        for img in img_list:
            img = self.padding(img)
            batch_img.append(img)
        return batch_img

    def pad_batch_shape(self, shape_info_list):
        height_list = list()
        width_list = list()
        for shape_info in shape_info_list:
            last_op_name = list(shape_info.keys())[-1]
            last_shape = shape_info[last_op_name]
            height_list.append(last_shape[1])
            width_list.append(last_shape[0])
        max_height = max(height_list)
        max_width = max(width_list)
        batch_shape_info = list()
        for shape_info in shape_info_list:
            shape_info = self.padding.infer_shape(shape_info)
            batch_shape_info.append(shape_info)
        return batch_shape_info
