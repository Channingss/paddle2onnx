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

import numpy as np

from .base_preprocessing import BasePreprocessor
from deploykit.common import DataBlob, ShapeInfo


class DetPreprocessor(BasePreprocessor):
    def __init__(self, config):
        super(DetPreprocessor, self).__init__(config)
        self.architecture = self.config['model_name']
        self.build_transforms(self.config['transforms'])

    def __call__(self, img_list):
        shape_info_list = list()
        for img in img_list:
            height, width, _ = img.shape
            shape_info = ShapeInfo()
            shape_info.set_origin_shape(height, width)
            shape_info_list.append(shape_info)

        img_list = self.transform_image(img_list)
        shape_info_list = self.infer_shape(shape_info_list)

        im_data_blob = DataBlob()
        im_data_blob.name = 'image'
        im_data_blob.data = np.asarray(img_list)
        im_data_blob.shape = im_data_blob.data.shape

        inputs = list()
        if self.architecture in ['RCNN', 'RetinaNet', 'SSD']:
            im_shape_blob = DataBlob()
            im_shape_blob.name = 'im_shape'
            im_shape_list = list()
            for shape_info in shape_info_list:
                origin_width, origin_height = shape_info.get_origin_shape()
                im_shape_list.append((origin_height, origin_width, 1))
            im_shape_blob.data = np.asarray(im_shape_list).astype(np.float32)
            if self.architecture == 'SSD':
                return (im_data_blob, im_shape_blob), shape_info_list

            im_info_blob = DataBlob()
            im_info_blob.name = 'im_info'
            im_info_list = list()
            for shape_info in shape_info_list:
                origin_width, origin_height = shape_info.get_origin_shape()
                im_info = (origin_height, origin_width, 1)
                for op_name in list(shape_info.shape.keys()):
                    if 'Resize' in op_name:
                        resized_scale = float(origin_width) / float(
                            shape_info.shape[op_name][0])
                        im_info = shape_info.shape[op_name] + (resized_scale, )
                im_info_list.append(np.asarray(im_info, dtype=np.float32))
            im_info_blob.data = np.concatenate(im_info_list, axis=0)
            return (im_data_blob, im_info_blob, im_shape_blob), shape_info_list
        elif self.architecture in ['YOLO']:
            im_size_blob = DataBlob()
            im_size_blob.name = 'im_size'
            im_size_list = list()
            for shape_info in shape_info_list:
                origin_width, origin_height = shape_info.get_origin_shape()
                im_size_list.append((origin_height, origin_width))
            im_size_blob.data = np.asarray(im_size_list).astype(np.int32)
            return (im_data_blob, im_size_blob), shape_info_list
