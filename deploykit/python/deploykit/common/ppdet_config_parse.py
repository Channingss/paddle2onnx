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

import os
import os.path as osp
import yaml
from collections import OrderedDict


def parse_ppdet_config(cfg_file):
    with open(cfg_file) as f:
        yaml_dict = yaml.load(f.read(), Loader=yaml.Loader)

    config = OrderedDict()
    config["model_format"] = "Paddle"
    if "arch" not in yaml_dict:
        raise Exception(
            "Failed to find `arch` in the recieved PaddleDetection yaml file, please check it should be set as one of YOLO/SSD/RetinaNet/EfficientDet/RCNN/Face/TTF/FCOS/SOLOv2."
        )
    config["model_name"] = yaml_dict["arch"]
    config["toolkit"] = "PaddleDetection"
    config["toolkit_version"] = "Unknown"
    if "label_list" not in yaml_dict:
        raise Exception(
            "Failed to find `label_list` in the recieved PaddleDetection yaml file."
        )
    config["labels"] = yaml_dict["label_list"]
    if 'mask_resolution' in yaml_dict:
        config['mask_resolution'] = yaml_dict['mask_resolution']

    # Parse preprocess in paddledetection yaml file
    if "Preprocess" not in yaml_dict:
        raise Exception(
            "Failed to find `Preprocess` in the recieved PaddleDetection yaml file."
        )
    preprocess_info = yaml_dict['Preprocess']
    config["transforms"] = parse_preprocess(preprocess_info)
    return config


def parse_preprocess(preprocess_info):
    transforms = list()
    transforms.append({'BGR2RGB': {}})
    for op_info in preprocess_info:
        op_name = op_info["type"]
        if op_name == "Normalize":
            op_attr = dict()
            op_attr["mean"] = op_info["mean"]
            op_attr["std"] = op_info["std"]
            op_attr["min_val"] = [0] * len(op_info["mean"])
            op_attr["max_val"] = [255.] * len(op_info["mean"])
            op_attr["is_scale"] = op_info["is_scale"]
            transforms.append({'Normalize': op_attr})
        elif op_name == "Permute":
            transforms.append({'Permute': {}})
            if op_info["to_bgr"]:
                transforms.append({'RGB2BRG': {}})
        elif op_name == "Resize":
            max_size = op_info["max_size"]
            if max_size != 0 and config["model_name"] in ["RCNN", "RetinaNet"]:
                op_attr = dict()
                op_attr["target_size"] = op_info["target_size"]
                op_attr["max_size"] = op_info["max_size"]
                op_attr["interp"] = op_info["interp"]
                transforms.append({'ResizeByShort': op_attr})
                if "image_shape" in op_info:
                    op_attr = dict()
                    op_attr['width'] = max_size
                    op_attr['height'] = max_size
                    transforms.append({'Padding': op_attr})
            else:
                op_attr = dict()
                op_attr["width"] = op_info["target_size"]
                op_attr["height"] = op_info["target_size"]
                op_attr["interp"] = op_info["interp"]
                transforms.append({'Resize': op_attr})
        elif op_name == "PadStride":
            op_attr = dict()
            op_attr["stride"] = op_info["stride"]
            transforms.append({'Padding': op_attr})
        else:
            raise Exception("Cannot parse the operation {}.".format(op_name))
    return transforms
