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

import cv2
import argparse
import os
import os.path as osp

from deploykit.common import ConfigParser
from deploykit.engine import PaddleInferenceConfig, PaddleInferenceEngine
from deploykit.preprocessing import DetPreprocessor
from deploykit.postprocessing import DetPostprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detection prediction using paddle inference engine')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='Path of inference model',
        type=str)
    parser.add_argument(
        '--cfg_file', dest='cfg_file', help='Path of yaml file', type=str)
    parser.add_argument(
        '--pp_type', dest='pp_type', help='Type of Paddle toolkit', type=str)
    parser.add_argument(
        '--image',
        dest='image',
        help='Path of yaml fileth of test image file',
        type=str)
    parser.add_argument(
        '--image_list',
        dest='image_list',
        help='Path of test image list file',
        type=str)
    parser.add_argument(
        '--use_gpu',
        dest='use_gpu',
        help='Infering with GPU or CPU',
        default=False,
        type=bool)
    parser.add_argument(
        '--gpu_id', dest='gpu_id', help='GPU card id', default=0, type=int)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Batch size of infering',
        default=1,
        type=int)
    return parser.parse_args()


def infer(args):
    parser = ConfigParser(args.cfg_file, args.pp_type)
    det_preprocess = DetPreprocessor(parser)
    det_postprocess = DetPostprocessor(parser)
    engine_config = PaddleInferenceConfig()
    engine = PaddleInferenceEngine(args.model_dir, engine_config)

    image = cv2.imread(args.image)
    inputs, shape_info = det_preprocess([image])
    outputs = engine.infer(inputs)
    det_results = det_postprocess(outputs, shape_info)
    for det_result in det_results:
        for bbox in det_result.bboxes:
            if bbox.score < 0.5:
                continue
            print(
                'class_id: {}, confidence: {:.04f}, left_top:[{:.02f},{:.02f}], right_bottom:[{:.02f},{:.02f}]'.
                format(bbox.category_id, bbox.score, bbox.coordinate.xmin,
                       bbox.coordinate.ymin, bbox.coordinate.xmax,
                       bbox.coordinate.ymax))


if __name__ == '__main__':
    args = parse_args()
    infer(args)
