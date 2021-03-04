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
from deploykit.engine import TritonInferenceEngine, TritonInferOptions
from deploykit.preprocessing import DetPreprocessor
from deploykit.postprocessing import DetPostprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detection prediction using paddle inference engine')
    parser.add_argument(
        '--model_name',
        dest='model_name',
        help='Name of inference model',
        type=str)
    parser.add_argument(
        '--url', dest='url', help='url of triton inference server', type=str)
    parser.add_argument(
        '--model_version',
        dest='model_version',
        default='',
        help='version of inference model',
        type=str)
    parser.add_argument(
        '--http_headers',
        dest='http_headers',
        help='http headers for request to server',
        type=str)
    parser.add_argument(
        '--cfg_file', dest='cfg_file', help='Path of yaml file', type=str)
    parser.add_argument(
        '--pp_type',
        dest='pp_type',
        default='det',
        help='Type of Paddle toolkit',
        type=str)
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
        '--batch_size',
        dest='batch_size',
        help='Batch size of infering',
        default=1,
        type=int)
    return parser.parse_args()


def infer(args):
    parser = ConfigParser(args.cfg_file, args.pp_type)

    if args.http_headers is not None:
        headers_dict = {
            l.split(':')[0]: l.split(':')[1]
            for l in args.http_headers
        }
    else:
        headers_dict = None

    det_preprocess = DetPreprocessor(parser)
    det_postprocess = DetPostprocessor(parser)
    engine = TritonInferenceEngine(args.url)
    image = cv2.imread(args.image)
    inputs, shape_info = det_preprocess([image])
    infer_options = TritonInferOptions(args.model_name, args.model_version)
    outputs = engine.infer(infer_options, inputs)
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
