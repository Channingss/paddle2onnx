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

from six import text_type as _text_type
import argparse
import sys


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=_text_type,
        default=None,
        help="define model file path for tensorflow or onnx")
    parser.add_argument(
        "--prototxt",
        "-p",
        type=_text_type,
        default=None,
        help="prototxt file of caffe model")
    parser.add_argument(
        "--weight",
        "-w",
        type=_text_type,
        default=None,
        help="weight file of caffe model")
    parser.add_argument(
        "--save_dir",
        "-s",
        type=_text_type,
        default=None,
        help="path to save translated model")
    parser.add_argument(
        "--framework",
        "-f",
        type=_text_type,
        default=None,
        help="define which deeplearning framework(tensorflow/caffe/onnx/paddle2onnx)"
    )
    parser.add_argument(
        "--caffe_proto",
        "-c",
        type=_text_type,
        default=None,
        help="optional: the .py file compiled by caffe proto file of caffe model"
    )
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        default=False,
        help="get version of paddle2onnx")
    parser.add_argument(
        "--without_data_format_optimization",
        "-wo",
        type=_text_type,
        default="True",
        help="tf model conversion without data format optimization")
    parser.add_argument(
        "--define_input_shape",
        "-d",
        action="store_true",
        default=False,
        help="define input shape for tf model")
    parser.add_argument(
        "--onnx_opset",
        "-oo",
        type=int,
        default=10,
        help="when paddle2onnx set onnx opset version to export")
    parser.add_argument(
        "--params_merge",
        "-pm",
        action="store_true",
        default=False,
        help="define whether merge the params")

    return parser


def dg2onnx(model, input_spec, save_dir, inputs_name=None, outputs_name=None, opset_version=10):
    from .decoder.dynamic2static import PaddleDynamicGraphDecoder
    from .op_mapper.paddle2onnx.paddle_op_mapper import PaddleOpMapper
    import paddle.fluid as fluid
    model = PaddleDynamicGraphDecoder(model, input_spec)
    mapper = PaddleOpMapper()
    mapper.convert(
        model.program,
        save_dir,
        opset_version=opset_version)

