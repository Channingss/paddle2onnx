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
from six import text_type as _text_type
import argparse
import sys
import os

from paddle2onnx.graph import StaticGraph
from paddle2onnx.converter import Converter
from paddle2onnx.optimizer import GraphOptimizer
from paddle2onnx import utils
import paddle.fluid as fluid


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=_text_type,
        default=None,
        help="define model file path for Paddle")
    parser.add_argument(
        "--mode_type",
        "-s",
        type=_text_type,
        default='program',
        help="path to save translated onnx model")
    parser.add_argument(
        "--save_dir",
        "-s",
        type=_text_type,
        default=None,
        help="path to save translated onnx model")
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        default=False,
        help="get version of paddle2onnx")
    parser.add_argument(
        "--opset_version",
        "-oo",
        type=int,
        default=9,
        help="when paddle2onnx set onnx opset version to export")

    return parser


def export_dygraph(layer,
                   save_dir,
                   input_spec=None,
                   configs=None,
                   opset_version=9):
    output_spec = None
    if configs is not None:
        output_spec = configs.output_spec
    static_graph = StaticGraph.parse_graph(layer, input_spec, output_spec)
    converter = Converter(opset_version)
    onnx_model = converter.convert(static_graph)
    optimizer = GraphOptimizer()
    optimizer.optimize(onnx_model)
    path, file_name = os.path.split(save_dir)
    if path != '' and not os.path.isdir(path):
        os.makedirs(path)
    with open(save_dir, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print("\nONNX model saved in {}".format(save_dir))


def export_program():
    pass


def main():
    if len(sys.argv) < 2:
        print("Use \"paddle2onnx -h\" to print the help information")
        print("For more information, please follow our github repo below:)")
        print("\nGithub: https://github.com/PaddlePaddle/paddle2onnx.git\n")
        return

    parser = arg_parser()
    args = parser.parse_args()

    if args.version:
        import paddle2onnx
        print("paddle2onnx-{} with python>=2.7, paddlepaddle>=1.8.0\n".format(
            paddle2onnx.__version__))
        return

    assert args.save_dir is not None, "--save_dir is not defined"

    try:
        import paddle
        v0, v1, v2 = paddle.__version__.split('.')
        print("paddle.__version__ = {}".format(paddle.__version__))
        if v0 == '0' and v1 == '0' and v2 == '0':
            print("[WARNING] You are use develop version of paddlepaddle")
        elif int(v0) != 1 or int(v1) < 8:
            print("[ERROR] paddlepaddle>=1.8.0 is required")
            return
    except:
        print(
            "[ERROR] paddlepaddle not installed, use \"pip install paddlepaddle\""
        )

    assert args.model is not None, "--model should be defined while translating paddle model to onnx"
    assert args.save_dir is not None, "--save_dir should be defined while translating paddle model to onnx"

    if args.model:
        export_dygraph(args.model, args.save_dir, opset_version=args.onnx_opset)
    else:
        pass
        #export_program(args.model, args.save_dir, opset_version=args.onnx_opset)


if __name__ == "__main__":
    main()