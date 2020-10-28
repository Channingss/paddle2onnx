#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import importlib


def try_import(module_name):
    """Try importing a module, with an informative error message on failure."""
    install_name = module_name
    try:
        mod = importlib.import_module(module_name)
        return mod
    except ImportError:
        err_msg = (
            "Failed importing {}. This likely means that some modules "
            "requires additional dependencies that have to be "
            "manually installed (usually with `pip install {}`). ").format(
                module_name, install_name)
        raise ImportError(err_msg)


def check_model(onnx_model):
    onnx = try_import('onnx')
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        raise Exception('ONNX model is not valid.')
    finally:
        print('ONNX model genarated is valid.')