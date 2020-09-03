# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import sys
import paddle2onnx
import os
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
from onnx import helper, onnx_pb

DTYPE_MAP = {
    core.VarDesc.VarType.FP32: onnx_pb.TensorProto.FLOAT,
    core.VarDesc.VarType.FP64: onnx_pb.TensorProto.DOUBLE,
    core.VarDesc.VarType.INT32: onnx_pb.TensorProto.INT32,
    core.VarDesc.VarType.INT16: onnx_pb.TensorProto.INT16,
    core.VarDesc.VarType.INT16: onnx_pb.TensorProto.UINT16,
    core.VarDesc.VarType.INT64: onnx_pb.TensorProto.INT64,
    core.VarDesc.VarType.BOOL: onnx_pb.TensorProto.BOOL,
}

DTYPE_NUMPY_MAP = {
    core.VarDesc.VarType.FP32: 'float32',
    core.VarDesc.VarType.FP64: 'float64',
    core.VarDesc.VarType.INT32: 'int32',
    core.VarDesc.VarType.INT64: 'int64',
    core.VarDesc.VarType.INT16: 'int16',
    core.VarDesc.VarType.INT16: 'uint16',
    core.VarDesc.VarType.BOOL: 'bool',
}

name_counter = dict()

def get_name(op_name, var_name):
    name = 'p2o.{}.{}'.format(op_name, var_name)
    if name not in name_counter:
        name_counter[name] = 0
    else:
        name_counter[name] += 1
    return name + '.{}'.format(name_counter[name])

def make_constant_node(name, dtype, value=None):
    if isinstance(value, list):
        dims = (len(value), )
    elif value is None:
        dims = ()
        value = []
    else:
        dims = ()
        value = [value]
    tensor = helper.make_tensor(
        name=name, data_type=dtype, dims=dims, vals=value)
    node = helper.make_node(
        'Constant', inputs=[], outputs=[name], value=tensor)
    return node

