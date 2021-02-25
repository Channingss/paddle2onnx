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
import numpy as np
from deploykit.common import DataBlob

#def numpy_triton_dtype_mapping(np_dtype):
#    if np_dtype
NUMPY_TRTITON_DTYPE_MAPPER = {
    np.dtype('float32'): "FP32",
    np.dtype('float64'): "FP64",
    np.dtype('float32'): "FP32",
    np.dtype('float16'): "FP16",
    np.dtype('int64'): "INT64",
    np.dtype('int32'): "INT32",
    np.dtype('int16'): "INT16",
    np.dtype('bool'): "BOOL",
}


class TritonInferenceEngine(object):
    def __init__(self, url, ssl=False, verbose=False):
        import tritonclient.http as httpclient
        from tritonclient.utils import InferenceServerException
        import gevent.ssl
        try:
            if ssl:
                triton_client = httpclient.InferenceServerClient(
                    url=url,
                    verbose=verbose,
                    ssl=True,
                    ssl_context_factory=gevent.ssl._create_unverified_context,
                    insecure=True)
            else:
                triton_client = httpclient.InferenceServerClient(
                    url=url, verbose=verbose)
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit(1)
        self.triton_client = triton_client

    def infer(self, model_name, model_version, headers, input_blobs):
        from tritonclient.utils import np_to_triton_dtype
        import tritonclient.http as httpclient
        try:
            model_metadata = self.triton_client.get_model_metadata(
                model_name=model_name, model_version=model_version)
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)
        inputs = []
        request_outputs = []
        for data_blob in input_blobs:
            input = httpclient.InferInput(
                data_blob.name, data_blob.data.shape,
                np_to_triton_dtype(data_blob.data.dtype))
            input.set_data_from_numpy(data_blob.data, binary_data=False)
            inputs.append(input)
        for output in model_metadata['outputs']:
            request_outputs.append(
                httpclient.InferRequestedOutput(
                    output['name'], binary_data=False))
        results = self.triton_client.infer(
            model_name, inputs, outputs=request_outputs, headers=headers)
        outputs = []
        for output in model_metadata['outputs']:
            output_blob = DataBlob()
            output_blob.name = output['name']
            output_blob.data = results.as_numpy(output['name'])
            print(output_blob.data)
            #output_data_blob.lod = output_tensor.lod()
            outputs.append(output_blob)
        return outputs
