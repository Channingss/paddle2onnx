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


class TritonInferOptions(object):
    def __init__(self, 
              model_name,
              model_version="",
              request_id="",
              sequence_id=0,
              sequence_start=False,
              sequence_end=False,
              priority=0,
              timeout=None):
        self.model_name=model_name
        self.model_version=model_version
        self.request_id=request_id
        self.sequence_id=sequence_id
        self.sequence_start=sequence_start
        self.sequence_end=sequence_end
        self.priority=priority
        self.timeout=timeout

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

    def infer(self, infer_options, input_blobs, headers=None, query_params=None):
        from tritonclient.utils import np_to_triton_dtype
        import tritonclient.http as httpclient
        try:
            model_metadata = self.triton_client.get_model_metadata(infer_options.model_name, model_version=infer_options.model_version, headers=headers)
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
            infer_options.model_name, 
            inputs, 
            model_version=infer_options.model_version,
            outputs=request_outputs, 
              request_id=infer_options.request_id,
              sequence_id=infer_options.sequence_id,
              sequence_start=infer_options.sequence_start,
              sequence_end=infer_options.sequence_end,
              priority=infer_options.priority,
              timeout=infer_options.timeout,
            headers=headers,
              query_params=query_params)
        outputs = []
        for output in model_metadata['outputs']:
            output_blob = DataBlob()
            output_blob.name = output['name']
            output_blob.data = results.as_numpy(output['name'])
            print(output_blob.data)
            #output_data_blob.lod = output_tensor.lod()
            outputs.append(output_blob)
        return outputs
