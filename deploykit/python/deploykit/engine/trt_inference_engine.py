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
import sys 

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from itertools import chain
import argparse

TRT_LOGGER = trt.Logger()

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TensorRTInferOptions(object):
    def __init__(self, ):
        self.dynamic_shape_info = {}

    def set_dynamic_shape_info(input_name, min_shape, opt_shape, max_shape):
        self.dynamic_shape_info[input_name] = [min_shape, opt_shape, max_shape]

def get_input_metadata(network):
    inputs = TensorMetadata()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        inputs.add(name=tensor.name, dtype=trt.nptype(tensor.dtype), shape=tensor.shape)
    return inputs


def get_output_metadata(network):
    outputs = TensorMetadata()
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        outputs.add(name=tensor.name, dtype=trt.nptype(tensor.dtype), shape=tensor.shape)
    return outputs

class TensorRTInferenceEngine(object):
    
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    def __init__(self, model_dir, max_workspace_size=1<<28, trt_cache_file=None, configs=None):
        builder = trt.Builder(TRT_LOGGER)
        #builder.max_batch_size = 4
        config =  builder.create_builder_config()
        print("TensorRT version:", trt.__version__)
        config.max_workspace_size = max_workspace_size

        network = builder.create_network(self.EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, TRT_LOGGER) 

        # Parse model file
        if not os.path.exists(model_dir):
            print('ONNX file {} not found, t.'.format(model_dir))
            exit(0)
        with open(model_dir, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        if configs is not None:
            if len(configs.dynamic_shape_info) != 0:
                profile = builder.create_optimization_profile()
                # profile.set_shape("image", (1, 3, 200, 200), (2, 3, 320, 320), (4, 3, 420, 420)) 
                profile.set_shape("image", (1, 3, 320, 320), (1, 3, 320, 320), (1, 3, 320, 320)) 
                profile.set_shape("im_size", (1, 2), (1, 2), (1, 2)) 
                config.add_optimization_profile(profile)
            else:
                print(network.get_input(0).shape)

        profile = builder.create_optimization_profile()
        # profile.set_shape("image", (1, 3, 200, 200), (2, 3, 320, 320), (4, 3, 420, 420)) 
        profile.set_shape("image", (1, 3, 320, 320), (1, 3, 320, 320), (1, 3, 320, 320)) 
        profile.set_shape("im_size", (1, 2), (1, 2), (1, 2)) 
        config.add_optimization_profile(profile)
        self.engine = self.get_engine(model_dir, trt_cache_file, builder, network, config)
        self.input_names = []
        self.output_names = []
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                self.input_names.append(binding)
            else:
                self.output_names.append(binding)

    def is_dynamic_input_shape(shape):
        for index, dim in enumerate(shape):
            if dim < 0 and index > 0:
                return True
        return False

    def is_dynamic_batch_size(shape):
        if shape[0] == -1:
            return True
        return False

    def allocate_buffers(self, engine, context):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for i, binding in enumerate(engine):
            print(context.get_binding_shape(i))
            size = trt.volume(context.get_binding_shape(i)) 
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference(self, context, bindings, inputs, outputs, stream, batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]
    
    # This function is generalized for multiple inputs/outputs for full dimension networks.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference_v2(self, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def get_engine(self, onnx_file_path, engine_file_path, builder, network, config):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        def build_engine():
            """Takes an ONNX file and creates a TensorRT engine to run inference with"""
            #network.get_input(0).shape = [1, 3, 320, 320]
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_engine(network, config)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()

    def infer(self, input_blobs):
        # Do inference
        context = self.engine.create_execution_context()

        #input_blobs[0].data = np.concatenate([input_blobs[0].data,input_blobs[0].data], axis=0)
        input_blobs[0].data = input_blobs[0].data
        for i, binding_name in enumerate(self.engine):
            if self.engine.binding_is_input(binding_name):
                binding_index = self.engine.get_binding_index(binding_name)
                context.set_binding_shape(self.engine[binding_name], input_blobs[i].data.shape)

        assert context.all_binding_shapes_specified

        inputs, outputs, bindings, stream = self.allocate_buffers(self.engine, context)

        for i in range(len(inputs)):
            data = input_blobs[i].data.ravel()
            np.copyto(inputs[i].host, data)

        trt_outputs = self.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        output_blobs = []
        for index, binding_name in enumerate(self.output_names):
            binding_index = self.engine.get_binding_index(binding_name)
            output_shape = context.get_binding_shape(binding_index) 
            output_blob = DataBlob()
            output_blob.name = binding_name 
            output_blob.data =  trt_outputs[index].reshape(output_shape)
            print(output_blob.data)
            #output_data_blob.lod = output_tensor.lod()
            output_blobs.append(output_blob)

        return output_blobs
